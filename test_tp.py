import os
import socket
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from config import ModelConfig
from initialize import (
    init_distributed,
    set_seed,
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
)
from model import Transformer

DTYPE = torch.float32
TP_SIZE = 4

BASE_KWARGS = dict(
    hidden_size=128,
    num_attention_heads=8,
    intermediate_size=256,
    num_layers=2,
    sequence_length=64,
    batch_size=2,
)


@dataclass
class RunResult:
    init_weights: dict[str, torch.Tensor]
    output: torch.Tensor
    input_grad: torch.Tensor
    weight_grads: dict[str, torch.Tensor]


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def enable_deterministic() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def make_input(
    seq_len: int, batch_size: int, hidden_size: int, seed: int = 1234
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn(seq_len, batch_size, hidden_size, generator=gen, dtype=DTYPE)


def _all_gather_dim0(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    out = torch.empty(
        (tensor.shape[0] * world_size, *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_gather_into_tensor(out, tensor.contiguous(), group=group)
    return out


def _gather_tp_tensor(
    name: str, tensor: torch.Tensor, tp_size: int, group: dist.ProcessGroup
) -> torch.Tensor:
    shards = [torch.empty_like(tensor) for _ in range(tp_size)]
    dist.all_gather(shards, tensor.contiguous(), group=group)

    if "q_proj" in name or "k_proj" in name or "v_proj" in name or "fc1" in name:
        return torch.cat(shards, dim=0)
    elif "out_proj" in name or "fc2" in name:
        return torch.cat(shards, dim=1)
    else:
        return tensor


def _gather_params(model: Transformer, config: ModelConfig) -> dict[str, torch.Tensor]:
    group = get_tp_group()
    tp_size = get_tp_world_size()
    result = {}
    for name, p in model.named_parameters():
        if config.use_tp:
            result[name] = _gather_tp_tensor(name, p.data, tp_size, group).cpu()
        else:
            # dummy all_gather to keep collectives symmetric
            shards = [torch.empty_like(p.data) for _ in range(tp_size)]
            dist.all_gather(shards, p.data.contiguous(), group=group)
            result[name] = p.data.cpu()
    return result


def run_model(config: ModelConfig) -> RunResult:
    rank = get_tp_rank()
    group = get_tp_group()
    tp_size = get_tp_world_size()

    model = Transformer(config).cuda()
    init_weights = _gather_params(model, config)

    full_input = make_input(
        config.sequence_length, config.batch_size, config.hidden_size
    ).cuda()

    if config.use_tp:
        x = full_input.chunk(tp_size, dim=0)[rank].clone().requires_grad_(True)
    else:
        x = full_input.clone().requires_grad_(True)

    out = model(x)
    # CE loss: treat hidden_size as vocab_size, output [s, b, h] as logits
    vocab_size = config.hidden_size
    seq_len = out.shape[0]
    labels = torch.randint(
        0, vocab_size, (seq_len, config.batch_size), device=out.device
    )
    loss = F.cross_entropy(out.reshape(-1, vocab_size), labels.reshape(-1))
    loss.backward()

    if config.use_tp:
        model.allreduce_replicated_grads()
        output_full = _all_gather_dim0(out, group)
        input_grad_full = _all_gather_dim0(x.grad, group)
    else:
        output_full = out
        input_grad_full = x.grad

    weight_grads = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if config.use_tp:
            weight_grads[name] = _gather_tp_tensor(name, p.grad, tp_size, group).cpu()
        else:
            weight_grads[name] = p.grad.cpu()

    return RunResult(
        init_weights=init_weights,
        output=output_full.detach().cpu(),
        input_grad=input_grad_full.detach().cpu(),
        weight_grads=weight_grads,
    )


def _diff_msg(
    label: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float
) -> str:
    diff = (actual - expected).abs()
    residual_for_atol = (diff - rtol * expected.abs()).clamp(min=0)
    needed_atol = residual_for_atol.max().item()
    nonzero_mask = expected.abs() > 0
    if nonzero_mask.any():
        residual_for_rtol = (diff - atol).clamp(min=0)
        needed_rtol = (
            (residual_for_rtol[nonzero_mask] / expected.abs()[nonzero_mask])
            .max()
            .item()
        )
    else:
        needed_rtol = float("inf")
    return (
        f"Mismatch: {label}. "
        f"max diff={diff.max().item():.6e}, num diffs={diff.ne(0).sum().item()}/{diff.numel()}, "
        f"need atol>={needed_atol:.6e} (with current rtol={rtol}), "
        f"need rtol>={needed_rtol:.6e} (with current atol={atol})"
    )


def assert_results_match(
    a: RunResult,
    b: RunResult,
    atol: float = 0.0,
    rtol: float = 0.0,
    compare_grads: bool = True,
) -> None:
    passed: list[str] = []
    failed: list[str] = []

    def _check(
        label: str, x: torch.Tensor, y: torch.Tensor, a: float, r: float
    ) -> None:
        if torch.allclose(x, y, atol=a, rtol=r):
            passed.append(label)
        else:
            failed.append(_diff_msg(label, x, y, a, r))

    def _flush(stage: str) -> None:
        nonlocal passed, failed
        if failed:
            summary = (
                f"\n  [{stage}]"
                f"\n  PASSED ({len(passed)}): {', '.join(passed) if passed else '(none)'}"
                f"\n  FAILED ({len(failed)}):\n    " + "\n    ".join(failed)
            )
            raise AssertionError(summary)
        passed.clear()
        failed.clear()

    for name in a.init_weights:
        _check(f"{name}", a.init_weights[name], b.init_weights[name], 0.0, 0.0)
    _flush("init weights")

    _check("forward output", a.output, b.output, atol, rtol)
    _flush("forward output")

    if compare_grads:
        _check("input grad", a.input_grad, b.input_grad, atol, rtol)
        for name in a.weight_grads:
            _check(f"{name}", a.weight_grads[name], b.weight_grads[name], atol, rtol)
        _flush("gradients")


def _worker_fn(
    rank: int,
    world_size: int,
    config_a: ModelConfig,
    config_b: ModelConfig,
    shared: dict[str, Any],
    port: int,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    enable_deterministic()
    init_distributed()

    set_seed()
    rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.random.get_rng_state()
    result_a = run_model(config_a)

    torch.random.set_rng_state(rng_state)
    torch.cuda.random.set_rng_state(cuda_rng_state)
    result_b = run_model(config_b)

    if rank == 0:
        shared["result_a"] = result_a
        shared["result_b"] = result_b

    dist.destroy_process_group()


def _run_comparison(
    config_a: ModelConfig,
    config_b: ModelConfig,
    world_size: int,
    atol: float = 0.0,
    rtol: float = 0.0,
    compare_grads: bool = True,
) -> None:
    manager = mp.Manager()
    shared = manager.dict()

    mp.spawn(
        _worker_fn,
        args=(world_size, config_a, config_b, shared, get_free_port()),
        nprocs=world_size,
        join=True,
    )

    assert_results_match(
        shared["result_a"],
        shared["result_b"],
        atol=atol,
        rtol=rtol,
        compare_grads=compare_grads,
    )


def test_tp_vs_no_tp():
    _run_comparison(
        ModelConfig(**BASE_KWARGS, tp_size=1, use_tp=False, use_overlap=False),
        ModelConfig(**BASE_KWARGS, tp_size=TP_SIZE, use_tp=True, use_overlap=False),
        world_size=TP_SIZE,
        atol=1e-3,
        rtol=1e-5,
        compare_grads=False,
    )


def test_overlap_vs_no_overlap():
    _run_comparison(
        ModelConfig(**BASE_KWARGS, tp_size=TP_SIZE, use_tp=True, use_overlap=False),
        ModelConfig(**BASE_KWARGS, tp_size=TP_SIZE, use_tp=True, use_overlap=True),
        world_size=TP_SIZE,
        atol=0.0,
        rtol=0.0,
        compare_grads=True,
    )
