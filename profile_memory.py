import os
import time
import socket
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from config import ModelConfig
from initialize import init_distributed, set_seed, get_tp_world_size
from model import Transformer

TP_SIZE = 4
NUM_STEPS = 100

MODEL_KWARGS = dict(
    hidden_size=512,
    num_attention_heads=8,
    intermediate_size=1024,
    num_layers=4,
    sequence_length=256,
    batch_size=4,
)


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@dataclass
class StepStats:
    loss: float
    grad_norm: float
    step_time_ms: float
    memory_mb: float


def _compute_grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.float().norm().item() ** 2
    return total**0.5


def _profile_worker(
    rank: int,
    world_size: int,
    config: ModelConfig,
    label: str,
    shared: dict,
    port: int,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    set_seed()
    init_distributed()

    tp_size = get_tp_world_size()
    model = Transformer(config).to(torch.bfloat16).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    if config.use_tp:
        sp_seq_len = config.sequence_length // tp_size
    else:
        sp_seq_len = config.sequence_length

    stats: list[dict] = []

    for step in range(NUM_STEPS):
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(
            sp_seq_len,
            config.batch_size,
            config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )

        t0 = time.perf_counter()
        out = model(x)
        vocab_size = config.hidden_size
        labels = torch.randint(
            0, vocab_size, (sp_seq_len, config.batch_size), device="cuda"
        )
        loss = torch.nn.functional.cross_entropy(
            out.reshape(-1, vocab_size), labels.reshape(-1)
        )
        loss.backward()
        if config.use_tp:
            model.allreduce_replicated_grads()
        optimizer.step()
        optimizer.zero_grad()
        # torch.cuda.synchronize()
        t1 = time.perf_counter()

        grad_norm = _compute_grad_norm(model)
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

        stats.append(
            dict(
                loss=loss.item(),
                grad_norm=grad_norm,
                step_time_ms=(t1 - t0) * 1000,
                memory_mb=peak_mem,
            )
        )

    if rank == 0:
        shared[label] = stats

    dist.destroy_process_group()


def run_profile(label: str, config: ModelConfig, world_size: int) -> list[dict]:
    manager = mp.Manager()
    shared = manager.dict()

    mp.spawn(
        _profile_worker,
        args=(world_size, config, label, shared, get_free_port()),
        nprocs=world_size,
        join=True,
    )

    return list(shared[label])


def plot_results(all_results: dict[str, list[dict]], output_path: str) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("loss", "Loss"),
        ("grad_norm", "Grad Norm"),
        ("step_time_ms", "Step Time (ms)"),
        ("memory_mb", "Peak Memory (MB)"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))

    for ax, (key, title) in zip(axes, metrics):
        for label, stats in all_results.items():
            values = [s[key] for s in stats]
            ax.plot(range(len(values)), values, label=label, alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def main():
    configs = {
        "no TP": (
            ModelConfig(**MODEL_KWARGS, tp_size=1, use_tp=False, use_overlap=False),
            1,
        ),
        "TP": (
            ModelConfig(
                **MODEL_KWARGS, tp_size=TP_SIZE, use_tp=True, use_overlap=False
            ),
            TP_SIZE,
        ),
        "TP + overlap": (
            ModelConfig(**MODEL_KWARGS, tp_size=TP_SIZE, use_tp=True, use_overlap=True),
            TP_SIZE,
        ),
    }

    all_results: dict[str, list[dict]] = {}
    for label, (config, world_size) in configs.items():
        print(f"\n{'=' * 60}")
        print(f"  Profiling: {label} (world_size={world_size})")
        print(f"{'=' * 60}")
        all_results[label] = run_profile(label, config, world_size)

        stats = all_results[label]
        avg_time = sum(s["step_time_ms"] for s in stats[10:]) / len(stats[10:])
        avg_mem = sum(s["memory_mb"] for s in stats[10:]) / len(stats[10:])
        print(f"  Avg step time (after warmup): {avg_time:.1f} ms")
        print(f"  Avg peak memory: {avg_mem:.1f} MB")
        print(f"  Final loss: {stats[-1]['loss']:.4f}")

    plot_results(all_results, "assets/profile.png")


if __name__ == "__main__":
    main()
