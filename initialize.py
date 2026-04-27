
import os

import torch
import torch.distributed as dist

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set global random seed for reproducibility. Call once at startup."""
    torch.manual_seed(seed)


def init_distributed() -> None:
    """Initialize torch.distributed with NCCL backend."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)


def get_tp_group() -> dist.ProcessGroup:
    """Return the TP process group (equals world group in this toy implementation)."""
    return dist.group.WORLD


def get_tp_rank() -> int:
    return dist.get_rank()


def get_tp_world_size() -> int:
    return dist.get_world_size()


def init_weight_cpu(
    full_shape: tuple[int, ...],
    shard_dim: int,
    rank: int,
    world_size: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Initialize weight from the global RNG, return the shard for *rank*.

    All ranks must call this in the same order so they draw from
    the same global RNG state and get identical full weights.
    """
    full_weight = torch.randn(full_shape, dtype=dtype)
    return full_weight.chunk(world_size, dim=shard_dim)[rank].contiguous()
