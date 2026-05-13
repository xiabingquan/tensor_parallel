import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from initialize import get_tp_group, get_tp_rank, get_tp_world_size, init_weight_cpu


def _all_gather(input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """AllGather along dim=0: [s/n, b, h] -> [s, b, h]."""

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_
    output = torch.empty(
        (input_.shape[0] * world_size, *input_.shape[1:]),
        dtype=input_.dtype,
        device=input_.device,
    )
    dist.all_gather_into_tensor(output, input_.contiguous(), group=group)
    return output


def _reduce_scatter(input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """ReduceScatter along dim=0: [s, b, h] -> [s/n, b, h]."""

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_
    chunk_size = input_.shape[0] // world_size
    output = torch.empty(
        (chunk_size, *input_.shape[1:]),
        dtype=input_.dtype,
        device=input_.device,
    )
    dist.reduce_scatter_tensor(output, input_.contiguous(), group=group)
    return output


class _ColumnParallelLinearFn(torch.autograd.Function):
    """Forward: AllGather input -> GEMM.  Backward: GEMM -> ReduceScatter."""

    @staticmethod
    def forward(ctx, input_, weight, group):
        full_input = _all_gather(input_, group)
        output = F.linear(full_input, weight)
        ctx.save_for_backward(full_input, weight)
        ctx.group = group
        return output

    @staticmethod
    def backward(ctx, grad_output):
        full_input, weight = ctx.saved_tensors
        group = ctx.group

        grad_input_full = grad_output.matmul(weight)
        grad_input = _reduce_scatter(grad_input_full, group)

        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        full_input_2d = full_input.reshape(-1, full_input.shape[-1])
        grad_weight = grad_output_2d.t().matmul(full_input_2d)

        return grad_input, grad_weight, None


class _RowParallelLinearFn(torch.autograd.Function):
    """Forward: GEMM -> ReduceScatter.  Backward: AllGather grad -> GEMM."""

    @staticmethod
    def forward(ctx, input_, weight, group):
        output_partial = F.linear(input_, weight)
        output = _reduce_scatter(output_partial, group)
        ctx.save_for_backward(input_, weight)
        ctx.group = group
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        group = ctx.group

        grad_output_full = _all_gather(grad_output, group)
        grad_input = grad_output_full.matmul(weight)

        grad_output_2d = grad_output_full.reshape(-1, grad_output_full.shape[-1])
        input_2d = input_.reshape(-1, input_.shape[-1])
        grad_weight = grad_output_2d.t().matmul(input_2d)

        return grad_input, grad_weight, None


def _ag_gemm_overlap(local_input, weight, group):
    """AllGather + GEMM overlap via ring P2P exchange.

    Args:
        local_input: This rank's input shard, shape [s/n, b, h].
        weight: Weight matrix for F.linear.
        group: TP process group.

    Notes:
        Each iteration: wait prev P2P -> dispatch next P2P -> GEMM.
        NCCL P2P runs on internal high-priority stream, overlapping with GEMM.

    Returns:
        Tuple of (gemm_output, gathered_input):
            gemm_output: [s, b, out_dim], chunks ordered by rank.
            gathered_input: [s, b, h], the full AllGathered input.
    """

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    if world_size == 1:
        return F.linear(local_input, weight), local_input

    input_chunks: list[torch.Tensor | None] = [None] * world_size
    output_chunks: list[torch.Tensor | None] = [None] * world_size
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size

    recv_bufs = [torch.empty_like(local_input), torch.empty_like(local_input)]
    recv_bufs[-1].copy_(local_input)
    reqs = []

    for step in range(world_size):
        # 1. Wait for previous round's P2P
        for r in reqs:
            r.wait()

        # 2. Dispatch this round's P2P (except last step)
        send_buf = recv_bufs[(step - 1) % 2]
        recv_buf = recv_bufs[step % 2]
        if step < world_size - 1:
            reqs = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.irecv, recv_buf, prev_rank, group),
                    dist.P2POp(dist.isend, send_buf, next_rank, group),
                ]
            )

        # 3. GEMM on current chunk (overlaps with P2P above)
        src_rank = (rank - step) % world_size
        input_chunks[src_rank] = send_buf.clone()
        output_chunks[src_rank] = F.linear(input_chunks[src_rank], weight)

    gemm_output = torch.cat(output_chunks, dim=0)
    gathered_input = torch.cat([input_chunks[i] for i in range(world_size)], dim=0)
    return gemm_output, gathered_input


def _gemm_rs_overlap(input_, weight, group):
    """GEMM + ReduceScatter overlap.

    Args:
        input_: Full input tensor, shape [s, b, h_in].
        weight: Weight matrix for F.linear.
        group: TP process group.

    Notes:
        Overlap flow: GEMM_i → wait reduce_{i-1} → dispatch reduce_i → GEMM_{i+1} → ...
        So GEMM_{i+1} and reduce_i run in parallel on compute stream and NCCL stream.

    Warnings:
        prev_out keeps the previous gemm_out tensor alive during its async reduce.
        Without it, Python refcount GC would free the buffer when gemm_out is
        reassigned, while NCCL is still reading from it.

    Returns:
        This rank's output portion, shape [s/n, b, h_out].

    """

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if world_size == 1:
        return F.linear(input_, weight)

    input_chunks = input_.chunk(world_size, dim=0)
    my_output: torch.Tensor | None = None
    req: dist.Work | None = None
    prev_out: torch.Tensor | None = None

    for i, inp_chunk in enumerate(input_chunks):
        # GEMM (overlaps with previous chunk's reduce on NCCL stream)
        gemm_out = F.linear(inp_chunk, weight)

        # Wait for previous reduce before dispatching new one
        if req is not None:
            req.wait()
            prev_out = None

        # Dispatch this chunk's reduce
        req = dist.reduce(gemm_out, dst=i, group=group, async_op=True)
        prev_out = gemm_out

        if i == rank:
            my_output = gemm_out

    # Wait for last reduce
    if req is not None:
        req.wait()

    return my_output


class _ColumnParallelOverlapFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, group):
        output, full_input = _ag_gemm_overlap(input_, weight, group)
        ctx.save_for_backward(full_input, weight)
        ctx.group = group
        return output

    @staticmethod
    def backward(ctx, grad_output):
        full_input, weight = ctx.saved_tensors
        group = ctx.group
        grad_input = _gemm_rs_overlap(grad_output, weight.t(), group)
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        full_input_2d = full_input.reshape(-1, full_input.shape[-1])
        grad_weight = grad_output_2d.t().matmul(full_input_2d)

        return grad_input, grad_weight, None


class _RowParallelOverlapFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, group):
        output = _gemm_rs_overlap(input_, weight, group)
        ctx.save_for_backward(input_, weight)
        ctx.group = group
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        group = ctx.group
        grad_input, grad_full = _ag_gemm_overlap(grad_output, weight.t(), group)
        grad_full_2d = grad_full.reshape(-1, grad_full.shape[-1])
        input_2d = input_.reshape(-1, input_.shape[-1])
        grad_weight = grad_full_2d.t().matmul(input_2d)

        return grad_input, grad_weight, None


class ColumnParallelLinear(nn.Module):
    """Linear layer with weight sharded along output dim (dim=0).

    Forward: AllGather input [s/n,b,h] -> [s,b,h], GEMM -> [s,b,h'/n].
    With use_overlap=True, uses ring P2P AllGather+GEMM pipeline.
    """

    def __init__(
        self, in_features: int, out_features: int, use_overlap: bool = False
    ) -> None:
        super().__init__()
        group = get_tp_group()
        tp_size = get_tp_world_size()
        tp_rank = get_tp_rank()

        assert out_features % tp_size == 0
        self.group = group
        self.use_overlap = use_overlap

        shard = init_weight_cpu(
            full_shape=(out_features, in_features),
            shard_dim=0,
            rank=tp_rank,
            world_size=tp_size,
        )
        self.weight = nn.Parameter(shard)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.use_overlap:
            return _ColumnParallelOverlapFn.apply(input_, self.weight, self.group)
        return _ColumnParallelLinearFn.apply(input_, self.weight, self.group)


class RowParallelLinear(nn.Module):
    """Linear layer with weight sharded along input dim (dim=1).

    Forward: GEMM on [s,b,h/n] -> partial sum [s,b,h], ReduceScatter -> [s/n,b,h].
    With use_overlap=True, uses chunked GEMM+ReduceScatter pipeline.
    """

    def __init__(
        self, in_features: int, out_features: int, use_overlap: bool = False
    ) -> None:
        super().__init__()
        group = get_tp_group()
        tp_size = get_tp_world_size()
        tp_rank = get_tp_rank()

        assert in_features % tp_size == 0
        self.group = group
        self.use_overlap = use_overlap

        shard = init_weight_cpu(
            full_shape=(out_features, in_features),
            shard_dim=1,
            rank=tp_rank,
            world_size=tp_size,
        )
        self.weight = nn.Parameter(shard)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.use_overlap:
            return _RowParallelOverlapFn.apply(input_, self.weight, self.group)
        return _RowParallelLinearFn.apply(input_, self.weight, self.group)
