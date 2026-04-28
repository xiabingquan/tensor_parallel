# tensor_parallel

A toy implementation of Tensor Parallel (TP) + Sequence Parallel (SP) with comm-compute overlap.

## Quick Start

```bash
# run unit tests (needs >= 4 GPUs)
bash run_tests.sh

# run profiling (no TP / TP / TP+overlap, needs >= 4 GPUs)
bash run_profile.sh
```

## Code Structure

| File | Description |
|---|---|
| `config.py` | `ModelConfig` dataclass |
| `initialize.py` | Distributed init, global seed, CPU weight init |
| `parallel_linear.py` | `ColumnParallelLinear`, `RowParallelLinear`, overlap variants, comm helpers |
| `model.py` | `RMSNorm`, `Attention`, `MLP`, `TransformerBlock`, `Transformer` |
| `test_tp.py` | Unit tests (TP vs no-TP, overlap vs no-overlap) |
| `profile_memory.py` | Training loop profiling (loss, grad norm, step time, peak memory) |

## Key Implementation Details

- **TP Linear**: Custom `autograd.Function` per layer. Forward saves the AllGathered input for backward reuse. `use_overlap` flag switches between basic and overlap autograd Functions within the same Module.
- **Overlap AG+GEMM**: Ring P2P exchange (`dist.batch_isend_irecv`) with pipelined GEMM on dual CUDA streams.
- **Overlap GEMM+RS**: Chunked GEMM with per-chunk `dist.reduce(dst=i)` (not `reduce_scatter`, which would scatter at the wrong granularity).
- **RMSNorm grads**: Replicated weights need `AllReduce` after backward since each rank only sees `s/n` positions.
- **Weight init**: Global `torch.manual_seed` → all ranks draw from the same RNG in the same order → sharded weights are consistent with the full model.
