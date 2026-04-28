# tensor_parallel

[English](README.md)

Tensor Parallel (TP) + Sequence Parallel (SP) 的 toy 实现，含通信计算重叠（overlap）。

## 快速开始

```bash
# 跑单元测试（需要 >= 4 张 GPU）
bash run_tests.sh

# 跑 profiling：不开 TP / 开 TP / 开 TP+overlap（需要 >= 4 张 GPU）
bash run_profile.sh
```

## 代码结构

| 文件 | 说明 |
|---|---|
| `config.py` | `ModelConfig` 配置类 |
| `initialize.py` | 分布式初始化、全局随机种子、CPU 权重初始化 |
| `parallel_linear.py` | `ColumnParallelLinear`、`RowParallelLinear`、overlap 实现、通信辅助函数 |
| `model.py` | `RMSNorm`、`Attention`、`MLP`、`TransformerBlock`、`Transformer` |
| `test_tp.py` | 单元测试（TP vs 非 TP、overlap vs 非 overlap） |
| `profile_memory.py` | 训练 profiling（loss、grad norm、step time、峰值显存） |

## 关键实现细节

- **TP Linear**：每个 linear 层用自定义 `autograd.Function` 封装 forward/backward。forward 保存 AllGather 后的完整输入供 backward 复用。`use_overlap` 参数控制是否启用 overlap，同一个 Module 内部根据 flag 分发到不同的 autograd Function。
- **Overlap AG+GEMM**：将 AllGather 拆成 ring P2P exchange（`dist.batch_isend_irecv`），每收到一个 chunk 就立即做 GEMM，通信和计算在两个 CUDA stream 上交替执行。
- **Overlap GEMM+RS**：将输入按 TP size 切分，每个 chunk 的 GEMM 完成后用 `dist.reduce(dst=i)` 归约到对应 rank（而非 `reduce_scatter`，后者按 chunk 独立调用时 scatter 粒度不对）。
- **RMSNorm 梯度**：RMSNorm 权重在所有 rank 上复制，但每个 rank 只看到 `s/n` 个位置的梯度，backward 后需要 AllReduce。
- **权重初始化**：全局 `torch.manual_seed` → 所有 rank 从同一 RNG 按相同顺序生成权重 → 切分后的 shard 与完整模型一致。
