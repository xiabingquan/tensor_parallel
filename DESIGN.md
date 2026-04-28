# Tensor Parallel 设计文档

## 符号约定

| 符号 | 含义 |
|---|---|
| `p` | TP 并行度（TP size），即参与 Tensor Parallel 的 GPU 数量 |
| `s` | 序列长度（sequence length） |
| `b` | 批大小（batch size） |
| `h` | 隐藏维度（hidden size） |
| `h'` | 线性层输出维度（可以与 `h` 不同，如 MLP 的中间维度） |
| `V` | 词表大小（vocabulary size） |
| `L` | Transformer 层数 |
| `n` | 注意力头数（number of attention heads） |

## 1. 基本信息

### 概述

Tensor Parallel（TP）是一种模型并行策略，将单个模块的权重切分到多个 GPU 上，每个 GPU 只持有权重的一部分，独立完成局部计算后通过集合通信合并结果。合并后的结果与在单张 GPU 上用完整权重计算在数学上完全等价。TP 的切分粒度是模块级别的，目标是在单层参数量过大时，将计算和存储分散到多个设备上。

TP 主要作用于 Linear 层，这是 Transformer 中参数量最大的模块。此外，Vocab Embedding 也可以沿词表维度做 TP 切分。本文聚焦于 Linear 层的 TP 实现。

### TP Linear 的通信模式

TP 对 Linear 层有两种切分方式：Column Parallel（按输出维度切分）和 Row Parallel（按输入维度切分）。线性层的计算公式为 `Y = X @ A^T`，其中权重 `A` 的 shape 为 `[h', h]`（PyTorch 约定，即 `[out_features, in_features]`）。

**Column Parallel Linear**

将权重 `A` 按第 0 维（输出维度）切分，每个 GPU 持有 `A_i`（shape `[h'/p, h]`），独立计算局部输出 `Y_i = X @ A_i^T`。各 GPU 的输出是完整结果的不同列，无需通信即可得到正确的局部结果。但在 backward 中，由于各 GPU 需要完整的 dX 来更新上游参数，需要一次 AllReduce 对 dX 求和。

| 阶段 | 权重 shape | 输入 shape | 输出 shape | 通信 |
|---|---|---|---|---|
| 初始化 | `[h'/p, h]` | — | — | 无 |
| Forward | `[h'/p, h]` | `[s, b, h]` | `[s, b, h'/p]` | 无 |
| Backward (dX) | `[h'/p, h]` | grad: `[s, b, h'/p]` | `[s, b, h]` | AllReduce |
| Backward (dW) | — | grad: `[s, b, h'/p]`, input: `[s, b, h]` | `[h'/p, h]` | 无 |

**Row Parallel Linear**

将权重 `A` 按第 1 维（输入维度）切分，每个 GPU 持有 `A_i`（shape `[h', h/p]`）。输入也需按最后一维切分为 `X_i`（shape `[s, b, h/p]`）。每个 GPU 计算 `Y_i = X_i @ A_i^T`，得到的是最终结果的一个部分和，需要通过 AllReduce 求和才能得到正确输出。Backward 中 dX 天然是按最后一维切分的局部结果，无需通信。

| 阶段 | 权重 shape | 输入 shape | 输出 shape | 通信 |
|---|---|---|---|---|
| 初始化 | `[h', h/p]` | — | — | 无 |
| Forward | `[h', h/p]` | `[s, b, h/p]` | `[s, b, h']` | AllReduce |
| Backward (dX) | `[h', h/p]` | grad: `[s, b, h']` | `[s, b, h/p]` | 无 |
| Backward (dW) | — | grad: `[s, b, h']`, input: `[s, b, h/p]` | `[h', h/p]` | 无 |

**成对使用**

在 Transformer 中，Column Parallel 和 Row Parallel 成对使用（Attention 中 QKV 投影 + 输出投影，MLP 中 FC1 + FC2）。Column Parallel 的输出 `[s, b, h'/p]` 可以直接作为 Row Parallel 的输入 `[s, b, h/p]`，中间无需通信。同时，Column Parallel backward 的 AllReduce 也可以省去，因为 Row Parallel 的 backward 会自然地产生正确的局部梯度。这样，每对 Column + Row 在 forward 和 backward 中各只需要一次 AllReduce。

### 从 TP 到 SP（Sequence Parallel）

虽然 TP Linear 本身的切分是无冗余的，但 Transformer 中有些模块无法做 TP 切分，典型的如 RMSNorm 和 Dropout。这些模块在每个 TP rank 上持有完整的 `[s, b, h]` 张量并执行相同的计算，造成了激活内存和计算的冗余。

SP 解决这一问题：将非 TP 区域的张量沿 sequence 维度切分为 `[s/p, b, h]`，每个 rank 只处理一部分 sequence。相应地，TP Linear 的通信模式从 AllReduce 变为 AllGather + ReduceScatter：进入 TP 区域前通过 AllGather 恢复完整 sequence，离开 TP 区域时通过 ReduceScatter 将结果分散回各 rank。

| 对比项 | 仅 TP | TP + SP |
|---|---|---|
| 非 TP 区域的张量 shape | `[s, b, h]`（各 rank 完全相同） | `[s/p, b, h]`（各 rank 持有不同 sequence 片段） |
| 非 TP 区域的计算 | 各 rank 重复计算 | 各 rank 只计算自己的 `1/p` |
| 非 TP 区域的激活内存 | 每个 rank 持有完整激活 | 每个 rank 只持有 `1/p` 激活 |
| 每对 TP Linear 的 forward 通信 | 1 次 AllReduce | 1 次 AllGather + 1 次 ReduceScatter |
| 每对 TP Linear 的通信总量 | `2N(p-1)/p` | `2N(p-1)/p`（相同） |

通信总量不变（AllReduce = ReduceScatter + AllGather），但非 TP 区域的激活内存和计算开销降低到 `1/p`。

本文后续讨论 TP 时，默认开启 SP。

### 优势

- **无冗余并行**：TP Linear 的权重按 `1/p` 切分，计算量按 `1/p` 分担，没有任何重复的显存占用或计算浪费。配合 SP，非 TP 区域的计算和激活同样无冗余。
- **数学等价**：切分后各 GPU 的局部计算结果经通信合并后，与单卡上的完整计算结果完全一致。
- **通信高效**：TP 通常部署在节点内（通过 NVLink 连接），通信带宽充足。

### 适用范围

- Transformer 中的所有 Linear 层：Attention 的 QKV 投影与输出投影、MLP 的两个全连接层。
- 通常与 Data Parallel、Pipeline Parallel 配合使用，TP 负责节点内并行，其他并行负责节点间扩展。

### 局限性

- TP degree 受限于 hidden size 的可整除性，一般为 2、4、8。
- 每对 TP Linear 的 forward 和 backward 各需要一次集合通信，跨节点部署时通信开销显著。
- TP degree 越大，每个 GPU 上的 GEMM 越小，计算效率越低。

## 2. 核心实现

### 通信原语

TP 的实现依赖以下通信原语，均沿 sequence 维度（第 0 维）操作：

| 原语 | 行为 | 输入 shape | 输出 shape | 用途 |
|---|---|---|---|---|
| AllGather | 收集各 rank 的片段并拼接 | `[s/p, b, h]` | `[s, b, h]` | ColumnParallel forward / RowParallel backward |
| ReduceScatter | 对各 rank 的完整张量求和后分散 | `[s, b, h]` | `[s/p, b, h]` | RowParallel forward / ColumnParallel backward |

这两个原语在 autograd 中以共轭形式出现：forward 中的 AllGather 对应 backward 中的 ReduceScatter，反之亦然。

### ColumnParallelLinear

每个 TP rank 持有权重的 `[h'/p, h]` 部分（即 `[out_features/p, in_features]`）。

| 阶段 | 操作 | 输入 shape | 输出 shape | 通信 |
|---|---|---|---|---|
| Forward | AllGather 输入 → GEMM | `[s/p, b, h]` | `[s, b, h'/p]` | AllGather |
| Backward (dX) | GEMM → ReduceScatter | grad: `[s, b, h'/p]` | `[s/p, b, h]` | ReduceScatter |
| Backward (dW) | GEMM（用保存的完整输入） | grad: `[s, b, h'/p]` × input: `[s, b, h]` | `[h'/p, h]` | 无 |

Forward 时先 AllGather 将输入从 `[s/p, b, h]` 恢复为 `[s, b, h]`，再做局部 GEMM 得到 `[s, b, h'/p]`。Backward 时先计算完整的 dX（shape `[s, b, h]`），再通过 ReduceScatter 分散到各 rank 得到 `[s/p, b, h]`。dW 直接用 forward 时保存的完整输入与 grad_output 计算，无需通信。

### RowParallelLinear

每个 TP rank 持有权重的 `[h, h/p]` 部分（即 `[out_features, in_features/p]`）。

| 阶段 | 操作 | 输入 shape | 输出 shape | 通信 |
|---|---|---|---|---|
| Forward | GEMM → ReduceScatter | `[s, b, h/p]` | `[s/p, b, h]` | ReduceScatter |
| Backward (dY) | AllGather → GEMM | grad: `[s/p, b, h]` | `[s, b, h/p]` | AllGather |
| Backward (dW) | GEMM（用 AllGather 后的 grad） | grad: `[s, b, h]` × input: `[s, b, h/p]` | `[h, h/p]` | 无 |

Forward 时先做局部 GEMM 得到 `[s, b, h]`（部分和），再通过 ReduceScatter 求和并分散到各 rank 得到 `[s/p, b, h]`。Backward 时先 AllGather 将 grad_output 从 `[s/p, b, h]` 恢复为 `[s, b, h]`，再计算 dX 和 dW。

### 权重初始化

TP 的权重初始化需要保证：各 rank 切分后的权重拼接起来，等价于对完整权重做同一种初始化。有两种实现方式：

- **CPU 初始化**：在 CPU 上用相同的随机种子初始化完整权重矩阵，然后按 rank 切出对应的 shard 拷贝到 GPU。简单可靠，但完整权重可能超出内存。
- **GPU 初始化**：每个 rank 用不同的随机种子直接在 GPU 上初始化自己的 shard。需要为 TP 参数维护独立的 RNG 状态，确保各 rank 的种子确定且与非 TP 参数隔离。Megatron 通过 `CudaRNGStatesTracker` 实现这一点。

### 在 Transformer 中的使用

一个 Transformer 层由 Attention + MLP 组成，每个子模块内部用一对 Column + Row Parallel Linear。TP 切分方式：

- **Attention**：QKV 投影用 ColumnParallel（输出按 head 切分，每个 rank 处理 `n/p` 个 head），输出投影用 RowParallel。
- **MLP**：FC1 用 ColumnParallel（输出维度 `h'/p`），FC2 用 RowParallel。
- **RMSNorm、Dropout 等**：处于 SP 区域，操作 `[s/p, b, h]` 的张量，各 rank 独立处理不同的 sequence 片段。

每个 Transformer 层 forward 共 4 次通信（2 次 AllGather + 2 次 ReduceScatter），backward 对称也是 4 次。

### 数据流

以下流程展示了 token 从输入到 loss 的完整数据流（TP + SP 视角下，每个 rank 的数据 shape 变化）：

```
输入 token IDs: [b, s]
        │
        ▼
   ┌─────────────┐
   │ Scatter 到各 │  按 sequence 维度切分
   │ TP rank     │
   └─────────────┘
        │
        ▼
每个 rank 持有: [b, s/p]
        │
        ▼
   ┌─────────────┐
   │   Vocab      │  权重 [V/p, h]，每个 rank 处理 V/p 个词
   │  Embedding   │  查表后 AllReduce（或 ReduceScatter）
   └─────────────┘
        │
        ▼
  Embedding 输出: [s/p, b, h]     ← SP 区域
        │
        ▼
  ╔═══════════════════════════════════════════╗
  ║          Transformer Layer × L            ║
  ║                                           ║
  ║  [s/p, b, h]                              ║
  ║       │                                   ║
  ║       ▼                                   ║
  ║   RMSNorm        [s/p, b, h]  ← SP 区域  ║
  ║       │                                   ║
  ║       ▼                                   ║
  ║   AllGather       [s/p, b, h] → [s, b, h] ║
  ║       │                                   ║
  ║       ▼                                   ║
  ║   QKV 投影        [s, b, h] → [s, b, 3h/p]║
  ║   (ColumnParallel)                        ║
  ║       │                                   ║
  ║       ▼                                   ║
  ║   Attention       [s, b, 3h/p]→[s, b, h/p]║
  ║   (局部计算)                               ║
  ║       │                                   ║
  ║       ▼                                   ║
  ║   输出投影        [s, b, h/p] → [s, b, h]  ║
  ║   (RowParallel)                           ║
  ║       │                                   ║
  ║       ▼                                   ║
  ║   ReduceScatter   [s, b, h] → [s/p, b, h] ║
  ║       │                                   ║
  ║       ▼                                   ║
  ║   Residual + RMSNorm  [s/p, b, h] ← SP   ║
  ║       │                                   ║
  ║       ▼                                   ║
  ║   AllGather       [s/p, b, h] → [s, b, h] ║
  ║       │                                   ║
  ║       ▼                                   ║
  ║   FC1             [s, b, h] → [s, b, h'/p]║
  ║   (ColumnParallel)                        ║
  ║       │                                   ║
  ║       ▼                                   ║
  ║   GeLU            [s, b, h'/p]  ← TP 区域 ║
  ║       │                                   ║
  ║       ▼                                   ║
  ║   FC2             [s, b, h'/p]→ [s, b, h]  ║
  ║   (RowParallel)                           ║
  ║       │                                   ║
  ║       ▼                                   ║
  ║   ReduceScatter   [s, b, h] → [s/p, b, h] ║
  ║       │                                   ║
  ║       ▼                                   ║
  ║   Residual        [s/p, b, h]  ← SP 区域  ║
  ║                                           ║
  ╚═══════════════════════════════════════════╝
        │
        ▼
  最终隐藏状态: [s/p, b, h]
        │
        ▼
   ┌─────────────┐
   │  RMSNorm     │  [s/p, b, h]  ← SP 区域
   └─────────────┘
        │
        ▼
   ┌─────────────┐
   │ Output Layer │  AllGather → GEMM，权重 [V/p, h]
   │ (Column     │  得到 logits [s, b, V/p]
   │  Parallel)  │
   └─────────────┘
        │
        ▼
  Logits: [s, b, V/p]    每个 rank 持有词表的一部分
        │
        ▼
   ┌─────────────┐
   │  Vocab       │  各 rank 在局部 logits 上计算
   │  Parallel    │  局部 softmax 和 cross entropy，
   │  CrossEntropy│  通过 AllReduce 同步全局统计量
   └─────────────┘
        │
        ▼
     Loss: 标量
```

## 3. 拓展实现：通信与计算 Overlap

### 问题与动机

在第 2 节的实现中，通信和计算是严格串行的。以 ColumnParallelLinear 的 forward 为例：先做一次完整的 AllGather 将输入恢复为 `[s, b, h]`，等 AllGather 完成后再做一次完整的 GEMM。在 AllGather 期间 GPU 的计算单元完全闲置，在 GEMM 期间通信链路完全闲置。

核心优化思路：将集合通信拆分成多个 chunk，让通信和计算以流水线方式交替执行，使两者的耗时相互重叠。

实现 overlap 需要使用独立的 CUDA stream 分离通信和计算，并通过 event 机制保证依赖关系。需要注意的是，异步通信的 kernel 必须先于计算 kernel 被调度到 GPU 上，否则计算会阻塞通信链路，导致 overlap 失效。

### Forward Overlap：Chunked Gather + Pipelined GEMM

ColumnParallel 的 forward 需要 AllGather + GEMM。Overlap 的做法是将 AllGather 拆成 `p` 轮 P2P ring exchange，每轮收到一个 chunk 后立即对该 chunk 做局部 GEMM：

```
不开 Overlap:

  AllGather(全部)            GEMM(全部)
  |=========================|=========================|
  ↑ 通信阶段，计算闲置        ↑ 计算阶段，通信闲置

开 Overlap (p=4):

  Gather chunk0   Gather chunk1   Gather chunk2   Gather chunk3
  |==============|==============|==============|
                 |==============|==============|==============|
                  GEMM chunk0    GEMM chunk1    GEMM chunk2    GEMM chunk3
```

Ring exchange 的具体流程：

1. 每个 rank 将自己的本地 shard 作为第一个 chunk，直接开始 GEMM。
2. 同时，将本地 chunk 通过 P2P 发送给下一个 rank，并从上一个 rank 接收新 chunk。
3. 收到新 chunk 后，立即对其做 GEMM，同时将该 chunk 继续转发给下一个 rank。
4. 经过 `p-1` 轮后，所有 chunk 处理完毕，将各 chunk 的 GEMM 结果按顺序拼接。

由于 AllGather 被拆成了逐个 chunk 的 P2P 收发，每个 chunk 收到后就可以立即参与计算，通信延迟被隐藏在 GEMM 执行中。

### Backward Overlap：Pipelined GEMM + ReduceScatter

RowParallel 的 forward 以及 ColumnParallel 的 backward 存在 GEMM + ReduceScatter 的串行。Overlap 方式是将 GEMM 输出按 chunk 切分，每完成一个 chunk 的 GEMM 就立即对该 chunk 发起 ReduceScatter：

```
不开 Overlap:

  GEMM(全部)                 ReduceScatter(全部)
  |=========================|=========================|

开 Overlap (4 chunks):

  GEMM chunk0    GEMM chunk1    GEMM chunk2    GEMM chunk3
  |============|============|============|
               |============|============|============|
                RS chunk0    RS chunk1    RS chunk2    RS chunk3
```

具体流程：

1. 将输入沿 sequence 维度切成 `num_chunks` 份。
2. 对第 `i` 个 chunk 做 GEMM，得到局部输出。
3. GEMM 完成后，在通信 stream 上对该 chunk 发起 ReduceScatter，同时计算 stream 开始处理第 `i+1` 个 chunk。
4. 所有 chunk 处理完毕后，将各 chunk 的 ReduceScatter 结果拼接。

与 AllGather overlap 不同，ReduceScatter overlap 的 chunk 数量不受 TP degree 限制，可以自由选择以平衡流水线的粒度和开销。

### 各层的 Overlap 策略

将上述两种 overlap 模式应用到 Transformer 层的各个 TP Linear 上：

| 层 | Forward | Backward (dX) | Backward (dW) |
|---|---|---|---|
| QKV 投影 (ColumnParallel) | AG + GEMM overlap | GEMM + RS overlap | 无 overlap |
| 输出投影 (RowParallel) | GEMM + RS overlap | AG + GEMM overlap | 无 overlap |
| FC1 (ColumnParallel) | AG + GEMM overlap | GEMM + RS overlap | 无 overlap |
| FC2 (RowParallel) | GEMM + RS overlap | AG + GEMM overlap | 无 overlap |

## 4. 拓展思考

### AllReduce 与 AllGather + ReduceScatter 的通信量对比

对于一个大小为 `N` 的张量在 `p` 个 rank 之间通信：

- **AllReduce**：通信量约为 `2N(p-1)/p`（ring AllReduce 本身由一次 ReduceScatter + 一次 AllGather 组成）。
- **AllGather + ReduceScatter（SP 模式）**：通信量分别为 `N(p-1)/p`，总计也是 `2N(p-1)/p`。

通信总量相同。SP 的优势不在于减少通信，而在于将激活内存从 `N` 降低到 `N/p`。

### Overlap 的收益上限

通信-计算 overlap 的理论收益取决于两者的耗时比：

- 若 GEMM 耗时远大于通信，overlap 后总耗时约等于 GEMM 耗时，通信被完全隐藏。
- 若通信耗时远大于 GEMM，overlap 后总耗时约等于通信耗时，收益有限。
- 实际中，节点内 NVLink 带宽（如 900 GB/s）通常足以使通信快于 GEMM，因此 overlap 在节点内 TP 场景下效果显著。
