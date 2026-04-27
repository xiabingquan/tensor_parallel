import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from initialize import get_tp_group, get_tp_world_size, init_weight_cpu
from parallel_linear import ColumnParallelLinear, RowParallelLinear


class PlainLinear(nn.Module):
    """Linear layer (no bias) that draws weight from the global RNG.

    Unlike nn.Linear, no kaiming init -- just torch.randn via init_weight_cpu.
    This keeps global RNG consumption identical to ColumnParallelLinear / RowParallelLinear.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        weight = init_weight_cpu(
            full_shape=(out_features, in_features),
            shard_dim=0,
            rank=0,
            world_size=1,
        )
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rms_norm(x, (self.hidden_size,), self.weight, self.eps)


class Attention(nn.Module):
    """Multi-head self-attention with optional TP."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.use_tp = config.use_tp
        tp_size = get_tp_world_size() if config.use_tp else 1
        self.num_heads_per_rank = config.num_attention_heads // tp_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        h = config.hidden_size

        if config.use_tp:
            self.q_proj = ColumnParallelLinear(h, h, use_overlap=config.use_overlap)
            self.k_proj = ColumnParallelLinear(h, h, use_overlap=config.use_overlap)
            self.v_proj = ColumnParallelLinear(h, h, use_overlap=config.use_overlap)
            self.out_proj = RowParallelLinear(h, h, use_overlap=config.use_overlap)
        else:
            self.q_proj = PlainLinear(h, h)
            self.k_proj = PlainLinear(h, h)
            self.v_proj = PlainLinear(h, h)
            self.out_proj = PlainLinear(h, h)

    def core_attn(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Naive scaled dot-product attention with fp32 softmax.

        Args:
            q, k, v: [b, nh, s, d]

        Returns:
            [b, nh, s, d]
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [b, nh, s, s]
        attn_weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        return torch.matmul(attn_weights, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        s, b, _ = q.shape
        q = q.view(s, b, self.num_heads_per_rank, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(s, b, self.num_heads_per_rank, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(s, b, self.num_heads_per_rank, self.head_dim).permute(1, 2, 0, 3)

        attn_out = self.core_attn(q, k, v)
        attn_out = attn_out.permute(2, 0, 1, 3).reshape(s, b, -1)
        return self.out_proj(attn_out)


class MLP(nn.Module):
    """Two-layer MLP with optional TP."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.use_tp:
            self.fc1 = ColumnParallelLinear(
                config.hidden_size,
                config.intermediate_size,
                use_overlap=config.use_overlap,
            )
            self.fc2 = RowParallelLinear(
                config.intermediate_size,
                config.hidden_size,
                use_overlap=config.use_overlap,
            )
        else:
            self.fc1 = PlainLinear(config.hidden_size, config.intermediate_size)
            self.fc2 = PlainLinear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.attn = Attention(config, layer_idx)
        self.norm2 = RMSNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Stacked Transformer blocks with final RMSNorm.

    When use_tp=True: input/output [s/n, b, h] (SP region).
    When use_tp=False: input/output [s, b, h].
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.use_tp = config.use_tp
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

    def allreduce_replicated_grads(self) -> None:
        """AllReduce gradients of replicated (non-sharded) parameters across TP ranks.

        RMSNorm weights are replicated on every rank but each rank only sees s/n
        sequence positions during SP. Their gradients must be summed across ranks
        to match the single-GPU gradient.
        """
        if not self.use_tp:
            return
        group = get_tp_group()
        for module in self.modules():
            if isinstance(module, RMSNorm):
                if module.weight.grad is not None:
                    dist.all_reduce(module.weight.grad, group=group)
