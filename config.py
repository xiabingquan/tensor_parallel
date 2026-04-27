
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the toy Transformer model.

    All fields must be explicitly provided -- no defaults.
    """

    hidden_size: int           # h
    num_attention_heads: int   # nh
    intermediate_size: int     # h' (MLP intermediate dim)
    num_layers: int            # L
    sequence_length: int       # s
    batch_size: int            # b
    tp_size: int               # n (TP degree, 1 = no TP)
    use_tp: bool               # whether to use tensor parallelism
    use_overlap: bool          # whether to use comm-compute overlap

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_attention_heads ({self.num_attention_heads})"
        )
        if self.use_tp:
            assert self.tp_size > 1, "use_tp=True requires tp_size > 1"
            assert self.hidden_size % self.tp_size == 0
            assert self.num_attention_heads % self.tp_size == 0
            assert self.intermediate_size % self.tp_size == 0
            assert self.sequence_length % self.tp_size == 0
            if self.use_overlap:
                assert self.sequence_length % (self.tp_size * self.tp_size) == 0
        if not self.use_tp:
            assert not self.use_overlap, "use_overlap requires use_tp=True"
