from dataclasses import dataclass


@dataclass
class PEFTConfig:
    peft_method: str
    lora_modules: str
    lora_rank: int
    lora_scale: float
    init_lora_weights: str


@dataclass
class QuantConfig:
    quant_method: str
    model_bits: int
    bnb_quant_type: float


@dataclass
class SparseConfig:
    sparsity_ratio: float
    sparse_warmup_ratio: float
    sparse_warmup_steps: int
    sparse_preserve_accuracy: bool
    sparse_prune_largest: bool
    sparse_n: int
    sparse_m: int
