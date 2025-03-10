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
    bnb_quant_type: float
