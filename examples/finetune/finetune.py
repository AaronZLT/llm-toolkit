import os
import json
import argparse

import torch
import transformers
from transformers import PretrainedConfig

import llmtoolkit
from llmtoolkit import (
    get_args,
    get_accelerate_model,
    build_data_module,
    get_unique_key,
    train,
    TrainingArguments,
    ModelArguments,
    print_rank_0,
    QuantConfig,
    PEFTConfig,
)


model_args, data_args, training_args = get_args()

if model_args.quant:
    quant_config = QuantConfig(
        quant_method=model_args.quant,
        model_bits=model_args.bits,
        bnb_quant_type=model_args.quant_type,
    )
else:
    quant_config = None

if model_args.peft:
    peft_config = PEFTConfig(
        peft_method=model_args.peft,
        lora_modules=model_args.lora_modules,
        lora_rank=model_args.lora_rank,
        lora_scale=model_args.lora_scale,
        init_lora_weights=model_args.init_lora_weights,
    )
else:
    peft_config = None

args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))

model, tokenizer = get_accelerate_model(
    args.model_name_or_path,
    quant_config,
    peft_config,
    flash_attn=args.flash_attn,
    compute_dtype=torch.bfloat16,
    parallelism=args.parallelism,
    gradient_checkpointing=args.gradient_checkpointing,
    deepspeed=args.deepspeed,
)

print_rank_0(model)
print_rank_0(model.config)
print_rank_0(getattr(model, "peft_config", None))
print_rank_0(getattr(model, "quantization_config", None))


data_module = build_data_module(tokenizer, data_args.dataset_name_or_path, data_args)

train(
    model,
    tokenizer,
    data_module["train_dataset"],
    data_module["eval_dataset"],
    data_module["data_collator"],
    training_args,
    get_unique_key(model_args, data_args, training_args),
)
