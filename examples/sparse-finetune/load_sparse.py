import os
import json
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
    load,
)

model, tokenizer = load(
    "/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Llama-2-7b-hf",
    peft_model_name_or_path="/mnt/sdb/zhanglongteng/workspace/llm-toolkit/examples/sparse-finetune/finetune.llama2_7b.metamath40k.4bit.dynamic_sparse.sparse_warmup_steps1/checkpoint-100",
    load_in_4bit=True,
    sparse_named_mask_path="/mnt/sdb/zhanglongteng/workspace/llm-toolkit/examples/sparse-finetune/finetune.llama2_7b.metamath40k.4bit.dynamic_sparse.sparse_warmup_steps1/checkpoint-100/named_mask.pth",
)

print(model)
