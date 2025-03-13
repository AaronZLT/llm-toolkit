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
    check_sparsity,
)

base_model_name_or_path = "/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Llama-2-7b-chat-hf"
peft_model_name_or_path = "/mnt/sdb/zhanglongteng/workspace/llm-toolkit/examples/sparse-finetune/validate/checkpoint-100"
sparse_named_mask_path = "/mnt/sdb/zhanglongteng/workspace/llm-toolkit/examples/sparse-finetune/validate/checkpoint-100/named_mask.pth"


model, tokenizer = load(
    base_model_name_or_path=base_model_name_or_path,
    peft_model_name_or_path=peft_model_name_or_path,
    load_in_4bit=False,
    sparse_named_mask_path=sparse_named_mask_path,
)


print_rank_0(model)
check_sparsity(model)

print_rank_0("Saving sparsed base model and tokenizer to sparsed/")
model.base_model.save_pretrained("sparsed")
tokenizer.save_pretrained("sparsed")
