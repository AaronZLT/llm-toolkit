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

base_model_name_or_path = "/mnt/sdb/zhanglongteng/sdd/zhanglongteng/llama-3-8B-Instruct"
peft_model_name_or_path = "/mnt/sdb/zhanglongteng/workspace/llm-toolkit/examples/sparse-finetune/validate/checkpoint-100"
load_in_4bit = True
sparse_named_mask_path = "/mnt/sdb/zhanglongteng/workspace/llm-toolkit/examples/sparse-finetune/validate/checkpoint-100/named_mask.pth"

model, tokenizer = load(
    base_model_name_or_path=base_model_name_or_path,
    peft_model_name_or_path=peft_model_name_or_path,
    load_in_4bit=load_in_4bit,
    sparse_named_mask_path=sparse_named_mask_path,
)

print(model)

"""
if you want to merge the model and save the model and tokenizer to the specified path:

>>> model = model.merge_and_unload()
>>> model.save_pretrained("merged")
>>> tokenizer.save_pretrained("merged")

note that when load_in_4bit is Ture, the model will be saved in 4-bit format.
"""
