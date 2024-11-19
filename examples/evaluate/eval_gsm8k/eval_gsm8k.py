import os

import transformers
from transformers import AutoTokenizer

import llmtoolkit
from llmtoolkit import (
    offline_evaluate,
    infly_evaluate,
    print_rank_0,
    safe_dict2file,
)

def find_adapter_model_paths(root_dir):
    matching_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        if 'adapter_model.safetensors' in filenames:
            matching_paths.append(dirpath)
    return matching_paths

ckpts = find_adapter_model_paths('/hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-power/power-pissa-olora/')
ckpts = ["/hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-power/power-pissa-olora/pissa-gsm8k-1e-5/checkpoint-14946"]

for ckpt in ckpts:
    acc = infly_evaluate("gsm8k", "/hpc2hdd/home/lzhang330/ssd_workspace/models/llama-2-7b-chat-hf", ckpt)
    results = {}
    results["model"] = "/hpc2hdd/home/lzhang330/ssd_workspace/models/llama-2-7b-chat-hf"
    results["lora"] = ckpt
    results["task"] = "gsm8k"
    results["accuracy"] = acc
    safe_dict2file(results, "eval_result.txt")
