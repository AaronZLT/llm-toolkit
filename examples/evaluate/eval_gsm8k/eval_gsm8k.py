import transformers
from transformers import AutoTokenizer

import llmtoolkit
from llmtoolkit import (
    offline_evaluate,
    infly_evaluate,
    print_rank_0,
    safe_dict2file,
)

ckpts = [
    "/hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-power/gsm8k-1e-5/checkpoint-7473",
    "/hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-power/gsm8k-1e-5/checkpoint-14946",
    "/hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-power/gsm8k-3e-5/checkpoint-7473",
    "/hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-power/gsm8k-3e-5/checkpoint-14946",
    "/hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-power/gsm8k-5e-5/checkpoint-7473",
    "/hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-power/gsm8k-5e-5/checkpoint-14946",
    "/hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-power/gsm8k-7e-5/checkpoint-7473",
    "/hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-power/gsm8k-7e-5/checkpoint-14946",
    "/hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-power/gsm8k-9e-5/checkpoint-7473",
    "/hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-power/gsm8k-9e-5/checkpoint-14946",
    "/hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-power/gsm8k-9e-6/checkpoint-7473",
    "/hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-power/gsm8k-9e-6/checkpoint-14946",
]

for ckpt in ckpts:
    results = {}
    acc = infly_evaluate("gsm8k", "/hpc2hdd/home/lzhang330/ssd_workspace/models/llama-2-7b-chat-hf", ckpt)
    result["model"] = "/hpc2hdd/home/lzhang330/ssd_workspace/models/llama-2-7b-chat-hf"
    result["lora"] = ckpt
    result["task"] = "gsm8k"
    result["accuracy"] = acc

safe_dict2file(results, "eval_result.txt")
