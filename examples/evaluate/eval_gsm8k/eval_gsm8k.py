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
]

results = {}
for ckpt in ckpts:
    results[ckpt] = infly_evaluate("gsm8k", "/hpc2hdd/home/lzhang330/ssd_workspace/models/llama-2-7b-chat-hf", ckpt)

safe_dict2file(results, "eval_result.txt")
