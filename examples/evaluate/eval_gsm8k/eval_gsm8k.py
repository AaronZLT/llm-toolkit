import transformers
from transformers import AutoTokenizer

import llmtoolkit
from llmtoolkit import (
    offline_evaluate,
    infly_evaluate,
    print_rank_0,
)

print_rank_0(f'Accuracy: {infly_evaluate("gsm8k", "/hpc2hdd/home/lzhang330/ssd_workspace/models/llama-2-7b-chat-hf", "/hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-Llama-2-7b-hf-16bit-bs8-lora/checkpoint-1870")}')
