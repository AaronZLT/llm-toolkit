#!/bin/bash

python tools/preprocess_data.py \
       --input /workspace/megatron/llm-benchmark/data/data.jsonl \
       --json-keys src complexity problem from\
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path /workspace/checkpoints/megatron_models/llama-2-7b-chat-hf-tp1 \
       --output-prefix codecomplex-llama2 \
       --workers 32 \
       --chunk-size 256 \
