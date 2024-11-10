#!/bin/bash

### Global env setting
LLM_TOOLKIT_PATH=/hpc2hdd/home/lzhang330/llm-toolkit
MODELS_PATH=/hpc2hdd/home/lzhang330/ssd_workspace/models

DATASET=$LLM_TOOLKIT_PATH/datasets
LLAMA2_1_3B=$MODELS_PATH/Llama-2-1.3b-hf
LLAMA2_7B=$MODELS_PATH/Llama-2-7b-hf
LLAMA2_13B=$MODELS_PATH/Llama-2-13b-hf
LLAMA2_70B=$MODELS_PATH/Llama-2-70b-chat-hf
PHI3MINI4K=$MODELS_PATH/Phi-3-mini-4k-instruct
PHI3MINI128K=$MODELS_PATH/Phi-3-mini-128k-instruct

GPT2XL=$MODELS_PATH/gpt2-xl

### User
sequence_lengths=(
    " --source_max_len 512 --target_max_len 512"
    # " --source_max_len 2048 --target_max_len 2048"
    # " --source_max_len 2560 --target_max_len 2560"
    # " --source_max_len 3072 --target_max_len 3072"
    # " --source_max_len 3584 --target_max_len 3584"
    # " --source_max_len 4096 --target_max_len 4096"
    # " --source_max_len 4608 --target_max_len 4608"
    # " --source_max_len 5120 --target_max_len 5120"
    # " --source_max_len 5632 --target_max_len 5632"
    # " --source_max_len 6144 --target_max_len 6144"
    # " --source_max_len 6656 --target_max_len 6656"
    # " --source_max_len 7168 --target_max_len 7168"
    # " --source_max_len 7680 --target_max_len 7680"
    # " --source_max_len 8192 --target_max_len 8192"
)


# CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch load_and_infer.py --model_name_or_path $LLAMA2_7B --peft_name_or_path /hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-Llama-2-7b-hf-BnB-4bit-bs8-lora/checkpoint-1870 --dataset gsm8k --bf16 --quant True --bits 4 --unify_load False

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch load_and_infer.py --model_name_or_path $LLAMA2_7B --peft_name_or_path /hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-Llama-2-7b-hf-BnB-4bit-bs8-lora/checkpoint-1870 --dataset gsm8k --bf16 --quant True --bits 8 --unify_load False

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch load_and_infer.py --model_name_or_path $LLAMA2_7B --peft_name_or_path /hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-Llama-2-7b-hf-BnB-4bit-bs8-lora/checkpoint-1870 --dataset gsm8k --bf16 --quant True --bits 4 --unify_load True

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch load_and_infer.py --model_name_or_path $LLAMA2_7B --peft_name_or_path /hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-Llama-2-7b-hf-BnB-4bit-bs8-lora/checkpoint-1870 --dataset gsm8k --bf16 --quant True --bits 8 --unify_load True


CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch load_and_infer.py --model_name_or_path $LLAMA2_7B --peft_name_or_path /hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-Llama-2-7b-hf-16bit-bs8-lora/checkpoint-1870 --dataset gsm8k --bf16 --quant True --bits 4

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch load_and_infer.py --model_name_or_path $LLAMA2_7B --peft_name_or_path /hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-Llama-2-7b-hf-16bit-bs8-lora/checkpoint-1870 --dataset gsm8k --bf16 --quant True --bits 8

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch load_and_infer.py --model_name_or_path $LLAMA2_7B --peft_name_or_path /hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-Llama-2-7b-hf-16bit-bs8-lora/checkpoint-1870 --dataset gsm8k --bf16 --quant True --bits 4 --unify_load True

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch load_and_infer.py --model_name_or_path $LLAMA2_7B --peft_name_or_path /hpc2hdd/home/lzhang330/llm-toolkit/tmp/gsm8k-Llama-2-7b-hf-16bit-bs8-lora/checkpoint-1870 --dataset gsm8k --bf16 --quant True --bits 8 --unify_load True