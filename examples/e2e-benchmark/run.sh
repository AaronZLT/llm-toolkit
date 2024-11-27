#!/bin/bash

### Global env setting
MODELS_PATH=llm-toolkit/models

LLAMA2_1_3B=$MODELS_PATH/Llama-2-1.3b-hf
LLAMA2_7B=$MODELS_PATH/Llama-2-7b-hf
LLAMA2_13B=$MODELS_PATH/Llama-2-13b-hf
LLAMA2_70B=$MODELS_PATH/Llama-2-70b-chat-hf

F=" --flash_attn True"
R=" --gradient_checkpointing True"
L=" --peft lora --lora_rank 64"
QL=" --quant True --bits 4 --peft lora --lora_rank 64"
Z2=" --deepspeed zero2.json"
Z2O=" --deepspeed zero2off.json"
Z3=" --deepspeed zero3.json"
Z3O=" --deepspeed zero3off.json"


### local env setting
gpus=(
    "CUDA_VISIBLE_DEVICES=0 "
    "CUDA_VISIBLE_DEVICES=0,1 "
    "CUDA_VISIBLE_DEVICES=0,1,2,3 "
    "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
)

techniques=(
    # 1 mix
    ""
    "${F}"
    "${R}"
    "${L}"
    "${QL}"
    "${Z2}"
    "${Z2O}"
    "${Z3}"
    "${Z3O}"
    # 2 mix
    "${F}${R}"
    "${F}${L}"
    "${F}${QL}"
    "${F}${Z2}"
    "${F}${Z2O}"
    "${F}${Z3}"
    "${F}${Z3O}"
    "${R}${L}"
    "${R}${QL}"
    "${R}${Z2}"
    "${R}${Z2O}"
    "${R}${Z3}"
    "${R}${Z3O}"
    "${L}${Z2}"
    "${L}${Z2O}"
    "${L}${Z3}"
    "${L}${Z3O}"
    # 3 mix
    "${F}${R}${L}"
    "${F}${R}${QL}"
    "${F}${R}${Z2}"
    "${F}${R}${Z2O}"
    "${F}${R}${Z3}"
    "${F}${R}${Z3O}"
    "${R}${L}${Z2}"
    "${R}${L}${Z2O}"
    "${R}${L}${Z3}"
    "${R}${L}${Z3O}"
    # 4 mix
    "${F}${R}${L}${Z2}"
    "${F}${R}${L}${Z2O}"
    "${F}${R}${L}${Z3}"
    "${F}${R}${L}${Z3O}"
)

model_sizes=(
    " --model_name_or_path $LLAMA2_1_3B"
    " --model_name_or_path $LLAMA2_7B"
    " --model_name_or_path $LLAMA2_13B"
    " --model_name_or_path $LLAMA2_70B"
)

sequence_lengths=(
    " --source_max_len 128 --target_max_len 128"
    " --source_max_len 256 --target_max_len 256"
    " --source_max_len 512 --target_max_len 512"
    " --source_max_len 1024 --target_max_len 1024"
    " --source_max_len 2048 --target_max_len 2048"
)

batch_sizes=(
    " --per_device_train_batch_size 1"
    " --per_device_train_batch_size 2"
    " --per_device_train_batch_size 4"
    " --per_device_train_batch_size 8"
    " --per_device_train_batch_size 16"
    " --per_device_train_batch_size 32"
    " --per_device_train_batch_size 64"
    " --per_device_train_batch_size 96"
    " --per_device_train_batch_size 128"
    " --per_device_train_batch_size 160"
    " --per_device_train_batch_size 192"
    " --per_device_train_batch_size 224"
    " --per_device_train_batch_size 256"
)

base="DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch benchmark.py --dataset alpaca --output_dir benchmark --logging_strategy steps --logging_steps 1 --save_strategy no --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps 5 --hard_padding True"

for gpu in "${gpus[@]}"; do
    for model_size in "${model_sizes[@]}"; do
        for technique in "${techniques[@]}"; do
            for sequence_length in "${sequence_lengths[@]}"; do
                for batch_size in "${batch_sizes[@]}"; do
                    command="${gpu}${base}${model_size}${sequence_length}${batch_size}${technique}"
                    echo "Executing: $command"
                    eval $command
                    if [ $? -ne 0 ]; then
                        echo "Command failed with batch size: ${batch_size}. Stopping further execution."
                        break
                    fi
                done
            done
        done
    done
done

