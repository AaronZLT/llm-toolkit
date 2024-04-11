#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

WORK_PATH=/mnt/sdb/zhanglongteng/llm-toolkit
MODEL_PATH=/mnt/sdb/zhanglongteng/data2/share/zhanglongteng_A6000
DATASET=alpaca

# 7b
MODEL_NAME=Llama-2-7b-hf

batchsize=(8)

ZERO2=" --deepspeed zero2.json"
ZERO2OFF=" --deepspeed zero2off.json"
ZERO3=" --deepspeed zero3.json"
ZERO3OFF=" --deepspeed zero3off.json"
FLASH=" --flash_attn True"
QUANT=" --quant True --double_quant True --bits 4"
RECOMP=" --gradient_checkpointing True"
LORA=" --use_lora True --fa False --lora_r 64 --lora_alpha 128 --lora_dropout 0.1"
LORAFA=" --use_lora True --fa True --lora_r 64 --lora_alpha 128 --lora_dropout 0.1"
QLORA=" --quant True --double_quant True --bits 4 --use_lora True --fa False --lora_r 64 --lora_alpha 128 --lora_dropout 0.1 --lora_modules all --percent 1.0 --init_method kaiming_uniform_"

# base
base="CUDA_VISIBLE_DEVICES=0 accelerate launch $WORK_PATH/test/agile/run_llama.py --model_name_or_path $MODEL_PATH/$MODEL_NAME --data_path $WORK_PATH/datasets --dataset $DATASET --metrics_path $WORK_PATH/metrics --output_dir output --num_train_epochs 15 --learning_rate 0.0002 --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --evaluation_strategy no --eval_steps 1 --eval_dataset_size 10 --max_eval_samples 10 --per_device_eval_batch_size 1 --max_new_tokens 32 --dataloader_num_workers 1 --remove_unused_columns False --do_train --do_eval --source_max_len 256 --target_max_len 256 --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --max_steps 5 --per_device_train_batch_size "

# for bs in "${batchsize[@]}"; do
#     eval ${base}$bs$LORA
# done

for bs in "${batchsize[@]}"; do
    eval ${base}$bs$LORAFA
done