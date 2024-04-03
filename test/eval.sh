CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DS_SKIP_CUDA_CHECK=1 accelerate launch -m lm_eval --model hf \
    --model_args="pretrained=/mnt/sdb/zhanglongteng/data2/share/zhanglongteng_A6000/Llama-2-7b-hf" \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size 1 \
    --output_path output