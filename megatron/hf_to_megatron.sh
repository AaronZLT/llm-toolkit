python tools/checkpoint_conversion/llama_checkpoint_conversion.py \
--load_path "/workspace/checkpoints/models/llama-2-7b-chat-hf" \
--save_path "/workspace/checkpoints/megatron_models/llama-2-7b-chat-hf-tp1" \
--target_tensor_model_parallel_size 1 \
--target_pipeline_model_parallel_size 1 \
--target_data_parallel_size 8 \
--target_params_dtype "fp16" \
--make_vocab_size_divisible_by 1 \
--print-checkpoint-structure \
--megatron-path "/workspace/megatron/Megatron-LLaMA-main"