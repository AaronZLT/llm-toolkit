import os
from os.path import exists, join, isdir
from typing import Dict

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaTokenizer
)
from transformers.pytorch_utils import Conv1D
import bitsandbytes as bnb
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from peft.tuners.lora import LoraLayer

from .utils import (
    print_rank_0,
    is_ipex_available,
    )
from .dataset import (
    DEFAULT_PAD_TOKEN,
)
from .flashattn import (
    replace_llama_attn_with_flash_attn,
)

def find_all_linear_names(args, model):
    linear_cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    conv1d_cls = Conv1D
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls) or isinstance(module, conv1d_cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def get_accelerate_model(args, checkpoint_dir):
    if args.flash_attn==True:
        print_rank_0("Use FLASH ATTENTION! Replacing......")
        replace_llama_attn_with_flash_attn()
        
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
        
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    if args.deepspeed != None:
        print_rank_0("Using deepspeed, disabling device_map...")
        device_map = None

    if not args.quant: assert args.bits in [16, 32]

    print_rank_0(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    if args.quant:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            device_map=device_map,
            max_memory=max_memory,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.double_quant,
                bnb_4bit_quant_type=args.quant_type,
            ),
            torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
            trust_remote_code=args.trust_remote_code,
            use_auth_token=args.use_auth_token
        )
    else:
        print_rank_0("=======LOAD UNQUANTED MODEL=======")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            device_map=device_map,
            torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
            trust_remote_code=args.trust_remote_code,
            use_auth_token=args.use_auth_token
        )
    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print_rank_0('='*80)
            print_rank_0('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print_rank_0('='*80)
            
    if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
        compute_dtype = torch.bfloat16
        print_rank_0('Intel XPU does not support float16 yet, so switching to bfloat16')

    # setattr(model, 'model_parallel', True)
    # setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False,
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None,
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
    )
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
        
    # if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
    #     # LLaMA tokenizer may not have correct special tokens set.
    #     # Check and add them if missing to prevent them from being parsed into different tokens.
    #     # Note that these are present in the vocabulary.
    #     # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
    #     print_rank_0('Adding special tokens.')
    #     tokenizer.add_special_tokens({
    #             "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
    #             "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
    #             "unk_token": tokenizer.convert_ids_to_tokens(
    #                 model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
    #             ),
    #     })
    
    # TODO ltzhang: find a correct way for quant training.
    if args.quant:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if args.use_lora:
        print_rank_0(f'adding LoRA modules...')

        modules=[]
        if args.lora_modules == "all":
            modules = find_all_linear_names(args, model)
        elif args.lora_modules == "attention":
            modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        elif args.lora_modules == "mlp":
            modules = ['up_proj', 'down_proj', 'gate_proj']
        else:
            modules = find_all_linear_names(args, model)
            target_modules = args.lora_modules.split(",")
            for m in target_modules:
                if m not in modules:
                    raise ValueError(f"You must choose your lora modules from {modules}.")
            modules = target_modules

        print_rank_0(modules)
        if args.fa:
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=2*args.lora_r,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                fa=args.fa,
                percent=args.percent,
                init_method = args.init_method,
            )
        else:
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=2*args.lora_r,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
        model = get_peft_model(model, config)

        if args.flash_attn or args.deepspeed != None:
            for name, module in model.named_modules():
                print_rank_0(name)
                if isinstance(module, LoraLayer):
                    module = module.to(torch.float16 if args.fp16 else torch.bfloat16)
                if 'norm' in name:
                    module = module.to(torch.float16 if args.fp16 else torch.bfloat16)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        if module.weight.dtype == torch.float32:
                            module = module.to(torch.float16 if args.fp16 else torch.bfloat16)
        
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)    

    return model, tokenizer

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print_rank_0(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")
    
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print_rank_0(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training
