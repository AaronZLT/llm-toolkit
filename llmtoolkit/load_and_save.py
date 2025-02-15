import os
from typing import Dict

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    PeftModel,
)

from .sparse import (
    apply_spare,
)
from .utils import (
    print_rank_0,
    is_ipex_available,
    rank_0,
    create_timestamp,
)


def load(
    base_model_name_or_path: str,
    peft_model_name_or_path: str = None,
    load_in_4bit: bool = False,
    sparse_named_mask_path: str = None,
):
    """
    Load a language model with optional PEFT adapter and sparse mask.

    This function loads a pre-trained causal language model and its tokenizer
    from the given `base_model_name_or_path`. It also supports loading a
    Parameter-Efficient Fine-Tuning (PEFT) adapter from `peft_model_name_or_path`,
    resizing the token embeddings if necessary. Additionally, it allows applying
    a sparse mask from `sparse_named_mask_path` to the model. *Note that the 
    quantization and sparse named mask is only applied to the base model.
    
    i.e, quantization(sparse(base_model)) + lora_model

    Args:
        base_model_name_or_path (str): Path or name of the base pre-trained model.
        peft_model_name_or_path (str, optional): Path or name of the PEFT adapter model.
            If provided, the adapter is loaded and integrated with the base model.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit precision
            for reduced memory usage. Default is False.
        sparse_named_mask_path (str, optional): Path to a sparse named mask file.
            If provided, the mask is applied to the model.

    Returns:
        Tuple[torch.nn.Module, transformers.PreTrainedTokenizer]:
            - The loaded model, with optional PEFT and sparse mask applied.
            - The tokenizer corresponding to the final model configuration.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path, load_in_4bit=load_in_4bit
    ).to(device)
    target_tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

    if peft_model_name_or_path:
        peft_tokenizer = AutoTokenizer.from_pretrained(peft_model_name_or_path)
        if len(target_tokenizer) != len(peft_tokenizer):
            print_rank_0(
                f"Since the embedding of base model mismatch peft adapter ({len(target_tokenizer)} - {len(peft_tokenizer)}), resizing."
            )
            model.resize_token_embeddings(len(peft_tokenizer))
        target_tokenizer = peft_tokenizer
        model = PeftModel.from_pretrained(model, peft_model_name_or_path)

    if sparse_named_mask_path:
        named_mask = torch.load(sparse_named_mask_path)
        apply_spare(model, named_mask)

    return model, target_tokenizer


def flexible_load(args):
    if args.flash_attn:
        import importlib.util

        flashattn_spec = importlib.util.find_spec("flash-attn")
        if flashattn_spec is None:
            raise FileNotFoundError(
                "You can not use flash_attn now since flash-attn was not installed."
            )

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()

    max_memory = f"{args.max_memory_MB}MB"
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = None

    if args.device_map is not None:
        # if we are in a distributed setting, we need to set the device map and max memory per device
        if os.environ.get("LOCAL_RANK") is not None:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            device_map = {"": local_rank}
            max_memory = {"": max_memory[local_rank]}

    if args.deepspeed is not None:
        print_rank_0("Using deepspeed, disabling device_map...")
        device_map = None

    if not args.quant:
        assert args.bits in [16, 32]

    print_rank_0(f"loading base model {args.model_name_or_path}...")
    compute_dtype = (
        torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    )

    if args.quant:
        print_rank_0("LOADING QUANTIZED MODEL")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
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
            torch_dtype=(
                torch.float16
                if args.fp16
                else (torch.bfloat16 if args.bf16 else torch.float32)
            ),
            trust_remote_code=args.trust_remote_code,
            use_auth_token=args.use_auth_token,
            attn_implementation="flash_attention_2" if args.flash_attn else "eager",
        )
    else:
        print_rank_0("LOADING UNQUANTIZED MODEL")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map=device_map,
            torch_dtype=(
                torch.float16
                if args.fp16
                else (torch.bfloat16 if args.bf16 else torch.float32)
            ),
            trust_remote_code=args.trust_remote_code,
            use_auth_token=args.use_auth_token,
            attn_implementation="flash_attention_2" if args.flash_attn else "eager",
        )
    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print_rank_0("=" * 80)
            print_rank_0(
                "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
            )
            print_rank_0("=" * 80)

    if compute_dtype == torch.float16 and (
        is_ipex_available() and torch.xpu.is_available()
    ):
        compute_dtype = torch.bfloat16
        print_rank_0("Intel XPU does not support float16 yet, so switching to bfloat16")

    # setattr(model, 'model_parallel', True)
    # setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype = (
        torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "right"

    # add special tokens
    # 1. add pad_token if pad_token is None, as unk_token or eos_token if unk_token is None
    # 2. add unk_token if unk_token is None, as pad_token or eos_token if pad_token is None
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = (
            tokenizer.unk_token
            if tokenizer.unk_token is not None
            else tokenizer.convert_ids_to_tokens(model.config.eos_token_id)
        )
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = tokenizer.convert_ids_to_tokens(
            model.config.eos_token_id
        )
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = tokenizer.convert_ids_to_tokens(
            model.config.bos_token_id
        )
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = (
            tokenizer.pad_token
            if tokenizer.pad_token is not None
            else tokenizer.convert_ids_to_tokens(model.config.eos_token_id)
        )

    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)
    print_rank_0(f"pad_token: {tokenizer.pad_token}")
    print_rank_0(f"eos_token: {tokenizer.eos_token}")
    print_rank_0(f"bos_token: {tokenizer.bos_token}")
    print_rank_0(f"unk_token: {tokenizer.unk_token}")

    if args.peft_name_or_path:
        print_rank_0("Loading adapter")
        model = PeftModel.from_pretrained(model, args.peft_name_or_path)
        if args.unify_load:
            model = model.merge_and_unload()

    return model, tokenizer


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    print_rank_0(f"adding special tokens, {special_tokens_dict}")
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


@rank_0
def merge_and_save(model_name_or_path, peft_path, output_path):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_path)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    save_url = os.path.join(output_path, f"merged_model_{create_timestamp()}")
    model.save_pretrained(save_url)
    tokenizer.save_pretrained(save_url)
    print_rank_0(f"Merged model has been successfully saved at {save_url}.")
    del model, tokenizer
