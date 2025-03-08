import os
from os.path import exists, join, isdir
from typing import Dict, Union

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.pytorch_utils import Conv1D
import bitsandbytes as bnb
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    AdaLoraConfig,
    VeraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    get_peft_model,
)
from peft.tuners.lora import LoraLayer

from .arguments import (
    ModelArguments,
    TrainingArguments,
)
from .utils import (
    print_rank_0,
    is_ipex_available,
    require_lib,
    timeit,
)
from .sparse import (
    prune_magnitude,
)


def find_all_linear_names(model, bits):
    linear_cls = (
        bnb.nn.Linear4bit
        if bits == 4
        else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    )
    conv1d_cls = Conv1D
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls) or isinstance(module, conv1d_cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")

    if "gate" in lora_module_names:
        lora_module_names.remove("gate")

    return list(lora_module_names)


@timeit
def peft_model(
    model: transformers.PreTrainedModel,
    model_bits: int,
    peft_method: str,
    lora_modules: Union[str, list],
    lora_rank: int,
    lora_scale: float,
    init_lora_weights: str,
):
    if peft_method in ["lora", "lorafa", "vera", "dora"]:
        attention_modules = [
            "query",
            "q_proj",
            "value",
            "v_proj",
            "key",
            "k_proj",
            "output",
            "o_proj",
        ]

        modules = find_all_linear_names(model, model_bits)

        if lora_modules == "all":
            pass
        elif lora_modules == "attention":
            modules = [
                s for s in modules if any(module in s for module in attention_modules)
            ]
        elif lora_modules == "mlp":
            modules = [
                s
                for s in modules
                if all(module not in s for module in attention_modules)
            ]
        else:
            target_modules = lora_modules.split(",")
            for m in target_modules:
                if m not in modules:
                    raise ValueError(
                        f"You must choose your lora modules from {modules}."
                    )
            modules = target_modules
        print_rank_0(f"adding LoRA modules to {modules}")

        if peft_method == "lora":
            config = LoraConfig(
                r=lora_rank,
                lora_alpha=int(lora_scale * lora_rank),
                target_modules=modules,
                bias="none",
                task_type="CAUSAL_LM",
                init_lora_weights=init_lora_weights
                if init_lora_weights is not None
                else True,
            )
            _peft_model = get_peft_model(model, config)
        elif peft_method == "lorafa":
            config = LoraConfig(
                r=lora_rank,
                lora_alpha=int(lora_scale * lora_rank),
                target_modules=modules,
                bias="none",
                task_type="CAUSAL_LM",
                init_lora_weights=init_lora_weights
                if init_lora_weights is not None
                else True,
            )
            _peft_model = get_peft_model(model, config)
            for name, param in _peft_model.named_parameters():
                if "lora_A" in name:
                    param.requires_grad_(False)
        elif peft_method == "dora":
            config = LoraConfig(
                r=lora_rank,
                use_dora=True,
                lora_alpha=int(lora_scale * lora_rank),
                target_modules=modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            _peft_model = get_peft_model(model, config)
        elif peft_method == "adalora":
            config = AdaLoraConfig(
                r=lora_rank,
                lora_alpha=int(lora_scale * lora_rank),
                target_modules=modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            _peft_model = get_peft_model(model, config)
        elif peft_method == "loraga":
            # please preprocess the model, no operations here
            pass
        elif peft_method == "vera":
            config = VeraConfig(r=lora_rank, target_modules=modules)
            _peft_model = get_peft_model(model, config)
    elif peft_method == "prefix":
        config = PrefixTuningConfig(
            num_virtual_tokens=30,
            task_type="CAUSAL_LM",
        )
        _peft_model = get_peft_model(model, config)
    elif peft_method == "prompt":
        config = PromptTuningConfig(
            task_type="CAUSAL_LM",
            prompt_tuning_init="TEXT",
            prompt_tuning_init_text="Below is an instruction that describes a task. Write a response.",
            num_virtual_tokens=40,
            tokenizer_name_or_path=model.config._name_or_path,
        )
        _peft_model = get_peft_model(model, config)
    elif peft_method == "embedding":
        _peft_model = model
        for name, param in _peft_model.named_parameters():
            if "embed" not in name:
                param.requires_grad_(False)
    else:
        _peft_model = model

    return _peft_model


"""
refactor the function get_accelerate_model, that list all args usd in the function instead of using the ModelArguments and TrainingArguments classes
"""


def get_accelerate_model(
    model_name_or_path: str,
    model_bits: int,
    quant: str,
    quant_type: str,
    peft: str,
    lora_modules: str,
    lora_rank: int,
    lora_scale: float,
    init_lora_weights: str,
    flash_attn: bool,
    compute_dtype: torch.dtype,
    parallelism: str,
    gradient_checkpointing: bool,
    deepspeed: str,
):
    if flash_attn is True:
        require_lib("flash_attn")
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "eager"

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()

    if not quant:
        assert model_bits in [16, 32]
    else:
        assert model_bits in [8, 4, 2, 1]

    pretrained_model_kwargs = {
        "pretrained_model_name_or_path": model_name_or_path,
        "attn_implementation": attn_implementation,
        "torch_dtype": compute_dtype,
    }
    if parallelism == "pp":
        pretrained_model_kwargs.update({"device_map": "auto"})

    if quant == "bnb":
        pretrained_model_kwargs.update(
            {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=model_bits == 4,
                    load_in_8bit=model_bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type=quant_type,
                    bnb_4bit_quant_storage="uint8",
                )
            }
        )
    elif quant == "hqq":
        raise NotImplementedError("HQQ is not supported yet")
    else:
        pretrained_model_kwargs.update(
            {
                "quantization_config": None,
            }
        )

    print_rank_0(f"Loading base model from {model_name_or_path}.")
    model = AutoModelForCausalLM.from_pretrained(**pretrained_model_kwargs)

    if compute_dtype == torch.float16 and model_bits == 4:
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

    if parallelism == "pp":
        setattr(model, "model_parallel", True)
        setattr(model, "is_parallelizable", True)

    model.config.torch_dtype = compute_dtype

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"

    # add special tokens
    # 1. add pad_token if pad_token is None, as "<|reserved_special_token_100|>", not as any in-use tokens. The reason to use "<|reserved_special_token_100|>" rather than "<|pad|>" is that we do not want to introduce extra tokens when finetune LLama3 models
    # 2. add unk_token if unk_token is None (even though for most models they do have unk_token), as "<|reserved_special_token_101|>"
    # 3. final check pad_token, eos_token, bos_token, unk_token
    # 4. the best case is when the tokenizer and model support these reserved tokens, since then we do not need to resize the embedding
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "<|reserved_special_token_100|>"
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = tokenizer.convert_ids_to_tokens(
            model.config.eos_token_id
        )
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = tokenizer.convert_ids_to_tokens(
            model.config.bos_token_id
        )
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = "<|reserved_special_token_101|>"
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)
    print_rank_0(f"pad_token: {tokenizer.pad_token}")
    print_rank_0(f"eos_token: {tokenizer.eos_token}")
    print_rank_0(f"bos_token: {tokenizer.bos_token}")
    print_rank_0(f"unk_token: {tokenizer.unk_token}")

    if quant:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )

    if peft:
        model = peft_model(
            model,
            model_bits,
            peft,
            lora_modules,
            lora_rank,
            lora_scale,
            init_lora_weights,
        )

        if flash_attn or deepspeed is not None:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(
                        torch.float16
                        if compute_dtype == torch.float16
                        else torch.bfloat16
                    )
                if "norm" in name:
                    module = module.to(
                        torch.float16
                        if compute_dtype == torch.float16
                        else torch.bfloat16
                    )
                if "lm_head" in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        if module.weight.dtype == torch.float32:
                            module = module.to(
                                torch.float16
                                if compute_dtype == torch.float16
                                else torch.bfloat16
                            )

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model, tokenizer


def print_trainable_parameters(model, debug=False):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            if debug:
                print_rank_0(f"{name} requires grad")
            trainable_params += param.numel()
    print_rank_0(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    return (trainable_params, all_param, 100 * trainable_params / all_param)


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


# deprecate


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, "completed"))
        if is_completed:
            return None, True  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith(
                "checkpoint"
            ):
                max_step = max(max_step, int(filename.replace("checkpoint-", "")))
        if max_step == 0:
            return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f"checkpoint-{max_step}")
        print_rank_0(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training
