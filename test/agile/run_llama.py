# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ltzhang
# ltzhang@comp.hkbu.edu.hk

from collections import defaultdict
import copy
import json
import os
import gc
import warnings
import threading
import time
import datetime
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence, Tuple, Union
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
import importlib
from packaging import version
from packaging.version import parse

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from datasets import load_dataset, load_from_disk, Dataset
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

import deepspeed

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)
import threading

def safe_write2file(dictionary, filename):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            lock = threading.Lock()
            with open(filename, 'a') as file:
                lock.acquire()
                try:
                    file.write('{')
                    for key, value in dictionary.items():
                        file.write(f'{key}: {value}, ')
                    file.write('}\n')
                finally:
                    lock.release()
    else:
        lock = threading.Lock()
        with open(filename, 'a') as file:
            lock.acquire()
            try:
                file.write('{')
                for key, value in dictionary.items():
                    file.write(f'{key}: {value}, ')
                file.write('}\n')
            finally:
                lock.release()

def get_unique_key(args):
    model = args.model_name_or_path.split('/')[-1]
    model = "7b" if "7b" in model else "13b" if "13b" in model else "70b" if "70b" in model else "unknown"
    
    bs = args.per_device_train_batch_size
    seq = args.source_max_len + args.target_max_len
    # lora
    lora = args.use_lora
    lora_config = f"r{args.lora_r}-a{int(args.lora_alpha)}-dropout{args.lora_dropout}-percent{args.percent}-module{args.lora_modules}"
    lora = "-" if not args.use_lora else f"lora-fa-{lora_config}" if args.fa else f"lora-{lora_config}"
    # flash attention
    flash = "flash" if args.flash_attn else "-"
    # recomputation
    recomputation = "recompute" if args.gradient_checkpointing else "-"
    # quant
    quant = "quant" if args.quant else "-"
    # datatype
    datatype = "fp16" if args.fp16 else "bf16" if args.bf16 else "-"
    # zero
    zero = "-" if not args.deepspeed else "zero3" if '3' in args.deepspeed else "zero2" if '2' in args.deepspeed else "-"
    # offload
    offload = "-" if not args.deepspeed else "off" if 'off' in args.deepspeed else "-"

    key = f"{model}-bs{bs}-seq{seq}-{lora}-{flash}-{recomputation}-{quant}-{datatype}-{zero}-{offload}"
    return key

# flash attention
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_kvpacked_func,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaModel,
    rotate_half,
)

def apply_rotary_pos_emb(q, k, cos_sin, position_ids):
    gather_indices = position_ids[:, :, None, None]  # [bsz, seq_len, 1, 1]
    gather_indices = gather_indices.repeat(
        1, 1, cos_sin[0].shape[1], cos_sin[0].shape[3]
    )
    bsz = gather_indices.shape[0]
    cos, sin = (
        torch.gather(x.transpose(1, 2).repeat(bsz, 1, 1, 1), 1, gather_indices)
        for x in cos_sin
    )
    q, k = ((x * cos) + (rotate_half(x) * sin) for x in (q, k))
    return q, k

def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()
    kv_heads = getattr(self, "num_key_value_heads", self.num_heads)

    q, k, v = (
        op(hidden_states).view(bsz, q_len, nh, self.head_dim)
        for op, nh in (
            (self.q_proj, self.num_heads),
            (self.k_proj, kv_heads),
            (self.v_proj, kv_heads),
        )
    )
    # shape: (b, s, num_heads, head_dim)

    kv_seq_len = k.shape[1]
    past_kv_len = 0
    if past_key_value is not None:
        past_kv_len = past_key_value[0].shape[1]
        kv_seq_len += past_kv_len

    cos_sin = self.rotary_emb(v, seq_len=kv_seq_len)
    q, k = apply_rotary_pos_emb(q, k, cos_sin, position_ids)

    if past_key_value is not None:
        # reuse k, v
        k = torch.cat([past_key_value[0], k], dim=1)
        v = torch.cat([past_key_value[1], v], dim=1)

    past_key_value = (k, v) if use_cache else None

    key_padding_mask = attention_mask
    # Ideally we could just do this:
    #  q, indices, cu_q_lens, max_s = unpad_input(q, key_padding_mask[:, -q_len:])
    # but this does not work as Flash attention treats the q seq and kv seq as starting at index 0
    # which then breaks the causality logic. Probably if q_len >> past_kv_len we should
    # just skip flash attention. Leaving this in for now to demonstrate correctness of
    # flash attention information even when q needs padding.
    # TODO(siddartha): delegate back to original implementation on this condition.
    if past_kv_len > 0:
        q = torch.cat(
            (
                torch.full(
                    (bsz, past_kv_len, self.num_heads, self.head_dim),
                    0.0,
                    dtype=q.dtype,
                    device=q.device,
                ),
                q,
            ),
            dim=1,
        )

    if key_padding_mask is None:
        output = flash_attn_func(q, k, v, 0.0, softmax_scale=None, causal=True).view(
            bsz, q_len + past_kv_len, -1
        )
    else:
        q, indices, cu_q_lens, max_s = unpad_input(q, key_padding_mask)
        # We can skip concat and call unpad twice but seems better to call unpad only once.
        kv, _, cu_k_lens, max_k = unpad_input(
            torch.stack((k, v), dim=2), key_padding_mask
        )
        output_unpad = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_q_lens,
            cu_k_lens,
            max_s,
            max_k,
            0.0,
            softmax_scale=None,
            causal=True,
        )
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        output = pad_input(output_unpad, indices, bsz, q_len + past_kv_len)

    # Need to strip off the zero query outputs.
    if past_kv_len > 0:
        output = output[:, past_kv_len:, ...]

    return self.o_proj(output), None, past_key_value


# Disable the transformation of the attention mask in LlamaModel as flash attention
# takes a boolean key_padding_mask. Fills in the past kv length for use in forward.
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat(
            (
                torch.full(
                    (input_shape[0], past_key_values_length),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ),
            dim=-1,
        )

    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples

    return attention_mask


def replace_llama_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )

    LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    LlamaAttention.forward = forward

def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True
    

if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True

# logging.basicConfig(level=logging.NOTSET)
logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "use lora? default = false"},
    )
    fa: bool = field(
        default=False,
        metadata={"help": "Use LoRA-FA? default = false"},
    )
    r: int = field(
        default=1,
        metadata={"help": "lora rank, default = 1"},
    )
    percent: Optional[float] = field(
        default=1.0,
        metadata={"help": "Lora layers percentage from 0-1. Default is 1.0, i.e., 100% lora layers will be applied. *FOR TEST ONLY DO NOT USE*"}
    )
    init_method: Optional[str] = field(
        default="kaiming_uniform_",
        metadata={"help": "The method to init LoRA_A. Choose from [ones_, normal_, kaiming_uniform_]. *FOR TEST ONLY DO NOT USE*"}
    )
    lora_modules: Optional[str] = field(
        default="all",
        metadata={"help": "Where to apply lora_modules. 1. [all|attention|mlp] - apply lora to [all|attention|mlp] linear layers. 2. module1,module2 - apply lora only to module 1 and moudule 2; moudles must be separated by ','."}
    )
    flash_attn: bool = field(
        default=False,
        metadata={"help": "Use flash attention? default = False"},
    )

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    hard_padding: bool = field(
        default=True,
        metadata={"help": "Force pad the length of input_ids (sequence length) to: source_max_len+target_max_len. Default = True"},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf|super-natural]"}
    )
    data_path: str = field(
        default=None,
        metadata={"help": "Where to find the dataset, download from huggingface if set to None."}
    )
    metrics_path: Optional[str] = field(
        default=None,
        metadata={"help": "Where to find the metrics, download from huggingface if set to None."}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for mmlu."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    quant: bool = field(
        default=False,
        metadata={"help": "Whether quant fine-tuning or not. Default False"}
    )
    double_quant: bool = field(
        default=False,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    max_memory_MB: int = field(
        default=40000,
        metadata={"help": "Free memory per gpu. *FOR TEST ONLY DO NOT USE*"}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    clean_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to clean the cache when training. *FOR TEST ONLY DO NOT USE*"}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='adamw_hf', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=False, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='linear', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis. Linear is better tuning the loss.'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=False, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=3, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    profiler: str = field(default=None, metadata={"help": 'To profile or not to profile, that is the question?'})
    profiler_warmup_step: int = field(default=2, metadata={"help": 'profiler_warmup_step. Default = 30 steps.'})


@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print_rank_0('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_accelerate_model(args, checkpoint_dir):

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
        use_fast=False, # Fast tokenizer giving issues.
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
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

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print_rank_0(
        f"==================="
        f"trainable params: {trainable_params} || "
        f"all params: {all_param}"
    )

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

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool
    hard_padding: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            padding='max_length' if self.hard_padding else False,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            padding='max_length' if self.hard_padding else False,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None

        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

def extract_super_natural_instructions_data(examples, extract_reformulations=False):
    out = {
        'input': [],
        'output': [],
    }
    print_rank_0(examples)
    for instance in examples:
        out['input'].append(instance['input'])
        out['output'].append(instance['output'])
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    print_rank_0(out)
    return out

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}

def local_dataset(dataset_name):
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - vicuna

    """
    def load_data(dataset_name, data_path=None):
        if data_path is None:
            if dataset_name == 'alpaca':
                return load_dataset("tatsu-lab/alpaca")
            elif dataset_name == 'alpaca-dummy':
                return load_dataset("Lohse/alpaca-dummy")
            elif dataset_name == 'alpaca-clean':
                return load_dataset("yahma/alpaca-cleaned")
            elif dataset_name == 'flanv2':
                return load_dataset("conceptofmind/FLAN_2022")
            elif dataset_name == 'chip2':
                return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
            elif dataset_name == 'self-instruct':
                return load_dataset("yizhongw/self_instruct", name='self_instruct')
            elif dataset_name == 'hh-rlhf':
                return load_dataset("Anthropic/hh-rlhf")
            elif dataset_name == 'longform':
                return load_dataset("akoksal/LongForm")
            elif dataset_name == 'oasst1':
                return load_dataset("timdettmers/openassistant-guanaco")
            elif dataset_name == 'vicuna':
                raise NotImplementedError("Vicuna data was not released.")
            else:
                if os.path.exists(dataset_name):
                    try:
                        args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                        full_dataset = local_dataset(dataset_name)
                        return full_dataset
                    except:
                        raise ValueError(f"Error loading dataset from {dataset_name}")
                else:
                    raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")
        else:
            if dataset_name in ['alpaca','alpaca-dummy','alpaca-clean','flanv2','hh-rlhf','longform','oasst1']:
                return load_dataset('json', data_dir=os.path.join(data_path,dataset_name))
            elif dataset_name == 'chip2':
                return load_dataset('json', data_dir=os.path.join(data_path,dataset_name), data_files='unified_chip2.jsonl')
            elif dataset_name == 'self-instruct':
                return load_dataset('json', data_dir=os.path.join(data_path,dataset_name), name='self_instruct')
            elif dataset_name == 'super-natural':
                return load_dataset('json', data_dir=os.path.join(data_path,dataset_name))
            elif dataset_name == 'vicuna':
                raise NotImplementedError("Vicuna data was not released.")
            else:
                if os.path.exists(dataset_name):
                    try:
                        args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                        full_dataset = local_dataset(dataset_name)
                        return full_dataset
                    except:
                        raise ValueError(f"Error loading dataset from {dataset_name}")
                else:
                    raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    def format_dataset(dataset, dataset_format):
        if (
            dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or dataset_format == 'alpaca-dummy' or
            (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean', 'alpaca-dummy'])
        ):
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif dataset_format == 'flanv2' or (dataset_format is None and args.dataset == 'flanv2'):
            dataset = dataset.map(lambda x: {'input': x['inputs'],'output': x['targets'],})
        elif dataset_format =='super-natural' or (dataset_format is None and args.dataset == 'super-natural'):
            dataset = dataset.map(remove_columns=['id'])
            # dataset = extract_super_natural_instructions_data(dataset)
            # dataset = Dataset.from_dict(dataset)
        elif dataset_format == 'input-output':
            # leave as is
            pass
        # Remove unused columns.
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
        )
        # print_rank_0(dataset["input"])
        # print_rank_0(dataset)
        return dataset

     # Load dataset.
    dataset = load_data(args.dataset, args.data_path)
    dataset = format_dataset(dataset, args.dataset_format)
    
    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            print_rank_0('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
        hard_padding=args.hard_padding,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )

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

def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    # merge all args
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print_rank_0(args)

    if args.flash_attn==True:
        print_rank_0("Use FLASH ATTENTION! Replacing......")
        replace_llama_attn_with_flash_attn()

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print_rank_0('Detected that training was already completed!')

    model, tokenizer = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    print_rank_0('model loaded')
    set_seed(args.seed)

    data_module = make_data_module(tokenizer=tokenizer, args=args)

    if not args.hard_padding:
        raise ValueError(f"--hard_padding must be True, or throughput may be incorrect.")
    num_gpus = torch.cuda.device_count()
    token_per_step = args.per_device_train_batch_size*num_gpus*(args.source_max_len+args.target_max_len)

    accuracy = evaluate.load("accuracy" if args.metrics_path is None else os.path.join(args.metrics_path,"accuracy"))
    
    # print({k:v for k,v in data_module.items() if k != 'predict_dataset'})

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )

    print_rank_0(model)
    
    try:
        print_rank_0(model.hf_device_map)
    except:
        print_rank_0("model has no hf_device_map.")

    # Callbacks
    if args.use_lora:
        pass
        # trainer.add_callback(SavePeftModelCallback)
    if args.do_mmlu_eval:
        if args.mmlu_dataset == 'mmlu-zs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': os.path.join(args.data_path,'mmlu/zero_shot_mmlu_val.json'),
                'test': os.path.join(args.data_path,'mmlu/zero_shot_mmlu_test.json'),
            })
            mmlu_dataset = mmlu_dataset.remove_columns('subject')
        # MMLU Five-shot (Eval/Test only)
        elif args.mmlu_dataset == 'mmlu' or args.mmlu_dataset == 'mmlu-fs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': os.path.join(args.data_path,'mmlu/five_shot_mmlu_val.json'),
                'test': os.path.join(args.data_path,'mmlu/five_shot_mmlu_test.json'),
            })
            # mmlu_dataset = mmlu_dataset.remove_columns('subject')
        mmlu_dataset = mmlu_dataset[args.mmlu_split]
        if args.max_mmlu_samples is not None:
            mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
        abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
        
        class MMLUEvalCallback(transformers.TrainerCallback):
            def __init__(self, key, lr, r):
                self.key = key
                self.lr = lr
                self.r = r
            
            def on_evaluate(self, args, state, control, model, **kwargs):
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() != 0:
                        return
                
                data_loader = trainer.get_eval_dataloader(mmlu_dataset)
                source_max_len = trainer.data_collator.source_max_len
                trainer.data_collator.source_max_len = args.mmlu_source_max_len
                trainer.model.eval()
                preds, refs = [], []
                loss_mmlu = 0
                for batch in tqdm(data_loader, total=len(data_loader)):
                    (loss, logits, labels) = trainer.prediction_step(trainer.model,batch,prediction_loss_only=False,)
                    # There are two tokens, the output, and eos token.
                    for i, logit in enumerate(logits):
                        label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                        logit_abcd = logit[label_non_zero_id-1][abcd_idx]
                        preds.append(torch.argmax(logit_abcd).item())
                    labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
                    refs += [abcd_idx.index(label) for label in labels.tolist()]
                    loss_mmlu += loss.item()
                # Extract results by subject.
                results = {'mmlu_loss':loss_mmlu/len(data_loader)}
                subject = mmlu_dataset['subject']
                subjects = {s:{'refs':[], 'preds':[]} for s in set(subject)}
                for s,p,r in zip(subject, preds, refs):
                    subjects[s]['preds'].append(p)
                    subjects[s]['refs'].append(r)
                subject_scores = []
                for subject in subjects:
                    if len(subjects[subject]['refs']) !=0 and len(subjects[subject]['preds']) !=0:
                        subject_score = accuracy.compute(
                            references=subjects[subject]['refs'],
                            predictions=subjects[subject]['preds']
                        )['accuracy']
                        results[f'mmlu_{args.mmlu_split}_accuracy_{subject}'] = subject_score
                        subject_scores.append(subject_score)
                results[f'mmlu_{args.mmlu_split}_accuracy'] = np.mean(subject_scores)
                
                mmlu_accuracy_dict={}
                mmlu_accuracy_dict[self.key]=""
                mmlu_accuracy_dict["lr"]=self.lr
                mmlu_accuracy_dict["r"]=self.r
                mmlu_accuracy_dict[f'mmlu_{args.mmlu_split}_accuracy'] = results[f'mmlu_{args.mmlu_split}_accuracy']
                
                safe_write2file(mmlu_accuracy_dict,"accuracy.txt")
                
                trainer.log(results)
                trainer.data_collator.source_max_len = source_max_len

        trainer.add_callback(MMLUEvalCallback(key="lora-fa" if (args.use_lora and args.fa) else "lora" if args.use_lora else "full",lr=args.learning_rate,r=args.lora_r))
    
    # empty cache every training step
    class EmptycacheCallback(transformers.TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.empty_cache()
            print_rank_0("Cache cleared [after step].")
        def on_train_begin(self, args, state, control, **kwargs):
            torch.cuda.empty_cache()
            print_rank_0("Cache cleared [before train].")
        def on_init_end(self, args, state, control, **kwargs):
            torch.cuda.empty_cache()
            print_rank_0("Cache cleared [after init].")
            
    if args.clean_cache:
        trainer.add_callback(EmptycacheCallback)

    class PT_ProfCallback(transformers.TrainerCallback):
        def __init__(self, prof):
            self.prof = prof

        def on_step_end(self, args, state, control, **kwargs):
            self.prof.step()

        def on_train_end(self, args, state, control, **kwargs):
            print_rank_0(self.prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

    class StepInfoCallback(transformers.TrainerCallback):
        def __init__(self, warmup_step, key, token_per_step):
            self.warmup_step = warmup_step
            self.key = key
            self.token_per_step = token_per_step
            self.step_times = []

        def on_step_begin(self, args, state, control, **kwargs):  
            self.start_time = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            self.end_time = time.time()
            self.step_times.append(self.end_time-self.start_time)

        def on_train_end(self, args, state, control, **kwargs):
            print_rank_0(self.step_times)
            
            mean_step_time = round(np.mean(self.step_times[self.warmup_step-1:-1]),3)
            std_step_time = round(np.std(self.step_times[self.warmup_step-1:-1]),3)
            profile_dict={}
            profile_dict["key"] = self.key
            profile_dict["step_time"] = f"{mean_step_time} s"
            profile_dict["step_time_std"] = f"{std_step_time} s"
            profile_dict["token/s"] = round(self.token_per_step/mean_step_time,2)
            profile_dict["mem"] = round((torch.cuda.mem_get_info(device=None)[1]-torch.cuda.mem_get_info(device=None)[0])/1024/1024/1024,2)
            safe_write2file(profile_dict, "profile.txt")

    if args.profiler==None:
        trainer.add_callback(StepInfoCallback(warmup_step=args.profiler_warmup_step, key=get_unique_key(args), token_per_step=token_per_step))
    elif args.profiler=="pytorch":
        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                on_trace_ready=torch.profiler.tensorboard_trace_handler('profile'),
                                profile_memory=True,
                                with_stack=True,
                                record_shapes=True)
        trainer.add_callback(PT_ProfCallback(prof=prof))

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    # for k, v in dtypes.items():
    #     print_rank_0(k, v)
    #     print_rank_0(v/total)

    all_metrics = {"run_name": args.run_name}
    print_rank_0("========START TRAIN========\n")
    # Training
    if args.do_train:
        print_rank_0("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        print_rank_0("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    # Prediction
    if args.do_predict:
        print_rank_0("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        print_rank_0(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))
    
if __name__ == "__main__":
    train()