import argparse
from typing import Optional
from dataclasses import dataclass, field

import transformers

from .utils import (
    print_rank_0,
    get_unique_key,
)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf"
    )
    peft_name_or_path: Optional[str] = field(
        default=None
    )
    unify_load: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to merge the adapter into base mode after loaded."}
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enables using Huggingface auth token from Git Credentials."}
    )
    peft: Optional[str] = field(
        default=None,
        metadata={
            "help": "To use peft, choose from [lora|lora-fa|vera|dora|prompt|embedding]"}
    )
    lora_r: int = field(
        default=1,
        metadata={"help": "lora rank, default = 1"},
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Lora dropout."}
    )
    lora_percent: Optional[float] = field(
        default=1.0,
        metadata={"help": "Lora layers percentage from 0-1. Default is 1.0, i.e., 100% lora layers will be applied. *FOR TEST ONLY DO NOT USE*"}
    )
    init_lora_weights: Optional[str] = field(
        default="kaiming_uniform_",
        metadata={
            "help": "The method to init LoRA_A. Choose from [ones_, normal_, kaiming_uniform_]. *FOR TEST ONLY DO NOT USE*"}
    )
    lora_modules: Optional[str] = field(
        default="all",
        metadata={
            "help": "Where to apply lora_modules. 1. [all|attention|mlp] - apply lora to [all|attention|mlp] linear layers. 2. module1,module2 - apply lora only to module 1 and moudule 2; moudles must be separated by ','."}
    )
    flash_attn: bool = field(
        default=False,
        metadata={"help": "Use flash attention? default = False"},
    )
    quant: bool = field(
        default=False,
        metadata={
            "help": "Quantize base model into quant_type data format. Default False"}
    )
    double_quant: bool = field(
        default=False,
        metadata={
            "help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={
            "help": "How many bits to use. In general we use bf16 training, here the bits is 16."}
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
        default=512,
        metadata={
            "help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=512,
        metadata={
            "help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    hard_padding: bool = field(
        default=False,
        metadata={
            "help": "Force pad the length of input_ids (sequence length) to: source_max_len + target_max_len. Set this to True may impact throughput, but is recommend in benchmark. Default = False."},
    )
    dataset_name_or_path: str = field(
        default='alpaca',
        metadata={
            "help": "Which dataset to finetune on. See dataset.py for options."}
    )
    metrics_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to find the metrics locally, otherwise it will download from huggingface if set to None."}
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to train on the input in addition to the target text. **Mostly used in pretraining."}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={
            "help": "Free memory per gpu. E.g., for H100 this should be set to 80000."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    clean_cache: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to clean the cache when training. *DEBUG ONLY - DO NOT USE*"}
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional descriptor for the run. Notably used for wandb logging."}
    )
    output_dir: str = field(default='default_output', metadata={
                            "help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='adamw_hf', metadata={
                       "help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={
                                             "help": 'The training batch size per GPU. Increase for better speed.'})
    auto_find_batch_size: bool = field(default=False, metadata={
                                       "help": 'Whether to find a batch size that will fit into memory automatically through exponential decay, avoiding CUDA Out-of-Memory errors. Requires accelerate to be installed (`pip install accelerate`)'})
    gradient_accumulation_steps: int = field(default=1, metadata={
                                             "help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(
        default=-1, metadata={"help": 'How many optimizer update steps to take. Works only when max_steps > 0.'})
    weight_decay: float = field(default=0.0, metadata={
                                "help": 'The L2 weight decay rate of AdamW'})
    learning_rate: float = field(default=0.0002, metadata={
                                 "help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={
                                        "help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={
                                 "help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=False, metadata={
                                         "help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={
                           "help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='linear', metadata={
                                   "help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis. Linear is better tuning the loss.'})
    warmup_ratio: float = field(default=0.03, metadata={
                                "help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={
                               "help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={
                                  "help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={
                               "help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={
                            "help": 'How often to save a model'})
    save_total_limit: int = field(default=10, metadata={
                                  "help": 'How many checkpoints to save before the oldest is overwritten'})
    profiler: str = field(default=None, metadata={
                          "help": 'To profile or not to profile, that is the question?'})
    profiler_warmup_step: int = field(
        default=30, metadata={"help": 'profiler_warmup_step. Default = 30 steps.'})
    profiler_step_log: bool = field(default=False, metadata={
                                    "help": 'Profile with detailed log (every step): train/eval loss, etc. Default = False'})
    device_map: str = field(default=None, metadata={
        "help": 'device map. Set to "auto" if enabled. Default is None.'})

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Turn on debug mode."}
    )


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
    min_new_tokens: Optional[int] = field(
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

def get_args():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, GenerationArguments))
    model_args, data_args, training_args, generation_args, extra_args = hfparser.parse_args_into_dataclasses(
        return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(
        **vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    # run_name is post inited in Transformers: if self.run_name is None: self.run_name = self.output_dir
    if args.run_name == args.output_dir:
        print_rank_0(
            f"Set run_name from '{args.output_dir}' to '{get_unique_key(args)}'")
        args.run_name = get_unique_key(args)
        training_args.run_name = get_unique_key(args)

    if args.output_dir == "default_output":
        print_rank_0(
            f"Set output_dir from 'default_output' to '{get_unique_key(args)}'")
        args.output_dir = get_unique_key(args)
        training_args.output_dir = get_unique_key(args)

    if args.deepspeed:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    print_rank_0(args)

    args_dict = {
        "model_args": model_args,
        "data_args": data_args,
        "training_args": training_args,
        "generation_args": generation_args,
        "extra_args": extra_args
    }
    return args, args_dict