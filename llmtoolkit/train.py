import os
import json
import argparse
import numpy as np

import torch
import transformers
from transformers import (
    set_seed,
    Seq2SeqTrainer,
)
import deepspeed

from .arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    GenerationArguments,
)
from .callbacks import (
    MMLUEvalCallback,
    EmptycacheCallback,
    PT_ProfCallback,
    StepInfoCallback,
)
from .dataset import (
    make_data_module,
    generate_mmlu_dataset,
)
from .model import (
    get_accelerate_model,
    get_last_checkpoint,
    print_trainable_parameters,
)
from .utils import (
    print_rank_0,
    safe_dict2file,
    get_unique_key,
    hardware_info,
)

def train():    
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print_rank_0(args)

    set_seed(args.seed)
    # deepspeed.ops.op_builder.CPUAdamBuilder().load()
    hardware = hardware_info()
    n_gpus = hardware.n_gpus
    
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print_rank_0('Detected that training was already completed!')

    model, tokenizer = get_accelerate_model(args, checkpoint_dir)
    model.config.use_cache = False
    print_rank_0('model loaded')
    print_rank_0(model)

    data_module = make_data_module(tokenizer=tokenizer, args=args)

    if not args.hard_padding:
        raise ValueError(f"--hard_padding must be True, or throughput may be incorrect.")
    
    token_per_step = args.per_device_train_batch_size*n_gpus*(args.source_max_len+args.target_max_len)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )
    
    try:
        print_rank_0(model.hf_device_map)
    except:
        print_rank_0("model has no hf_device_map.")

    # Callbacks
    if args.use_lora:
        pass
        # trainer.add_callback(SavePeftModelCallback)

    if args.do_mmlu_eval:
        mmlu_dataset = generate_mmlu_dataset(args=args)
        trainer.add_callback(MMLUEvalCallback(args=args,key=get_unique_key(args),trainer=trainer,tokenizer=tokenizer,mmlu_dataset=mmlu_dataset))
        
    if args.clean_cache:
        trainer.add_callback(EmptycacheCallback)
        
    trainer.add_callback(StepInfoCallback(warmup_step=args.profiler_warmup_step, key=get_unique_key(args), token_per_step=token_per_step,output_dir=args.output_dir))

    if args.profiler=="deepspeed":
        return NotImplementedError("deepspeed is not supported")
    if args.profiler=="pytorch":
        trainer.add_callback(PT_ProfCallback(warmup_step=args.profiler_warmup_step, key=get_unique_key(args),output_dir=args.output_dir))
        
    print_trainable_parameters(model)

    all_metrics = {"run_name": args.run_name}
    
    print_rank_0("========START TRAIN========\n")
    if args.do_train:
        print_rank_0("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    if args.do_eval:
        print_rank_0("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
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
