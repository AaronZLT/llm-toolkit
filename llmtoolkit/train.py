import os
import json
import argparse
import numpy as np

import torch
from torch.profiler._memory_profiler import MemoryProfileTimeline
import transformers
from transformers import (
    set_seed,
    Seq2SeqTrainer,
)
import accelerate
from accelerate import Accelerator
from accelerate.utils import DistributedType

from .arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    GenerationArguments,
    get_args,
)
from .callbacks import (
    EmptycacheCallback,
    PT_ProfCallback,
    StepInfoCallback,
    EvalCallback,
)
from .dataset import (
    build_data_module,
)
from .model import (
    get_accelerate_model,
    get_last_checkpoint,
    print_trainable_parameters,
)
from .trainer import (
    Seq2SeqTrainer_llmtoolkit,
)
from .utils import (
    print_rank_0,
    safe_dict2file,
    get_unique_key,
    hardware_info,
    clear_torch_cache,
)
from .memory_profiler import (
    export_memory_timeline_html,
)


def train():
    args, args_dict = get_args()
    set_seed(args.seed)
    
    # no jit CPUAdamBuilder since it is too slow or may break the training process
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

    trainable_param, all_param, trainable_rate = print_trainable_parameters(
        model, args.debug_mode)

    data_module = build_data_module(tokenizer=tokenizer, args=args)

    trainer = Seq2SeqTrainer_llmtoolkit(
        model=model,
        tokenizer=tokenizer,
        args=args_dict["training_args"],
        **{k: v for k, v in data_module.items() if k != 'predict_dataset'},
    )

    try:
        print_rank_0(f"Premade device map: {model.hf_device_map}")
    except:
        print_rank_0("No hf_device_map has been set.")

    # Callbacks
    if args.clean_cache:
        trainer.add_callback(EmptycacheCallback)

    trainer.add_callback(StepInfoCallback(trainer=trainer, warmup_step=args.profiler_warmup_step, key=get_unique_key(
        args), trainable_param=trainable_param, step_log=args.profiler_step_log, output_dir=args.output_dir))

    if args.profiler == "deepspeed":
        return NotImplementedError("deepspeed is not supported")
    if args.profiler == "pytorch":
        MemoryProfileTimeline.export_memory_timeline_html = export_memory_timeline_html
        trainer.add_callback(PT_ProfCallback(
            warmup_step=args.profiler_warmup_step, key=get_unique_key(args), output_dir=args.output_dir))

    trainer.add_callback(EvalCallback())

    all_metrics = {"run_name": args.run_name}

    if args.do_train:
        print_rank_0("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        if args.save_strategy is transformers.IntervalStrategy.STEPS or args.save_strategy is transformers.IntervalStrategy.EPOCH:
            trainer.save_metrics("train", metrics)
            trainer.save_state()
            trainer.save_model()
        all_metrics.update(metrics)
    if args.do_eval:
        print_rank_0("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    if args.do_predict:
        print_rank_0("*** Predict ***")
        prediction_output = trainer.predict(
            test_dataset=data_module['predict_dataset'], metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100,
                               predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(
                    example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        print_rank_0(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))
