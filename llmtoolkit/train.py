import os
import json

import torch
import transformers
from transformers import (
    set_seed,
)
from accelerate.utils import DistributedType
from peft import PeftModel

from .arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    get_unique_key,
)
from .callbacks import (
    EmptycacheCallback,
    PT_ProfCallback,
    StepInfoCallback,
    DynamicSparseCallback,
    StaticSparseCallback,
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
    BaseSeq2SeqTrainer,
    Seq2SeqTrainer_lorafa,
)
from .utils import (
    print_rank_0,
)
from .memory_profiler import (
    export_memory_timeline_html,
)


def train(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    data_collator,
    training_args: TrainingArguments,
    key: str,
):
    set_seed(training_args.seed)
    if training_args.deepspeed:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    trainable_param, all_param, trainable_rate = print_trainable_parameters(
        model, training_args.debug_mode
    )

    if training_args.adamw_lorafa and isinstance(model, PeftModel):
        trainer = Seq2SeqTrainer_lorafa(
            lora_scale=model.peft_config["default"].lora_alpha
            / model.peft_config["default"].r,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
    else:
        trainer = BaseSeq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

    try:
        print_rank_0(f"device map: {model.hf_device_map}")
    except:
        pass

    # Callbacks
    if training_args.clean_cache:
        trainer.add_callback(EmptycacheCallback)

    if training_args.sparse:
        if training_args.sparse_type == "dynamic_sparse":
            trainer.add_callback(
                DynamicSparseCallback(
                    model=model,
                    sparsity_ratio=training_args.sparsity_ratio,
                    sparse_warmup_ratio=training_args.sparse_warmup_ratio,
                    sparse_warmup_steps=training_args.sparse_warmup_steps,
                )
            )
        elif training_args.sparse_type == "static_sparse":
            trainer.add_callback(StaticSparseCallback(model=model))

    trainer.add_callback(
        StepInfoCallback(
            trainer=trainer,
            warmup_step=training_args.profiler_warmup_step,
            key=key,
            trainable_param=trainable_param,
            step_log=training_args.profiler_step_log,
            output_dir=training_args.output_dir,
        )
    )

    if training_args.profiler == "deepspeed":
        return NotImplementedError("deepspeed is not supported")
    if training_args.profiler == "pytorch":
        torch.profiler._memory_profiler.MemoryProfileTimeline.export_memory_timeline_html = export_memory_timeline_html
        trainer.add_callback(
            PT_ProfCallback(
                warmup_step=training_args.profiler_warmup_step,
                key=key,
                output_dir=training_args.output_dir,
            )
        )

    all_metrics = {"run_name": training_args.run_name}

    if training_args.do_train:
        print_rank_0("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        save_strategy = (
            training_args.save_strategy
            if isinstance(training_args.save_strategy, str)
            else training_args.save_strategy.value
        )
        if save_strategy == "steps" or save_strategy == "epoch":
            trainer.save_metrics("train", metrics)
            trainer.save_state()
            if training_args.unify_save and isinstance(trainer.model, PeftModel):
                print_rank_0(
                    f"merged model will be save at {os.path.join(training_args.output_dir, 'merged')}"
                )
                trainer.model = trainer.model.merge_and_unload()
                trainer.save_model(os.path.join(training_args.output_dir, 'merged'))
            else:
                trainer.save_model()
        all_metrics.update(metrics)
    if training_args.do_eval:
        print_rank_0("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

    if training_args.do_train or training_args.do_eval:
        with open(os.path.join(training_args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


def train_cli(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
):
    # args, args_dict = get_args()
    # args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))
    set_seed(training_args.seed)
    key = get_unique_key(model_args, data_args, training_args)

    # no jit CPUAdamBuilder since it is too slow or may break the training process
    # deepspeed.ops.op_builder.CPUAdamBuilder().load()

    if training_args.deepspeed:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    checkpoint_dir, completed_training = get_last_checkpoint(training_args.output_dir)
    if completed_training:
        print_rank_0("Detected that training was already completed!")

    model, tokenizer = get_accelerate_model(model_args, training_args)
    model.config.use_cache = False
    print_rank_0("model loaded")
    print_rank_0(model)

    trainable_param, all_param, trainable_rate = print_trainable_parameters(
        model, training_args.debug_mode
    )

    data_module = build_data_module(
        tokenizer, data_args.dataset_name_or_path, data_args
    )

    trainer = BaseSeq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
    )

    if hasattr(model, "hf_device_map"):
        print_rank_0(model.hf_device_map)

    # Callbacks
    if training_args.clean_cache:
        trainer.add_callback(EmptycacheCallback)

    trainer.add_callback(
        StepInfoCallback(
            trainer=trainer,
            warmup_step=training_args.profiler_warmup_step,
            key=key,
            trainable_param=trainable_param,
            step_log=training_args.profiler_step_log,
            output_dir=training_args.output_dir,
        )
    )

    if training_args.profiler == "deepspeed":
        return NotImplementedError("deepspeed is not supported")
    if training_args.profiler == "pytorch":
        torch.profiler._memory_profiler.MemoryProfileTimeline.export_memory_timeline_html = export_memory_timeline_html
        trainer.add_callback(
            PT_ProfCallback(
                warmup_step=training_args.profiler_warmup_step,
                key=key,
                output_dir=training_args.output_dir,
            )
        )

    all_metrics = {"run_name": training_args.run_name}

    if training_args.do_train:
        print_rank_0("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        if (
            training_args.save_strategy is transformers.IntervalStrategy.STEPS
            or training_args.save_strategy is transformers.IntervalStrategy.EPOCH
        ):
            trainer.save_metrics("train", metrics)
            trainer.save_state()
            trainer.save_model()
        all_metrics.update(metrics)
    if training_args.do_eval:
        print_rank_0("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

    if training_args.do_train or training_args.do_eval:
        with open(os.path.join(training_args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))
