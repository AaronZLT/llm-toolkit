from tqdm.auto import tqdm
import math
import os
import argparse
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import (
    set_seed,
    get_scheduler,

)

import accelerate
from accelerate import Accelerator
from accelerate.utils import DistributedType

from .arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    GenerationArguments,
    get_unique_key,
)
from .dataset import (
    build_data_module,
)
from .model import (
    get_accelerate_model,
    get_last_checkpoint,
    print_trainable_parameters,
)
from .utils import (
    print_rank_0,
    hardware_info,
    get_rank,
)


r'''
For now, memory_tracer only print on rank_0
'''


class MemoryTracer():
    def __init__(self, args):
        self.memory = [0]
        self.memory_allocated = [0]
        self.total_mem = 0
        self.time_stamp = []
        self.name = []
        self.device_rank = get_rank()
        self.enable_trace = args.debug_mode == True
        self.output_dir = args.output_dir
        print_rank_0(
            f"Memory tracer successfully created on rank-{self.device_rank}")

    def trace(self):
        if not self.enable_trace:
            return
        mem = round(torch.cuda.memory_allocated(
            self.device_rank)/1024/1024/1024, 3)
        self.memory_allocated.append(mem)
        self.memory.append(round(mem-sum(self.memory), 3))

    def save(self, data):
        fig, ax = plt.subplots()
        ax.plot(data, marker='o')

        for index, value in enumerate(data):
            ax.text(index, value, str(value), fontsize=12, ha='right')

        ax.set_xlabel('stamp')
        ax.set_ylabel('Memory (GB)')
        ax.grid(True)
        plt.savefig(os.path.join(self.output_dir, "memory_alloc_stamp"))

    def print_alloc_mem(self):
        print_rank_0("+++++++++++++++++ Memory Trace START +++++++++++++++++")
        for index, value in enumerate(self.memory_allocated):
            print_rank_0(f"{index} Memory on rank-{self.device_rank}: {value}")
        print_rank_0("+++++++++++++++++ Memory Trace END +++++++++++++++++")
        self.save(self.memory_allocated)

    def print(self):
        print_rank_0("+++++++++++++++++ Memory Trace START +++++++++++++++++")
        for index, value in enumerate(self.memory):
            print_rank_0(f"{index} Memory on rank-{self.device_rank}: {value}")
        print_rank_0("+++++++++++++++++ Memory Trace END +++++++++++++++++")


def train_no_trainer():
    r'''
    This is the minimal custom training loop without Trainer. We mainly utilize accelerate for single/distribute training. In the context, loaded assets are further prepared by accelerate.
    '''

    r'''
    We first init the args, this involves
    1. mapping all input args to the argument dataclass
    2. init some fields in the args after the post-init  
    '''

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
    set_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.logging_strategy == "steps" or args.logging_strategy == "epoch":
        with_tracking = True
    else:
        with_tracking = False

    accelerator = (
        Accelerator(log_with=args.report_to,
                    project_dir=args.output_dir) if with_tracking else Accelerator()
    )
    memory_tracer = MemoryTracer(args)

    torch_overhead = torch.randn(1).cuda()
    memory_tracer.trace()

    # no jit CPUAdamBuilder since it is too slow or may break the training process
    # deepspeed.ops.op_builder.CPUAdamBuilder().load()

    hardware = hardware_info()
    n_gpus = hardware.n_gpus

    r'''
    The preparation follows the order below
    1. model and tokenizer
    2. data and data_collector
    3. optimizer
    '''

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print_rank_0('Detected that training was already completed!')

    model, tokenizer = get_accelerate_model(args, checkpoint_dir)
    model.config.use_cache = False
    print_rank_0('model loaded')
    print_rank_0(model)

    trainable_param, all_param, trainable_rate = print_trainable_parameters(
        model, True)

    memory_tracer.trace()

    # for now we only create dataloaders for train_dataset and eval_dataset
    data_module = build_data_module(tokenizer=tokenizer, args=args)

    train_dataloader = DataLoader(
        data_module["train_dataset"], shuffle=True, collate_fn=data_module["data_collator"], batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        data_module["eval_dataset"], collate_fn=data_module["data_collator"], batch_size=args.per_device_eval_batch_size)
    print_rank_0('data loaded')

    memory_tracer.trace()

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate)
    print_rank_0('optimizer loaded')

    memory_tracer.trace()

    r'''
    Now we define the max_steps and lr_scheduler, and feed all into accelerator
    1. lr_scheduler needs optimizer and max_steps, so we fisrt calculate max_steps based
    on the length of train_dataloader
    2. accelerator.prepare
    3. re-calculate the max_steps, since hr size of the train_dataloader may have changed

    We also need to define the save strategy durining training.
    '''

    overrode_max_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_steps is None or args.max_steps <= 0:
        args.max_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    print_rank_0('model, optimizer are wraped via accelerate loaded')

    memory_tracer.trace()

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_steps:
        args.max_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(
        args.max_steps / num_update_steps_per_epoch)

    def should_save(step) -> bool:
        if step <= 0:
            return False
        if args.save_strategy == "steps":
            return step % args.save_steps == 0
        if args.save_strategy == "epoch":
            return step % num_update_steps_per_epoch == 0
        return False

    r'''
    End of preparation
    '''

    r'''
    Define Training Loop

    1. in custom training loop, we do not support callbacks for now.
    '''
    if args.do_train:
        total_batch_size = args.per_device_train_batch_size * \
            accelerator.num_processes * args.gradient_accumulation_steps

        print_rank_0("***** Running training *****")
        print_rank_0(f"  Num examples = {len(data_module['train_dataset'])}")
        print_rank_0(f"  Num Epochs = {args.num_train_epochs}")
        print_rank_0(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        print_rank_0(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print_rank_0(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print_rank_0(f"  Total optimization steps = {args.max_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_steps),
                            disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                checkpoint_path = args.resume_from_checkpoint
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                # Sorts folders by date modified, most recent checkpoint is the last
                path = dirs[-1]
                checkpoint_path = path
                path = os.path.basename(checkpoint_path)

            print_rank_0(f"Resumed from checkpoint: {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(
                    training_difference.replace("epoch_", "")) + 1
                resume_step = None
                completed_steps = starting_epoch * num_update_steps_per_epoch
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = int(training_difference.replace(
                    "step_", "")) * args.gradient_accumulation_steps
                starting_epoch = resume_step // len(train_dataloader)
                completed_steps = resume_step // args.gradient_accumulation_steps
                resume_step -= starting_epoch * len(train_dataloader)

        # update the progress_bar
        progress_bar.update(completed_steps)

        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            if with_tracking:
                total_loss = 0
            if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader = accelerator.skip_first_batches(
                    train_dataloader, resume_step)
            else:
                active_dataloader = train_dataloader
            for step, batch in enumerate(active_dataloader):
                outputs = model(**batch)

                memory_tracer.trace()

                loss = outputs.loss
                # We keep track of the loss at each epoch
                if with_tracking:
                    total_loss += loss.detach().float()
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)

                memory_tracer.trace()

                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    if accelerator.is_local_main_process:
                        progress_bar.write(
                            f"Step-{step} training loss: {loss.detach().float()}")
                    memory_tracer.trace()

                if should_save(step):
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

                if completed_steps >= args.max_steps:
                    break

    # for now we do not support eval, just skip
    if args.do_eval:
        pass

    if with_tracking:
        accelerator.end_training()

    memory_tracer.print_alloc_mem()
