import os
import time
import numpy as np
from tqdm import tqdm
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import transformers
import evaluate
from accelerate import Accelerator

from .utils import (
    get_world_size,
    print_rank_0,
    safe_dict2file,
    rank_0,
)
from .dataset import (
    IGNORE_INDEX,
)
from .trainer import (
    Seq2SeqTrainer_llmtoolkit,
)
from .evaluate import (
    eval_perplexity,
)

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

class PT_ProfCallback(transformers.TrainerCallback):
    def __init__(self, warmup_step, key, output_dir:str = ""):
        self.prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=warmup_step, active=10, repeat=1),
            profile_memory=True,
            with_stack=True,
            record_shapes=True
            )
        self.warmup_step = warmup_step
        self.key = key
        self.output_dir = output_dir

    def on_train_begin(self, args, state, control, **kwargs):
        self.prof.start()

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()

    def on_train_end(self, args, state, control, **kwargs):
        self.prof.stop()
        self.dump_trace()
            
    @rank_0
    def dump_trace(self):
        self.prof.export_chrome_trace(os.path.join(self.output_dir,f"Trace_{self.key}_step_{self.warmup_step}_to_{self.prof.step_num}.json"))
        self.prof.export_memory_timeline(os.path.join(self.output_dir,f"Trace_{self.key}_step_{self.warmup_step}_to_{self.prof.step_num}.html"))
        
class StepInfoCallback(transformers.TrainerCallback):
    def __init__(self, trainer, warmup_step, key, output_dir:str = ""):
        self.trainer = trainer
        self.warmup_step = warmup_step
        self.key = key
        self.output_dir = output_dir
        self.step_times = []
    
    def get_token_per_step(self) -> List:
        seq = self.trainer.get_trained_seq()
        return seq

    def on_step_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        self.end_time = time.time()
        self.step_times.append(self.end_time-self.start_time)
        

    def on_train_end(self, args, state, control, **kwargs):
        accelerator = Accelerator()
        global_step = state.global_step

        mean_step_time = round(np.mean(self.step_times[self.warmup_step:]),3)
        std_step_time = round(np.std(self.step_times[self.warmup_step:]),3)

        total_FLOPs = state.total_flos
        FLOPs_per_step_per_device = total_FLOPs/global_step/get_world_size()
        FLOPS_per_device = FLOPs_per_step_per_device/mean_step_time

        seq = self.get_token_per_step()
        mean_seq = round(np.mean(seq[self.warmup_step:]),3)

        local_token_per_second = torch.tensor(round(mean_seq/mean_step_time,2), device = accelerator.device)
        all_token_per_second = accelerator.gather(local_token_per_second).sum().item()

        local_mem = torch.tensor((torch.cuda.mem_get_info(device=None)[1]-torch.cuda.mem_get_info(device=None)[0])/1024/1024/1024, device = accelerator.device)
        peak_mem = accelerator.gather(local_mem).max().item()

        profile_dict={}
        profile_dict["key"] = self.key
        profile_dict["batch_size"] = state.train_batch_size
        profile_dict["step_time (s)"] = mean_step_time
        profile_dict["step_time_std (s)"] = std_step_time
        profile_dict["token/s"] = round(all_token_per_second,2)
        profile_dict["FLOPs_per_step_per_device (TFLOPs)"] = round(FLOPs_per_step_per_device/1e12,3)
        profile_dict["FLOPS_per_device (TFLOPS)"] = round(FLOPS_per_device/1e12,3)
        profile_dict["mem (GB)"] = round(peak_mem,2)
        safe_dict2file(profile_dict, os.path.join(self.output_dir,"profiler.txt"))

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

class EvalCallback(transformers.TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs["metrics"]
        eval_loss = metrics["eval_loss"]
        ppl = eval_perplexity(eval_loss)
        kwargs["metrics"]["eval_perplexity"] = ppl
