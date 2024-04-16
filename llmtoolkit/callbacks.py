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
    print_rank_0,
    safe_dict2file,
)
from .dataset import (
    IGNORE_INDEX,
)
from .trainer import (
    Seq2SeqTrainer_llmtoolkit,
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
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                self.prof.export_chrome_trace(os.path.join(self.output_dir,f"Trace_{self.key}_step_{self.warmup_step}_to_{self.prof.step_num}.json"))
                self.prof.export_memory_timeline(os.path.join(self.output_dir,f"Trace_{self.key}_step_{self.warmup_step}_to_{self.prof.step_num}.html"))
        else:
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

        seq = self.get_token_per_step()
        mean_seq = round(np.mean(seq[self.warmup_step:]),3)

        mean_step_time = round(np.mean(self.step_times[self.warmup_step:]),3)
        std_step_time = round(np.std(self.step_times[self.warmup_step:]),3)

        local_token_per_second = torch.tensor(round(mean_seq/mean_step_time,2), device = accelerator.device)
        all_token_per_second = accelerator.gather(local_token_per_second).sum().item()

        local_mem = torch.tensor((torch.cuda.mem_get_info(device=None)[1]-torch.cuda.mem_get_info(device=None)[0])/1024/1024/1024, device = accelerator.device)
        peak_mem = accelerator.gather(local_mem).max().item()

        profile_dict={}
        profile_dict["key"] = self.key
        profile_dict["step_time (s)"] = mean_step_time
        profile_dict["step_time_std (s)"] = std_step_time
        profile_dict["token/s"] = round(all_token_per_second,2)
        profile_dict["mem (GB)"] = round(peak_mem,2)
        safe_dict2file(profile_dict, os.path.join(self.output_dir,"profiler.txt"))