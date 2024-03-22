import os
import json
import threading
import importlib
from importlib import metadata
import functools
from functools import wraps
import warnings
from typing import Dict, List
from packaging import version
from collections import Counter

import torch
from pynvml import *

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func

def rank_0(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        rank = 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()
        if rank == 0:
            return func(*args, **kwargs)
        # else:
        #     print(f"Skipping function {func.__name__} on rank {rank}")
        #     # You can return a default value if needed, for example:
        #     # return None
    return wrapper

def run_once(func):
    has_run = False
    def wrapper(*args, **kwargs):
        nonlocal has_run
        if not has_run:
            has_run = True
            return func(*args, **kwargs)
    return wrapper

@rank_0
def print_rank_0(message):
    print(message, flush=True)

@rank_0
def safe_dict2file(dictionary:Dict, filename):
    lock = threading.Lock()
    lock.acquire()
    with open(filename, 'a') as json_file:
        try:
            json.dump(dictionary, json_file, indent=4)
            json_file.write("\n")
        finally:
            lock.release()
            
@rank_0
def safe_list2file(l:List, filename):
    lock = threading.Lock()
    lock.acquire()
    with open(filename, 'a') as file:
        try:
            for i in l:
                file.write(i + "\n")
        finally:
            lock.release()

def safe_readjson(filename):
    lock = threading.Lock()
    lock.acquire()
    with open(filename, 'r') as json_file:
        try:
            d = json.load(json_file)
        finally:
            lock.release()    
    return d

def get_unique_key(args):
    model = args.model_name_or_path.split('/')[-1]
    
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

def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = metadata.version("torch")
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

class hardware_info:
    def __init__(self) -> None:
        self.n_gpus = 0
        self.gpu_info = {}
        self.n_xpus = 0
        self.xpu_info = {}
        if(torch.cuda.is_available()):
            self.get_gpu_info()
        if is_ipex_available() and torch.xpu.is_available():
            self.get_xpu_info()
        self.summary()
        
    def get_gpu_info(self):
        self.gpu_info = []
        nvmlInit()
        self.n_gpus = torch.cuda.device_count()
        for i in range(self.n_gpus):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)

            name = torch.cuda.get_device_name(i)
            
            total_memory = int(info.total / 1024 / 1024)
            free_memory = int(info.free / 1024 / 1024)
            used_memory = int(info.used / 1024 / 1024)
            
            self.gpu_info.append({
                "name":name,
                "total_memory":total_memory,
                "free_memory":free_memory,
                "used_memory":used_memory,
            })
    
        nvmlShutdown()
    
    # TODO
    def get_xpu_info(self):
        self.xpu_info = []
        self.n_xpus = torch.xpu.device_count()
        
    def summary(self):
        print_rank_0("")
        print_rank_0(f"Detected {self.n_gpus} GPU(s)")
        gpu_tuple_list = [(gpu['name'], gpu['total_memory']) for gpu in self.gpu_info]
        counter = Counter(gpu_tuple_list)
        for gpu, count in counter.items():
            name, memory = gpu
            print_rank_0(f"{count} x {name}, Memory: {memory}")
        
        print_rank_0("")
        print_rank_0(f"Detected {self.n_xpus} XPU(s)")
