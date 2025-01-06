import os
import gc
import json
import time
import threading
import importlib
import datetime
from importlib import metadata
import functools
from functools import wraps
import warnings
from typing import List
from packaging import version
from collections import Counter
import matplotlib.pyplot as plt

import torch
from pynvml import *


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def get_rank():
    rank = 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()
    return rank


def get_world_size():
    world_size = (
        1
        if not torch.distributed.is_initialized()
        else torch.distributed.get_world_size()
    )
    return world_size


def rank_0(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if get_rank() == 0:
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
def create_timestamp():
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_time


@rank_0
def print_rank_0(message):
    print(f"\033[1;33m[llm toolkit]\033[0m: {message}", flush=True)


@rank_0
def safe_dict2file(dictionary: dict, filename: str):
    lock = threading.Lock()
    with lock:
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(filename, "a") as json_file:
            json.dump(dictionary, json_file, indent=4)
            json_file.write("\n")


@rank_0
def safe_list2file(source: List, filename):
    lock = threading.Lock()
    with lock:
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(filename, "a") as file:
            for i in source:
                file.write(i + "\n")


def safe_readjson(filename):
    lock = threading.Lock()
    with lock:
        with open(filename, "r") as json_file:
            d = json.load(json_file)
    return d


def clear_torch_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


@rank_0
def plot_xy(x, y, title):
    if len(x) != len(y):
        raise ValueError("length x != length y")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    return fig


@rank_0
def save_fig(fig, path):
    fig.savefig(path)


@rank_0
def require_lib(pylib: str):
    import importlib.util

    pylib_spec = importlib.util.find_spec(pylib)
    if pylib_spec is None:
        raise FileNotFoundError(f"{pylib} is required but not installed.")


def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return (
            str(version.parse(full_version).major)
            + "."
            + str(version.parse(full_version).minor)
        )

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
        self.gpu_info_detailed = []
        self.n_xpus = 0
        self.xpu_info = {}
        if torch.cuda.is_available():
            self.get_gpu_info()
        if is_ipex_available() and torch.xpu.is_available():
            self.get_xpu_info()
        self.summary()

    def get_gpu_info(self) -> dict:
        self.gpu_info = {}
        self.gpu_info_detailed = []
        nvmlInit()
        self.n_gpus = torch.cuda.device_count()
        for i in range(self.n_gpus):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)

            name = torch.cuda.get_device_name(i)

            total_memory = int(info.total / 1024 / 1024 / 1024)
            free_memory = int(info.free / 1024 / 1024 / 1024)
            used_memory = int(info.used / 1024 / 1024 / 1024)

            self.gpu_info_detailed.append(
                {
                    "name": name,
                    "total_memory": total_memory,
                    "free_memory": free_memory,
                    "used_memory": used_memory,
                }
            )
        self.gpu_info["GPU"] = self.gpu_info_detailed[0]["name"]
        self.gpu_info["Num of GPU"] = self.n_gpus
        self.gpu_info["Memory per GPU (GB)"] = self.gpu_info_detailed[0]["total_memory"]
        nvmlShutdown()
        return self.gpu_info

    # TODO
    def get_xpu_info(self):
        self.xpu_info = []
        self.n_xpus = torch.xpu.device_count()

    @rank_0
    def summary(self):
        print_rank_0(f"Detected {self.n_gpus} GPU(s)")
        gpu_tuple_list = [
            (gpu["name"], gpu["total_memory"]) for gpu in self.gpu_info_detailed
        ]
        counter = Counter(gpu_tuple_list)
        for gpu, count in counter.items():
            name, memory = gpu
            print_rank_0(f"{count} x {name}, Memory per GPU (GB): {memory}")

        print_rank_0(f"Detected {self.n_xpus} XPU(s)")


class global_system_info:
    def __init__(self):
        self.hardware = hardware_info()

        self.info = {
            "ngpu": self.hardware.n_gpus,
            "gpu_info": self.hardware.gpu_info,
            "overhead (s)": {},
        }

    def dump(self, output_dir):
        safe_dict2file(self.info, os.path.join(output_dir, "system_info.txt"))

    def __repr__(self):
        return f"{self.__class__.__name__}(info={self.info})"


gsi = global_system_info()


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print_rank_0(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        gsi.info["overhead (s)"][func.__name__] = end_time - start_time
        return result

    return wrapper
