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
from typing import Optional, Dict, Sequence, Tuple, Union, List
import numpy as np
from tqdm import tqdm
import logging
import pandas as pd
import importlib
from packaging import version
from packaging.version import parse
import argparse

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer
)
from transformers.activations import ACT2FN

import bitsandbytes as bnb
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

import lm_eval

'''
param
'''

# llama2chat = "/hpc2hdd/home/lzhang330/ssd_workspace/models/llama-2-7b-chat-hf"
# llama2 = "/hpc2hdd/home/lzhang330/ssd_workspace/models/Llama-2-7b-hf"
llama = "/mnt/sdb/zhanglongteng/data2/share/llama-1/llama-7b-hf"
llama2 = "/mnt/sdb/zhanglongteng/data2/share/zhanglongteng_A6000/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(
    llama,
    padding_side="right",
    use_fast=False, # Fast tokenizer giving issues.
    tokenizer_type='llama', # Needed for HF name change
)

'''
tokenizer test
'''

abcd_idx = [
    tokenizer("A").input_ids[1],
    tokenizer("B").input_ids[1],
    tokenizer("C").input_ids[1],
    tokenizer("D").input_ids[1],
]

print(abcd_idx)

'''
functions
'''

task2shot = {
    "mmlu":5,
    "gsm8k":5,
    "winogrande":5,
    "piqa":5,
    "hellaswag":10,
    "truthfulqa_mc1":0,
    "arc_challenge":25,
}
# task2shot = {
#     "winogrande":5,
#     "hellaswag":10,
# }
'''
Note:
1. The number of fewshot of truthfulqa_mc1 is set to 0, however, TruthfulQA is technically a 6-shot task in the Harness because each example is prepended with 6 Q/A pairs, even in the 0-shot setting.
2. To eval the baseline score of gsm8k, we are better to finetune the model on the full GSM8K training set for 50 epochs
'''
def safe_dict2file(dictionary:Dict, filename):
    lock = threading.Lock()
    lock.acquire()
    with open(filename, 'a') as json_file:
        try:
            json.dump(dictionary, json_file, indent=4)
            json_file.write("\n")
        finally:
            lock.release()

#TODO: batch_size = "auto", results["results"], rank 0 execute      
def eval(model_name_or_path:str, task2shot:Dict[str,int], output_dir:str = "output"):
    task_manager = lm_eval.tasks.TaskManager()
    
    shot2task={}
    results=[]
    for key, value in task2shot.items():
        if value not in shot2task:
            shot2task[value] = [key]
        else:
            shot2task[value].append(key)
    
    for shot,tasks in shot2task.items():
        print(f"evaluating shot = {shot}")
        results.append(lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_name_or_path},tokenizer={model_name_or_path}",
            tasks=tasks,
            num_fewshot=shot,
            task_manager=task_manager,
            batch_size="auto"))
        
    # results = lm_eval.simple_evaluate(
    #     model="hf",
    #     model_args=f"pretrained={model_name_or_path},tokenizer={model_name_or_path}",
    #     tasks=[task],
    #     num_fewshot=num_fewshot,
    #     task_manager=task_manager,
    #     batch_size="auto")
    
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            for result in results:
                safe_dict2file(result["results"],os.path.join(output_dir,"eval_result.txt"))
    else:
        for result in results:
            safe_dict2file(result["results"],os.path.join(output_dir,"eval_result.txt"))

if __name__ == "__main__":
    # eval(llama2,["mmlu"])
    # eval(llama2,["gsm8k"])
    # eval(llama2,["piqa","winogrande"])
    eval(llama2,task2shot)