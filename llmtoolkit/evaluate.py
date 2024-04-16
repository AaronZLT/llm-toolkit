import os
import time
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Sequence, Tuple, Union, List

import torch
import transformers
import evaluate

import lm_eval

from .utils import (
    print_rank_0,
    safe_dict2file,
)
from .dataset import (
    IGNORE_INDEX,
)

'''
Note:
1. The number of fewshot of truthfulqa_mc1 is set to 0, however, TruthfulQA is technically a 6-shot task in the Harness because each example is prepended with 6 Q/A pairs, even in the 0-shot setting.
2. To eval the baseline score of gsm8k, we are better to finetune the model on the full GSM8K training set for 50 epochs.
'''
task2shot = {
    "mmlu":5,
    "gsm8k":5,
    "winogrande":5,
    "piqa":5,
    "hellaswag":10,
    "truthfulqa_mc1":0,
    "arc_challenge":25,
    "openbookqa":5,
}

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
        print_rank_0(f"evaluating {tasks}")
        results.append(lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_name_or_path},tokenizer={model_name_or_path}",
            tasks=tasks,
            num_fewshot=shot,
            task_manager=task_manager,
            batch_size="auto"))
        
    for result in results:
        safe_dict2file(result["results"], os.path.join(output_dir,"eval_result.txt"))
