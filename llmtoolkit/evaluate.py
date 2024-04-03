import os
import time
import numpy as np
from tqdm import tqdm

import torch
import transformers
import evaluate

from .utils import (
    print_rank_0,
    safe_dict2file,
)
from .dataset import (
    IGNORE_INDEX,
)

def MMLU_eval(model, dataset):
    
    pass