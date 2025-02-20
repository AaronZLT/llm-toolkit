import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from datasets import load_dataset, Dataset

from llmtoolkit import Prepeocess_dataset


LLAMA3_8B_INST = "/mnt/sdb/zhanglongteng/sdd/zhanglongteng/llama-3-8B-Instruct"
LLAMA2_7B_INST = "/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Llama-2-7b-hf"
DEEPSEEK_R1 = "/mnt/sdb/zhanglongteng/sdd/zhanglongteng/DeepSeek-R1"

tokenizer = AutoTokenizer.from_pretrained(LLAMA3_8B_INST)
tokenizer.padding_side = "right"

dataset = load_dataset("cais/mmlu", name = "auxiliary_train", split = "train")
print(dataset)
Prepeocess_dataset.preprocess_mmlu(dataset, tokenizer)