import os
import tempfile
import shutil

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

import llmtoolkit
from llmtoolkit import (
    infly_evaluate,
    safe_dict2file,
)


def find_adapter_model_paths(root_dir):
    matching_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "adapter_model.safetensors" in filenames:
            matching_paths.append(dirpath)
    return matching_paths


def eval_merge(
    task: str = "gsm8k",
    model_name_or_path: str = None,
    peft_name_or_path: str = None,
    load_in_4bit: bool = False,
):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(peft_name_or_path)
    # todo: check if len(tokenizer) == model's vocabsize
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, peft_name_or_path)
    model = model.merge_and_unload()
    temp_dir = tempfile.mkdtemp(dir=".")
    model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)
    del model
    del tokenizer
    torch.cuda.empty_cache()

    acc = infly_evaluate(
        task=task,
        model_name_or_path=temp_dir,
        load_in_4bit=load_in_4bit,
    )
    # acc = 0
    results = {}
    results["model"] = model_name_or_path
    results["peft"] = peft_name_or_path
    results["bits"] = 4 if load_in_4bit else 16
    results["task"] = task
    results["accuracy"] = acc
    safe_dict2file(results, "eval_result.txt")
    shutil.rmtree(temp_dir)


def eval(
    task: str = "gsm8k",
    model_name_or_path: str = None,
    peft_name_or_path: str = None,
    load_in_4bit: bool = False,
):
    acc = infly_evaluate(
        task="gsm8k",
        model_name_or_path=model_name_or_path,
        peft_name_or_path=peft_name_or_path,
        load_in_4bit=load_in_4bit,
    )
    # acc = 0
    results = {}
    results["model"] = model_name_or_path
    results["peft"] = peft_name_or_path
    results["bits"] = 4 if load_in_4bit else 16
    results["task"] = task
    results["accuracy"] = acc
    safe_dict2file(results, "eval_result.txt")

LLAMA2_7B="meta-llama/Llama-2-7b-hf"
ckpts = [
    "/hpc2hdd/home/lzhang330/llm-toolkit/tmp/metamath/output_lora_rank16_scale1/checkpoint-2250",
]

if __name__ == "__main__":
    for ckpt in ckpts:
        eval_merge(
            model_name_or_path=LLAMA2_7B,
            peft_name_or_path=ckpt,
            load_in_4bit=False,
        )

    for ckpt in ckpts:
        eval_merge(
            model_name_or_path=LLAMA2_7B,
            peft_name_or_path=ckpt,
            load_in_4bit=True,
        )
