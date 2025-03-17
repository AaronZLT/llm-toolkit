import os
import tempfile
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from llmtoolkit import (
    infly_evaluate,
    safe_dict2file,
    print_rank_0,
    load,
    check_embedding_need_to_resize,
)


def eval_base_model(
    task: str,
    base_model_name_or_path: str,
    load_in_4bit: bool = False,
):
    acc = infly_evaluate(
        task=task,
        model_name_or_path=base_model_name_or_path,
        load_in_4bit=load_in_4bit,
    )

    results = {}
    results["model"] = base_model_name_or_path
    results["bits"] = 4 if load_in_4bit else 16
    results["task"] = task
    results["accuracy"] = acc
    safe_dict2file(results, "eval_result.txt")


def eval_peft_model(
    task: str,
    base_model_name_or_path: str,
    peft_model_name_or_path: str,
    load_in_4bit: bool = False,
):
    resized = check_embedding_need_to_resize(
        base_model_name_or_path=base_model_name_or_path,
        peft_model_name_or_path=peft_model_name_or_path,
    )
    if resized:
        acc = infly_evaluate(
            task=task,
            model_name_or_path=resized,
            peft_name_or_path=peft_model_name_or_path,
            load_in_4bit=load_in_4bit,
        )
    else:
        acc = infly_evaluate(
            task=task,
            model_name_or_path=base_model_name_or_path,
            peft_name_or_path=peft_model_name_or_path,
            load_in_4bit=load_in_4bit,
        )
    results = {}
    results["model"] = base_model_name_or_path
    results["peft"] = peft_model_name_or_path
    results["bits"] = 4 if load_in_4bit else 16
    results["task"] = task
    results["accuracy"] = acc
    safe_dict2file(results, "eval_result.txt")


if __name__ == "__main__":
    eval_base_model(
        task="mmlu",
        base_model_name_or_path="/mnt/sdb/zhanglongteng/sdd/zhanglongteng/llama-2-7b-chat-hf",
        load_in_4bit=False,
    )

    eval_base_model(
        task="mmlu",
        base_model_name_or_path="/mnt/sdb/zhanglongteng/sdd/zhanglongteng/llama-3-8B-Instruct",
        load_in_4bit=True,
    )

    eval_peft_model(
        task="mmlu",
        base_model_name_or_path="/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Llama-2-7b-chat-hf",
        peft_model_name_or_path="/mnt/sdb/zhanglongteng/workspace/llm-toolkit/tmp/test_eval_peft/validate/checkpoint-5",
        load_in_4bit=True,
    )
