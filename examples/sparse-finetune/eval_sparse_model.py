import os
import json
import torch
import transformers
from transformers import PretrainedConfig

from llmtoolkit import (
    print_rank_0,
    load,
    check_sparsity,
    infly_evaluate,
    safe_dict2file,
)


def eval_sparse_quant_model(
    task, base_model_name_or_path, peft_model_name_or_path, sparse_named_mask_path
):
    # 1. load model, apply sparse mask to the model, and save
    # since we only need to apply sparse mask and save the model, we do not
    # need to quantize it, thus load_in_4bit=False.
    model, tokenizer = load(
        base_model_name_or_path=base_model_name_or_path,
        peft_model_name_or_path=peft_model_name_or_path,
        load_in_4bit=False,
        sparse_named_mask_path=sparse_named_mask_path,
    )
    print_rank_0(model)
    check_sparsity(model)
    print_rank_0("Saving sparsed base model and tokenizer to sparsed/")
    model.base_model.save_pretrained("sparsed")
    tokenizer.save_pretrained("sparsed")

    # 2. eval model via infly_evaluate
    acc = infly_evaluate(
        task=task,
        model_name_or_path="sparsed",
        peft_name_or_path=peft_model_name_or_path,
        load_in_4bit=True,
    )

    # 3. save the end2end result
    results = {}
    results["model"] = base_model_name_or_path
    results["peft"] = peft_model_name_or_path
    results["bits"] = 4
    results["task"] = task
    results["accuracy"] = acc
    safe_dict2file(results, "eval_result.txt")


def eval_sparse_model(
    task, base_model_name_or_path, peft_model_name_or_path, sparse_named_mask_path
):
    # 1. load model, apply sparse mask to the model, and save
    # since we only need to apply sparse mask and save the model, we do not
    # need to quantize it, thus load_in_4bit=False.
    model, tokenizer = load(
        base_model_name_or_path=base_model_name_or_path,
        peft_model_name_or_path=peft_model_name_or_path,
        load_in_4bit=False,
        sparse_named_mask_path=sparse_named_mask_path,
    )
    print_rank_0(model)
    check_sparsity(model)
    print_rank_0("Saving sparsed base model and tokenizer to sparsed/")
    model.base_model.save_pretrained("sparsed")
    tokenizer.save_pretrained("sparsed")

    # 2. eval model via infly_evaluate
    acc = infly_evaluate(
        task=task,
        model_name_or_path="sparsed",
        peft_name_or_path=peft_model_name_or_path,
        load_in_4bit=False,
    )

    # 3. save the end2end result
    results = {}
    results["model"] = base_model_name_or_path
    results["peft"] = peft_model_name_or_path
    results["bits"] = 4
    results["task"] = task
    results["accuracy"] = acc
    safe_dict2file(results, "eval_result.txt")


if __name__ == "__main__":
    base_model_name_or_path = (
        "/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Llama-2-7b-chat-hf"
    )
    peft_model_name_or_path = "/mnt/sdb/zhanglongteng/workspace/llm-toolkit/examples/sparse-finetune/validate/checkpoint-100"
    sparse_named_mask_path = "/mnt/sdb/zhanglongteng/workspace/llm-toolkit/examples/sparse-finetune/validate/checkpoint-100/named_mask.pth"

    eval_sparse_quant_model(
        task="mmlu",
        base_model_name_or_path=base_model_name_or_path,
        peft_model_name_or_path=peft_model_name_or_path,
        sparse_named_mask_path=sparse_named_mask_path,
    )
