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
    prune_magnitude,
)


def find_adapter_model_paths(root_dir):
    matching_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "adapter_model.safetensors" in filenames:
            matching_paths.append(dirpath)
    return matching_paths


def eval(
    task: str = "gsm8k",
    base_model_name_or_path: str = None,
    peft_model_name_or_path: str = None,
    load_in_4bit: bool = False,
    sparsity_ratio: float = None,
    structured_sparse: bool = False,
):
    temp_dirs = []  # reserved for temp dirs used in current eval process
    if structured_sparse:
        raise NotImplementedError("structured_sparse is not implemented.")
    if sparsity_ratio:
        assert sparsity_ratio < 1 and sparsity_ratio > 0
        sparse_temp_dir = tempfile.mkdtemp(dir=".")
        temp_dirs.append(sparse_temp_dir)
        model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        prune_magnitude(model, sparsity_ratio)
        model.save_pretrained(sparse_temp_dir)
        tokenizer.save_pretrained(sparse_temp_dir)
        base_model_name_or_path = sparse_temp_dir
        print_rank_0(f"base_model_name_or_path has changed to {sparse_temp_dir}.")

    if not peft_model_name_or_path:
        acc = infly_evaluate(
            task=task,
            model_name_or_path=base_model_name_or_path,
            load_in_4bit=load_in_4bit,
        )
    else:
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        peft_tokenizer = AutoTokenizer.from_pretrained(peft_model_name_or_path)
        if len(base_tokenizer) != len(peft_tokenizer):
            print_rank_0(
                f"Since the embedding of base model mismatch peft adapter ({len(base_tokenizer)} - {len(peft_tokenizer)}), merging."
            )
            model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
            model.resize_token_embeddings(len(peft_tokenizer))
            model = PeftModel.from_pretrained(model, peft_model_name_or_path)
            model = model.merge_and_unload()
            merge_temp_dir = tempfile.mkdtemp(dir=".")
            temp_dirs.append(merge_temp_dir)
            model.save_pretrained(merge_temp_dir)
            peft_tokenizer.save_pretrained(merge_temp_dir)
            del model
            del base_tokenizer
            del peft_tokenizer
            torch.cuda.empty_cache()
            acc = infly_evaluate(
                task=task,
                model_name_or_path=merge_temp_dir,
                load_in_4bit=load_in_4bit,
            )
        else:
            del base_tokenizer
            del peft_tokenizer
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
    results["sparse"] = sparsity_ratio
    results["task"] = task
    results["accuracy"] = acc
    safe_dict2file(results, "eval_result.txt")
    for t in temp_dirs:
        shutil.rmtree(t)


if __name__ == "__main__":
    # base model eval 16-bit and 4-bit
    eval(
        base_model_name_or_path="meta-llama/Llama-2-7b-hf",
        load_in_4bit=False,
    )
    eval(
        base_model_name_or_path="meta-llama/Llama-2-7b-hf",
        load_in_4bit=True,
    )

    # lora eval 16-bit and 4-bit
    eval(
        base_model_name_or_path="meta-llama/Llama-2-7b-hf",
        peft_model_name_or_path="llama2-7b.metamath40k.lora.checkpoint",
        load_in_4bit=False,
    )
    eval(
        base_model_name_or_path="meta-llama/Llama-2-7b-hf",
        peft_model_name_or_path="llama2-7b.metamath40k.lora.checkpoint",
        load_in_4bit=True,
    )

    # sparse eval 16-bit and 4-bit
    # it is suggest to keep the sparsity_ratio the same as the checkpoint
    eval(
        base_model_name_or_path="meta-llama/Llama-2-7b-hf",
        peft_model_name_or_path="llama2-7b.metamath40k.sparse0.5.lora.output/checkpoint-1000",
        sparsity_ratio=0.5,
        load_in_4bit=False,
    )
    eval(
        base_model_name_or_path="meta-llama/Llama-2-7b-hf",
        peft_model_name_or_path="llama2-7b.metamath40k.sparse0.5.lora.output/checkpoint-1000",
        sparsity_ratio=0.5,
        load_in_4bit=True,
    )
