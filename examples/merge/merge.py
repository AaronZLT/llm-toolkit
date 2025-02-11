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


def merge(
    base_model_name_or_path: str,
    peft_model_name_or_path: str,
    output_dir: str = None,
):
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    peft_tokenizer = AutoTokenizer.from_pretrained(peft_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
    if len(base_tokenizer) != len(peft_tokenizer):
        print_rank_0(
            f"Since the embedding of base model mismatch peft adapter ({len(base_tokenizer)} - {len(peft_tokenizer)}), resizing the embedding."
        )
        model.resize_token_embeddings(len(peft_tokenizer))

    model = PeftModel.from_pretrained(model, peft_model_name_or_path)
    model = model.merge_and_unload()
    if not output_dir:
        merge_temp_dir = tempfile.mkdtemp(dir=".")
        model.save_pretrained(merge_temp_dir)
        peft_tokenizer.save_pretrained(merge_temp_dir)
        print_rank_0(f"model has been save at {merge_temp_dir}.")
    else:
        model.save_pretrained(output_dir)
        peft_tokenizer.save_pretrained(output_dir)
        print_rank_0(f"model has been save at {output_dir}.")


if __name__ == "__main__":
    merge("/hpc2hdd/home/lzhang330/asset/models/Llama-2-7b-hf", "/hpc2hdd/home/lzhang330/workspace/llm-toolkit/tmp/tuluv3/lora-finetune-output/checkpoint-179218", "llama2-7b-merged-epoch2")
    merge("/hpc2hdd/home/lzhang330/asset/models/Llama-2-7b-hf", "/hpc2hdd/home/lzhang330/workspace/llm-toolkit/tmp/tuluv3/lora-finetune-output/checkpoint-268827", "llama2-7b-merged-epoch3")
