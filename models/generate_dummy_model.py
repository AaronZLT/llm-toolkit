import argparse
import shutil
import json
import os
import re
import sys
import types

import torch
import transformers
from transformers import LlamaForCausalLM, LlamaConfig

model_list = ["Llama-2-1.3b-hf","Llama-2-7b-hf","Llama-2-13b-hf","Llama-2-70b-chat-hf"]

def add_args(parser):
    parser.add_argument("--llm_benchmark_path", type=str, default=None, help="Base directory of llm-benchmark repository.")
    parser.add_argument("--model", type=str, nargs='+', default=["all"], help="Which model shall be generated? Default = all.")
    return parser

def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    if args.llm_benchmark_path is None:
        print("Please specify llm_benchmark_path. Exit.")
        return

    models_to_save = []
    if "all" in args.model:
        models_to_save = model_list.copy()
    else:
        for i in args.model:
            models_to_save.append(i)

    for m in models_to_save:
        cfg = LlamaConfig.from_pretrained(os.path.join(args.llm_benchmark_path,"models",m))
        print(f"Generating dummy checkpoint of {m}")
        model = LlamaForCausalLM(cfg)
        print(f"Saving dummy checkpoint to {os.path.join(args.llm_benchmark_path,'models',m)}")
        model.save_pretrained(os.path.join(args.llm_benchmark_path,"models",m))


if __name__ == "__main__":
    main()
