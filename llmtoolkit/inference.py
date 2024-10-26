import torch
import os
import typing as tp
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

import wandb


def single_inference(model, tokenizer, input: str, task_type: str = "CausalLM", source_max_len: str = 512, target_max_len: str = 512):
    if task_type== 'CausalLM':
        inputs = tokenizer(
            input + " ",
            return_tensors="pt",
            max_length=source_max_len,
            truncation=True,
            return_token_type_ids=False,
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=target_max_len,
                eos_token_id=tokenizer.eos_token_id,
                top_p=0.95,
                temperature=0.8,
            )
        pred_text = tokenizer.decode(
            outputs.sequences[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
        )
    elif task_type == "ConditionalGeneration":
        inputs = tokenizer(input, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=target_max_len)
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return pred_text

def batched_inference(model, tokenizer, input: list, task_type: str = "CausalLM", source_max_len: str = 512, target_max_len: str = 512):
    if task_type== 'CausalLM':
        inputs = tokenizer(
            input + " ",
            return_tensors="pt",
            max_length=source_max_len,
            truncation=True,
            return_token_type_ids=False,
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=target_max_len,
                eos_token_id=tokenizer.eos_token_id,
                top_p=0.95,
                temperature=0.8,
            )
        pred_text = tokenizer.decode(
            outputs.sequences[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
        )
    elif task_type == "ConditionalGeneration":
        inputs = tokenizer(input, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=target_max_len)
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return pred_text