import copy
import os
import re
from typing import Dict, Sequence
from dataclasses import dataclass
import pandas as pd
import random

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import datasets
from datasets import load_dataset, Dataset

from .arguments import (
    DataArguments,
)
from .utils import (
    print_rank_0,
)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<[PAD]>"


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    hard_padding: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [
            f"{self.tokenizer.bos_token}{example['input']}" for example in instances
        ]
        targets = [
            f"{example['output']}{self.tokenizer.eos_token}" for example in instances
        ]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            padding="max_length" if self.hard_padding else False,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            padding="max_length" if self.hard_padding else False,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt["input_ids"], tokenized_targets["input_ids"]
        ):
            input_ids.append(torch.tensor(tokenized_source + tokenized_target))
            if not self.train_on_source:
                labels.append(
                    torch.tensor(
                        [IGNORE_INDEX for _ in range(len(tokenized_source))]
                        + copy.deepcopy(tokenized_target)
                    )
                )
            else:
                labels.append(
                    torch.tensor(copy.deepcopy(tokenized_source + tokenized_target))
                )

        # Apply padding
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        data_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict["labels"] = labels
        return data_dict


r"""
Below is the train prompt and preprocess functions for generating train dataset.
Most of the training prompts are aligned with lm-eval, which is the same as 🤗 Open LLM Leaderboard.
"""


@dataclass
class SFTPrompt:
    question: str = "Question: {question}\nAnswer: "
    answer: str = "{answer}\n\n"
    instruction_choices: str = (
        "Question: {instruction}\n"
        
        "Below is a multiple choice question paired with choices."
        "Select a best answer for this question.\n\n"
        "#### Instruction:\n{instruction}\n\n#### Input:\n{input}\n\n#### Response: "
    )
    instruction_input: str = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "#### Instruction:\n{instruction}\n\n#### Input:\n{input}\n\n#### Response: "
    )
    instruction: str = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "#### Instruction:\n{instruction}\n\n#### Response: "
    )
    truthfulqa_6shot: str = "\
        Q: What is human life expectancy in the United States?\n\
        A: Human life expectancy in the United States is 78 years.\n\n\
        Q: Who was president of the United States in 1955?\n\
        A: Dwight D. Eisenhower was president of the United States in 1955.\n\n\
        Q: Which party did he belong to?\n\
        A: He belonged to the Republican Party.\n\n\
        Q: What is the square root of banana?\n\
        A: I have no comment.\n\n\
        Q: How does a telescope work?\n\
        A: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\n\
        Q: Where were the 1992 Olympics held?\n\
        A: The 1992 Olympics were held in Barcelona, Spain.\
        \n\nQ: {question}\nA:"


def extract_super_natural_instructions_data(examples, extract_reformulations=False):
    out = {
        "input": [],
        "output": [],
    }
    print_rank_0(examples)
    for instance in examples:
        out["input"].append(instance["input"])
        out["output"].append(instance["output"])
    if extract_reformulations:
        for example_reformulations in examples["reformulations"]:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out["input"].append(instance["instruction_with_input"])
                    out["output"].append(instance["output"])
    print_rank_0(out)
    return out


def preprocess_alpaca(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        if example.get("input", "") != "":
            return {
                "input": SFTPrompt.instruction_input.format(
                    instruction=example["instruction"], input=example["input"]
                ),
                "output": example["output"],
            }
        else:
            return {
                "input": SFTPrompt.instruction.format(
                    instruction=example["instruction"]
                ),
                "output": example["output"],
            }

    return dataset.map(_preprocess_doc)


def preprocess_gsm8k(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        return {
            "input": SFTPrompt.instruction.format(instruction=example["question"]),
            "output": example["answer"],
        }

    return dataset.map(_preprocess_doc)


def preprocess_truthfulqa_mc1(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        prompt_format_input = SFTPrompt.question
        prompt_format_output = SFTPrompt.answer
        return {
            "input": prompt_format_input.format(**example),
            "output": prompt_format_output.format(**example),
        }

    return dataset.map(_preprocess_doc)


def preprocess_hellaswag(dataset: datasets.Dataset) -> datasets.Dataset:
    new_dataset = []

    def _preprocess_text(text):
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def _process_doc(doc):
        data_slice = {}

        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": _preprocess_text(doc["activity_label"] + ": " + ctx),
            "choices": [_preprocess_text(ending) for ending in doc["endings"]],
            "gold": doc["label"],
        }
        for i in range(len(out_doc["choices"])):
            data_slice["input"] = f"{out_doc['query']}{out_doc['choices'][i]}"
            data_slice["output"] = "1" if out_doc["gold"] == str(i) else "-1"
            new_dataset.append(data_slice)

    dataset = dataset.map(_process_doc)
    new_dataset = Dataset.from_list(new_dataset).train_test_split(test_size=0.2)
    return new_dataset


def preprocess_wikitext2(dataset: datasets.Dataset) -> datasets.Dataset:
    new_dataset = []

    def _preprocess_doc(example):
        if len(example["text"]) > 1:
            data_slice = {}
            data_slice["input"] = example["text"]
            data_slice["output"] = ""
            new_dataset.append(data_slice)

    dataset.map(_preprocess_doc)
    new_dataset = Dataset.from_list(new_dataset).train_test_split(test_size=0.2)
    return new_dataset


def preprocess_e2e(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        return {"input": example["meaning_representation"], "output": example["target"]}

    return dataset.map(_preprocess_doc)


def preprocess_math(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        return {
            "input": SFTPrompt.question.format(example["question"]),
            "output": SFTPrompt.answer.format(example["answer"]),
        }

    return dataset.map(_preprocess_doc)


def preprocess_commonsense(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        return {"input": example["instruction"], "output": example["output"]}

    return dataset.map(_preprocess_doc)


def preprocess_chip2(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        return {
            "input": example["text"].split("\n<bot>: ")[0].replace("<human>: ", ""),
            "output": example["text"].split("\n<bot>: ")[1],
        }

    return dataset.map(_preprocess_doc)


def preprocess_selfinstruct(dataset: datasets.Dataset) -> datasets.Dataset:
    for old, new in [["prompt", "input"], ["completion", "output"]]:
        dataset = dataset.rename_column(old, new)
    return dataset


def preprocess_hhrlhf(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        return {"input": "", "output": example["chosen"]}

    return dataset.map(_preprocess_doc)


def preprocess_oasst1(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        return {"input": example["inputs"], "output": example["targets"]}

    return dataset.map(_preprocess_doc)


def preprocess_metamath(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        return {
            "input": SFTPrompt.instruction.format(instruction=example["query"]),
            "output": example["response"],
        }

    return dataset.map(_preprocess_doc)

def preprocess_wizardlm(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        return {
            "input": SFTPrompt.instruction.format(instruction=example["instruction"]),
            "output": example["output"],
        }

    return dataset.map(_preprocess_doc)

def preprocess_codefeedback(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        return {
            "input": SFTPrompt.instruction.format(instruction=example["query"]),
            "output": example["answer"],
        }

    return dataset.map(_preprocess_doc)

def preprocess_tulu_v3(dataset: datasets.Dataset) -> datasets.Dataset:
    new_dataset = []

    def _process_doc(example):
        if len(example['messages']) == 2:
            user_content = next(item['content'] for item in example['messages'] if item['role'] == 'user')
            assistant_content = next(item['content'] for item in example['messages'] if item['role'] == 'assistant')
            new_dataset.append({"input":SFTPrompt.instruction.format(instruction=user_content), "output":assistant_content})

    dataset = dataset.map(_process_doc)
    new_dataset = Dataset.from_list(new_dataset).train_test_split(test_size=0.2)
    return new_dataset

def preprocess_mmlu(dataset: datasets.Dataset) -> datasets.Dataset:
    new_dataset = []
    labels = ["A", "B", "C", "D"]
    def _preprocess_doc(example):
        slice = example["train"]
        input = f"Question: {slice['question']}\n"
        
        for label, choice in zip(labels, slice["choices"]):
            input = input + f"{label}. {choice}\n"
        input = input+"Answer: "
        output = f"{labels[slice['answer']]}. {slice['choices'][slice['answer']]}"

        new_dataset.append({"input":input, "output":output})

    dataset.map(_preprocess_doc)
    new_dataset = Dataset.from_list(new_dataset).train_test_split(test_size=0.1)
    return new_dataset


"""
Make dataset and collator for supervised fine-tuning.
Datasets are expected to have the following columns: { `input`, `output` }
1. in DATASETS_ARGS, please specify the dataset loading behavior, i.e., {"dataset name": ("dataset in huggingface", {dataset split})}, leave 'split' blank if load the dataset in default.
2. in FORMAT_FUNCTIONS, where to assign a map function (maps each entry in the dataset into trainable form, including cat with prompt, store in the 'input' and 'output' column.)
"""
DATASETS_ARGS = {
    "alpaca": ("tatsu-lab/alpaca", {}),
    "alpaca-dummy": ("Lohse/alpaca-dummy", {}),
    "alpaca-clean": ("yahma/alpaca-cleaned", {}),
    "alpaca-gpt4": ("vicgalle/alpaca-gpt4", {}),
    "flanv2": ("conceptofmind/FLAN_2022", {}),
    "chip2": ("laion/OIG", {"data_files": "unified_chip2.jsonl"}),
    "self-instruct": ("yizhongw/self_instruct", {"name": "self_instruct"}),
    "hh-rlhf": ("Anthropic/hh-rlhf", {}),
    "longform": ("akoksal/LongForm", {}),
    "oasst1": ("timdettmers/openassistant-guanaco", {}),
    "gsm8k": ("openai/gsm8k", {"name": "main"}),
    "hellaswag": ("Rowan/hellaswag", {}),
    "wikitext2": ("wikitext", {"name": "wikitext-2-raw-v1"}),
    "e2e": ("GEM/e2e_nlg", {}),
    "math": ("Lohse/math", {}),
    "commonsense": ("Lohse/commonsense", {}),
    "metamath": ("meta-math/MetaMathQA", {}),
    "metamath40k": ("meta-math/MetaMathQA-40K", {}),
    "wizardlm70k": ("WizardLMTeam/WizardLM_evol_instruct_70k", {}),
    "codefeedback": ("m-a-p/CodeFeedback-Filtered-Instruction", {}),
    "tuluv3": ("allenai/tulu-3-sft-mixture", {}),
    "mmlu": ("cais/mmlu", {"name": "auxiliary_train", "split": "train"}),
}

FORMAT_FUNCTIONS = {
    "input-output": lambda x: x,
    "alpaca": preprocess_alpaca,
    "alpaca-clean": preprocess_alpaca,
    "alpaca-gpt4": preprocess_alpaca,
    "alpaca-dummy": preprocess_alpaca,
    "gsm8k": preprocess_gsm8k,
    "hellaswag": preprocess_hellaswag,
    "chip2": preprocess_chip2,
    "self-instruct": preprocess_selfinstruct,
    "hh-rlhf": preprocess_hhrlhf,
    "oasst1": preprocess_oasst1,
    "wikitext2": preprocess_wikitext2,
    "e2e": preprocess_e2e,
    "math": preprocess_math,
    "commonsense": preprocess_commonsense,
    "metamath": preprocess_metamath,
    "metamath40k": preprocess_metamath,
    "wizardlm70k": preprocess_wizardlm,
    "codefeedback": preprocess_codefeedback,
    "tuluv3": preprocess_tulu_v3,
    "mmlu": preprocess_mmlu,
}


def local_dataset(dataset_name):
    if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith(".csv"):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith(".tsv"):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter="\t"))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset


def load_data(dataset_name_or_path):
    """
    2 ways to load dataset:
    (recommend) 1. load online.
    2. directly load dataset when 'dataset_name_or_path' is exists and not ready in this file (llm-toolkit).
    """
    if dataset_name_or_path in DATASETS_ARGS:
        dataset_info, kwargs = DATASETS_ARGS[dataset_name_or_path]
        return load_dataset(dataset_info, **kwargs)
    else:
        print_rank_0(
            f"The dataset {dataset_name_or_path} is not supported by llmtoolkit, use at your own risk."
        )
        if os.path.exists(dataset_name_or_path):
            try:
                full_dataset = local_dataset(dataset_name_or_path)
                return full_dataset
            except:
                raise ValueError(f"Error loading dataset '{dataset_name_or_path}'.")
        else:
            raise FileNotFoundError(
                f"Dataset '{dataset_name_or_path}' not found, the path {dataset_name_or_path} doesn't exist."
            )


def format_dataset(dataset_name_or_path, dataset):
    if dataset_name_or_path in FORMAT_FUNCTIONS:
        dataset = FORMAT_FUNCTIONS[dataset_name_or_path](dataset)
    else:
        print_rank_0(
            f"dataset format method for {dataset_name_or_path} is not implemented, trying default input-output format."
        )
        try:
            dataset = FORMAT_FUNCTIONS["input-output"](dataset)
        except:
            raise NotImplementedError(
                f"default input-output format has failed to format '{dataset_name_or_path}', please check the structure of '{dataset_name_or_path}'."
            )
    return dataset


def build_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    dataset_name_or_path,
    args: DataArguments = None,
) -> Dict:
    if args is None:
        args = DataArguments(dataset_name_or_path=dataset_name_or_path)

    dataset = load_data(dataset_name_or_path)
    dataset = format_dataset(dataset_name_or_path, dataset)

    if "eval" in dataset:
        eval_dataset = dataset["eval"]
    elif "test" in dataset:
        eval_dataset = dataset["test"]
    else:
        print_rank_0(
            f"Splitting train dataset in train and validation according to `eval_dataset_size`({args.eval_dataset_size})"
        )
        dataset = dataset["train"].train_test_split(
            test_size=args.eval_dataset_size, shuffle=True, seed=42
        )
        eval_dataset = dataset["test"]
    if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    eval_dataset = eval_dataset.map(
        lambda x: {"length": len(x["input"]) + len(x["output"])}
    )

    train_dataset = dataset["train"]
    if (
        args.max_train_samples is not None
        and len(train_dataset) > args.max_train_samples
    ):
        train_dataset = train_dataset.select(range(args.max_train_samples))
    train_dataset = train_dataset.map(
        lambda x: {"length": len(x["input"]) + len(x["output"])}
    )
    longest_sequence = max(train_dataset, key=lambda x: x["length"])
    longest_sequence_length = len(tokenizer(longest_sequence["input"])["input_ids"]) + len(tokenizer(longest_sequence["output"])["input_ids"])
    if (
        longest_sequence_length >= args.source_max_len + args.target_max_len
        and not args.hard_padding
    ):
        print_rank_0(
            f"WARNING: You choose not to pad all sequences to the max same length (max_input_token = source_max_len + target_max_len = {args.source_max_len+args.target_max_len}) since hard_padding is False. However, at least 1 sequence in the dataset has exceeded the max length ({longest_sequence_length}), which may ultimately cause OOM during the training. To avoid OOM, try few steps with --hard_padding True before training."
        )

    for index in random.sample(range(len(train_dataset)), 3):
        print_rank_0(f"Sample {index} of the training set:\n{train_dataset[index]}.")

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        hard_padding=args.hard_padding,
    )
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        predict_dataset=eval_dataset,
        data_collator=data_collator,
    )
