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
    gsi,
)

IGNORE_INDEX = -100


def get_eos_token(tokenizer: transformers.PreTrainedTokenizer) -> str:
    """
    get eos token, since some specific tokenizers have more than one eos tokens.
    For example, llama-3 has two eos tokens, <|end_of_text|> and <|eot_id|>, where <|eot_id|> is used in multi-turn chat.

    <|eot_id|> for llama-3
    <|end|> for phi3
    tokenizer.eos_token for other models
    """

    if "<|eot_id|>" in tokenizer.get_vocab():
        return "<|eot_id|>"
    else:
        return tokenizer.eos_token


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    hard_padding: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [example["input"] for example in instances]
        targets = [example["output"] for example in instances]
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


@dataclass
class SFTPrompt:
    """
    Below is the train prompt and preprocess functions for generating train dataset.
    Most of the training prompts are aligned with lm-eval, which is the same as 🤗 Open LLM Leaderboard.
    """

    question: str = "Question: {question}\n\nAnswer: "
    answer: str = "{answer}\n\n"
    instruction_input: str = (
        "Instruction: {instruction}\n\nInput: {input}\n\nResponse: "
    )
    instruction: str = "Instruction: {instruction}\n\nResponse: "
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


def apply_chat_template(
    system_content: str,
    input_content: str,
    output_content: str,
    eos_token: str,
    tokenizer: transformers.PreTrainedTokenizer,
):
    if tokenizer.chat_template:
        chat = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": input_content},
        ]
        _source = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        _target = f"{output_content}{eos_token}"
    else:
        _source = f"{tokenizer.bos_token}\n\n{system_content}\n\n{input_content}"
        _target = f"{output_content}{eos_token}"

    return _source, _target


class PreprocessDataset:
    """
    A processed dataset is supposed to have the following format:

    system: This is the system message for LLMs, such as an introduction to the training data.
    input: This is the input with basic instructions.
    output: Contains only the output.

    The system, input, and output will be sent to `apply_chat_template`.
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.eos_token = get_eos_token(self.tokenizer)
        self.FORMAT_FUNCTIONS = {
            "mmlu": self._preprocess_mmlu,
            "alpaca": self._preprocess_alpaca,
            "gsm8k": self._preprocess_gsm8k,
            "tuluv3": self._preprocess_tuluv3,
            "wikitext2": self._preprocess_wikitext2,
            "math": self._preprocess_math,
            "metamath": self._preprocess_metamath,
            "metamath40k": self._preprocess_metamath,
            "wizardlm70k": self._preprocess_wizardlm,
            "codefeedback": self._preprocess_codefeedback,
            "input-output": self._preprocess_default,
        }

    def register_format_function(self, name: str, func):
        """
        Dynamically register a new dataset preprocessing function.
        """
        self.FORMAT_FUNCTIONS[name] = func

    def preprocess(self, dataset_name_or_path: str, dataset: datasets.Dataset):
        """
        Preprocess the dataset based on the dataset name or path.
        """
        if dataset_name_or_path in self.FORMAT_FUNCTIONS:
            return self.FORMAT_FUNCTIONS[dataset_name_or_path](dataset)

        # Handle default case when no specific format function is found
        print_rank_0(
            f"Dataset format method for '{dataset_name_or_path}' is not implemented. Using default input-output format."
        )
        try:
            return self.FORMAT_FUNCTIONS["input-output"](dataset)
        except KeyError:
            raise NotImplementedError(
                f"Default input-output format failed for '{dataset_name_or_path}'. Please check the structure of the dataset."
            )

    def _preprocess_default(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """
        Preprocess the input-output pair dataset.
        """
        system_message = "You are a helpful assistant. Below is an instruction that describes a task. Write a response that appropriately completes the request."

        def _preprocess_doc(example):
            input_str = SFTPrompt.instruction.format(instruction=example["input"])
            output_str = example["output"]
            _source, _target = apply_chat_template(
                system_message, input_str, output_str, self.eos_token, self.tokenizer
            )
            return {"input": _source, "output": _target}

        return dataset.map(_preprocess_doc, num_proc=gsi.info["n_cpus"])

    def _preprocess_mmlu(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """
        Preprocess the MMLU dataset.
        """
        system_message = (
            "You are a helpful assistant who can select the most appropriate "
            "answer to each user question from a set of given multiple-choice options."
        )
        labels = ["A", "B", "C", "D"]

        def _preprocess_example(example):
            question_data = example["train"]
            input_str = f"Question: {question_data['question']}\n"
            for label, choice in zip(labels, question_data["choices"]):
                input_str += f"{label}. {choice}\n"
            input_str += "Answer: "

            output_str = f"{labels[question_data['answer']]}. {question_data['choices'][question_data['answer']]}"

            _source, _target = apply_chat_template(
                system_message, input_str, output_str, self.eos_token, self.tokenizer
            )
            return {"input": _source, "output": _target}

        return dataset.map(
            _preprocess_example, num_proc=gsi.info["n_cpus"]
        ).train_test_split(test_size=0.1)

    def _preprocess_alpaca(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """
        Preprocess the Alpaca dataset.
        """
        system_message = (
            "You are a helpful assistant. Below is an instruction that describes a task, "
            "paired with an input that provides further context. Write a response that appropriately completes the request."
        )

        def _preprocess_example(example):
            if example.get("input", "").strip():
                input_str = SFTPrompt.instruction_input.format(
                    instruction=example["instruction"], input=example["input"]
                )
            else:
                input_str = SFTPrompt.instruction.format(
                    instruction=example["instruction"]
                )
            output_str = example["output"]

            _source, _target = apply_chat_template(
                system_message, input_str, output_str, self.eos_token, self.tokenizer
            )
            return {"input": _source, "output": _target}

        return dataset.map(_preprocess_example, num_proc=gsi.info["n_cpus"])

    # todo: use a better system prompt for gsm8k
    def _preprocess_gsm8k(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """
        Preprocess the GSM8K dataset.
        """
        system_message = "You are a helpful assistant. Below is an instruction that describes a task. Write a response that appropriately completes the request."

        def _preprocess_doc(example):
            input_str = SFTPrompt.instruction.format(instruction=example["question"])
            output_str = example["answer"]
            _source, _target = apply_chat_template(
                system_message, input_str, output_str, self.eos_token, self.tokenizer
            )
            return {"input": _source, "output": _target}

        return dataset.map(_preprocess_doc, num_proc=gsi.info["n_cpus"])

    # todo
    def _preprocess_wikitext2(self, dataset: datasets.Dataset) -> datasets.Dataset:
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

    # todo: use a better system prompt for math
    def _preprocess_math(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """
        Preprocess the MATH dataset.
        """
        system_message = "You are a helpful assistant. Answer to each user question."

        def _preprocess_doc(example):
            input_str = SFTPrompt.question.format(example["question"])
            output_str = example["answer"]
            _source, _target = apply_chat_template(
                system_message, input_str, output_str, self.eos_token, self.tokenizer
            )
            return {"input": _source, "output": _target}

        return dataset.map(_preprocess_doc, num_proc=gsi.info["n_cpus"])

    def _preprocess_metamath(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """
        Preprocess the Metamath dataset.
        """
        system_message = "You are a helpful assistant. Below is an instruction that describes a task. Write a response that appropriately completes the request."

        def _preprocess_doc(example):
            input_str = SFTPrompt.instruction.format(instruction=example["query"])
            output_str = example["response"]
            _source, _target = apply_chat_template(
                system_message, input_str, output_str, self.eos_token, self.tokenizer
            )
            return {"input": _source, "output": _target}

        return dataset.map(_preprocess_doc, num_proc=gsi.info["n_cpus"])

    def _preprocess_wizardlm(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """
        Preprocess the wizardlm dataset.
        """
        system_message = "You are a helpful assistant. Below is an instruction that describes a task. Write a response that appropriately completes the request."

        def _preprocess_doc(example):
            input_str = SFTPrompt.instruction.format(instruction=example["instruction"])
            output_str = example["output"]
            _source, _target = apply_chat_template(
                system_message, input_str, output_str, self.eos_token, self.tokenizer
            )
            return {"input": _source, "output": _target}

        return dataset.map(_preprocess_doc, num_proc=gsi.info["n_cpus"])

    def _preprocess_codefeedback(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """
        Preprocess the codefeedback dataset.
        """
        system_message = "You are a helpful assistant. Below is an instruction that describes a code generation task. Write a answer that appropriately completes the request."

        def _preprocess_doc(example):
            input_str = SFTPrompt.instruction.format(instruction=example["query"])
            output_str = example["answer"]
            _source, _target = apply_chat_template(
                system_message, input_str, output_str, self.eos_token, self.tokenizer
            )
            return {"input": _source, "output": _target}

        return dataset.map(_preprocess_doc, num_proc=gsi.info["n_cpus"])

    def _preprocess_tuluv3(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """
        Preprocess the tulu v3 dataset.
        """
        if not self.tokenizer.chat_template:
            raise NotImplementedError(
                "Tulu v3 is a chat dataset, thus a chat template is required."
            )

        def _preprocess_doc(example):
            input_str = example["messages"]
            _source = self.tokenizer.apply_chat_template(
                input_str, tokenize=False, add_generation_prompt=False
            )
            return {"input": _source, "output": ""}

        return dataset.map(_preprocess_doc, num_proc=gsi.info["n_cpus"])


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
    "wikitext2": ("EleutherAI/wikitext_document_level", {"name": "wikitext-2-raw-v1"}),
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


def build_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    dataset_name_or_path,
    args: DataArguments = None,
) -> Dict:
    preprocessdataset = PreprocessDataset(tokenizer=tokenizer)

    if args is None:
        args = DataArguments(dataset_name_or_path=dataset_name_or_path)

    dataset = load_data(dataset_name_or_path)
    dataset = preprocessdataset.preprocess(dataset_name_or_path, dataset)

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
        lambda x: {"length": len(x["input"]) + len(x["output"])},
        num_proc=gsi.info["n_cpus"],
    )

    train_dataset = dataset["train"]
    if (
        args.max_train_samples is not None
        and len(train_dataset) > args.max_train_samples
    ):
        train_dataset = train_dataset.select(range(args.max_train_samples))
    train_dataset = train_dataset.map(
        lambda x: {"length": len(x["input"]) + len(x["output"])},
        num_proc=gsi.info["n_cpus"],
    )
    longest_sequence = max(train_dataset, key=lambda x: x["length"])
    longest_sequence_length = len(
        tokenizer(longest_sequence["input"])["input_ids"]
    ) + len(tokenizer(longest_sequence["output"])["input_ids"])
    if (
        longest_sequence_length >= args.source_max_len + args.target_max_len
        and not args.hard_padding
    ):
        print_rank_0(
            f"WARNING: You choose not to pad all sequences to the max same length (max_input_token = source_max_len + target_max_len = {args.source_max_len + args.target_max_len}) since hard_padding is False. However, at least 1 sequence in the dataset has exceeded the max length ({longest_sequence_length}), which may ultimately cause OOM during the training. To avoid OOM, try few steps with --hard_padding True before training."
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
