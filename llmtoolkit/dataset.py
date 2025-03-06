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


default_template = """
{%- if messages[0]['role'] == 'system' %}
    {% set system_message = messages[0]['content'] %}
    {% set loop_messages = messages[1:] %}
{% else %}
    {% set system_message = '' %}
    {% set loop_messages = messages %}
{% endif -%}
{{ bos_token }}

{% if system_message != '' %}
{{ system_message }}

{% endif %}
{% for message in loop_messages %}
{{ message['content'] }}

{% if not loop.last %}
{% elif message['role'] == 'assistant' %}{{ eos_token }}
{% endif %}
{% endfor %}
"""


def apply_chat_template_to_train(
    chat: Sequence[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
):
    # todo: check if the chat is system/user/assitant format
    if tokenizer.chat_template:
        _source = tokenizer.apply_chat_template(
            chat[:-1], tokenize=False, add_generation_prompt=True
        )
    else:
        _source = tokenizer.apply_chat_template(
            chat[:-1],
            tokenize=False,
            add_generation_prompt=True,
            chat_template=default_template,
        )
    _target = f"{chat[-1]['content']}\n\n{tokenizer.eos_token}"

    return _source, _target


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


class PrepareDataset:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.DATASET_FUNCTIONS = {
            "mmlu": self.prepare_mmlu,
            "default": self.prepare_default,
            "alpaca": self.prepare_alpaca,
            "gsm8k": self.prepare_gsm8k,
            "tuluv3": self.prepare_tuluv3,
            "metamath": self.prepare_metamath,
            "metamath40k": self.prepare_metamath40k,
            "wizardlm70k": self.prepare_wizardlm,
            "codefeedback": self.prepare_codefeedback,
        }

    def register_function(self, name: str, func):
        """
        Dynamically register a new dataset function.
        """
        self.DATASET_FUNCTIONS[name] = func

    def prepare(self, dataset_name_or_path: str):
        """
        Prepare the dataset based on the dataset name or path.
        """
        if dataset_name_or_path in self.DATASET_FUNCTIONS:
            return self.DATASET_FUNCTIONS[dataset_name_or_path]()

        # Handle default case when no specific format function is found
        print_rank_0(
            f"Dataset method for '{dataset_name_or_path}' is not implemented. Using default load and format method."
        )
        try:
            return self.DATASET_FUNCTIONS["default"](dataset_name_or_path)
        except KeyError:
            raise NotImplementedError(
                f"Default method failed for '{dataset_name_or_path}'. Please check the structure of the dataset. To use default method, the dataset should has 'input' and 'output' columns."
            )

    def prepare_default(self, dataset_name_or_path: str) -> datasets.Dataset:
        """
        Prepare the default dataset given dataset_name_or_path.
        """
        system_message = "You are a helpful assistant. Below is an instruction that describes a task. Write a response that appropriately completes the request."
        dataset = load_dataset(dataset_name_or_path)

        def _preprocess_doc(example):
            input_str = SFTPrompt.instruction.format(instruction=example["input"])
            output_str = example["output"]
            chat = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_str},
                {"role": "assistant", "content": output_str},
            ]
            _source, _target = apply_chat_template_to_train(chat, self.tokenizer)
            return {"input": _source, "output": _target}

        return dataset.map(_preprocess_doc, num_proc=gsi.info["n_cpus"])

    def prepare_mmlu(self) -> datasets.Dataset:
        """
        MMLU dataset.
        """
        system_message = (
            "You are a helpful assistant who can select the most appropriate answer to each user question from a set of given multiple-choice options. "
            "When responding, only provide the correct answer in the format: 'A. text', 'B. text', 'C. text', or 'D. text'. "
            "Do not include any explanation, reasoning, or additional text."
        )
        labels = ["A", "B", "C", "D"]
        dataset = load_dataset("cais/mmlu", name="all")
        dataset["train"] = dataset.pop("auxiliary_train")

        def preprocess_test():
            def _preprocess_example(example):
                pass

            # dataset["test"] = dataset["test"].map(
            #     _preprocess_example, num_proc=gsi.info["n_cpus"]
            # )

        def preprocess_train():
            def _preprocess_example(example):
                input_str = f"Question: {example['question']}\n"
                for label, choice in zip(labels, example["choices"]):
                    input_str += f"{label}. {choice}\n"
                input_str += "\nAnswer: "

                output_str = f"{labels[example['answer']]}. {example['choices'][example['answer']]}"

                chat = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": input_str},
                    {"role": "assistant", "content": output_str},
                ]
                _source, _target = apply_chat_template_to_train(chat, self.tokenizer)

                return {"input": _source, "output": _target}

            dataset["train"] = dataset["train"].map(
                _preprocess_example, num_proc=gsi.info["n_cpus"]
            )
            dataset["test"] = dataset["test"].map(
                _preprocess_example, num_proc=gsi.info["n_cpus"]
            )

        preprocess_train()
        preprocess_test()
        return dataset

    def prepare_alpaca(self) -> datasets.Dataset:
        """
        Preprocess the Alpaca dataset.
        """
        system_message = (
            "You are a helpful assistant. Below is an instruction that describes a task, "
            "paired with an input that provides further context. Write a response that appropriately completes the request."
        )

        dataset = load_dataset("vicgalle/alpaca-gpt4")

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
            chat = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_str},
                {"role": "assistant", "content": output_str},
            ]
            _source, _target = apply_chat_template_to_train(chat, self.tokenizer)

            return {"input": _source, "output": _target}

        return dataset.map(_preprocess_example, num_proc=gsi.info["n_cpus"])

    # todo: use a better system prompt for gsm8k
    def prepare_gsm8k(self) -> datasets.Dataset:
        """
        Preprocess the GSM8K dataset.
        """
        system_message = "You are a helpful assistant. Below is an instruction that describes a task. Write a response that appropriately completes the request."
        dataset = load_dataset("openai/gsm8k", name="main")

        def _preprocess_doc(example):
            input_str = SFTPrompt.instruction.format(instruction=example["question"])
            output_str = example["answer"]
            chat = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_str},
                {"role": "assistant", "content": output_str},
            ]
            _source, _target = apply_chat_template_to_train(chat, self.tokenizer)

            return {"input": _source, "output": _target}

        return dataset.map(_preprocess_doc, num_proc=gsi.info["n_cpus"])

    def prepare_metamath(self) -> datasets.Dataset:
        """
        Preprocess the Metamath dataset.
        """
        system_message = "You are a helpful assistant. Below is an instruction that describes a task. Write a response that appropriately completes the request."
        dataset = load_dataset("meta-math/MetaMathQA")

        def _preprocess_doc(example):
            input_str = SFTPrompt.instruction.format(instruction=example["query"])
            output_str = example["response"]
            chat = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_str},
                {"role": "assistant", "content": output_str},
            ]
            _source, _target = apply_chat_template_to_train(chat, self.tokenizer)
            return {"input": _source, "output": _target}

        return dataset.map(_preprocess_doc, num_proc=gsi.info["n_cpus"])

    def prepare_metamath40k(self) -> datasets.Dataset:
        """
        Preprocess the Metamath dataset.
        """
        system_message = "You are a helpful assistant. Below is an instruction that describes a task. Write a response that appropriately completes the request."
        dataset = load_dataset("meta-math/MetaMathQA-40K")

        def _preprocess_doc(example):
            input_str = SFTPrompt.instruction.format(instruction=example["query"])
            output_str = example["response"]
            chat = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_str},
                {"role": "assistant", "content": output_str},
            ]
            _source, _target = apply_chat_template_to_train(chat, self.tokenizer)
            return {"input": _source, "output": _target}

        return dataset.map(_preprocess_doc, num_proc=gsi.info["n_cpus"])

    def prepare_codefeedback(self) -> datasets.Dataset:
        """
        Preprocess the codefeedback dataset.
        """
        system_message = "You are a helpful assistant. Below is an instruction that describes a code generation task. Write a answer that appropriately completes the request."
        dataset = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction")

        def _preprocess_doc(example):
            input_str = SFTPrompt.instruction.format(instruction=example["query"])
            output_str = example["answer"]
            chat = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_str},
                {"role": "assistant", "content": output_str},
            ]
            _source, _target = apply_chat_template_to_train(chat, self.tokenizer)

            return {"input": _source, "output": _target}

        return dataset.map(_preprocess_doc, num_proc=gsi.info["n_cpus"])

    def prepare_wizardlm(self) -> datasets.Dataset:
        """
        Preprocess the wizardlm dataset.
        """
        system_message = "You are a helpful assistant. Below is an instruction that describes a task. Write a response that appropriately completes the request."
        dataset = load_dataset("WizardLMTeam/WizardLM_evol_instruct_70k")

        def _preprocess_doc(example):
            input_str = SFTPrompt.instruction.format(instruction=example["instruction"])
            output_str = example["output"]
            chat = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_str},
                {"role": "assistant", "content": output_str},
            ]
            _source, _target = apply_chat_template_to_train(chat, self.tokenizer)

            return {"input": _source, "output": _target}

        return dataset.map(_preprocess_doc, num_proc=gsi.info["n_cpus"])

    def prepare_tuluv3(self) -> datasets.Dataset:
        """
        Preprocess the tulu v3 dataset.
        """
        dataset = load_dataset("allenai/tulu-3-sft-mixture")

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


def draw_number_line_with_highlight(title, unsorted_data, given_number):
    data = sorted(unsorted_data)

    n = len(data)
    percentages = [int(p * (n - 1) / 100) for p in range(0, 101, 10)]
    values = [data[i] for i in percentages]
    percentage_labels = [f"{p}%" for p in range(0, 101, 10)]

    smaller_count = sum(1 for x in data if x < given_number)
    percentage_of_given = smaller_count / n * 100
    given_percentage_label = f"{round(percentage_of_given)}%"

    line_length = 50
    highlight_position = int((percentage_of_given / 100) * line_length)

    result = f"{title}\n"

    percentage_line = ""
    segment_length = line_length // (len(percentage_labels) - 1)
    for i, percent in enumerate(percentage_labels):
        if i == 0:
            percentage_line += percent.ljust(1)
        else:
            percentage_line += percent.rjust(segment_length)
    result += percentage_line + "\n"

    arrow_line_top = ""
    for i in range(len(percentage_labels)):
        if i == 0:
            arrow_line_top += "↓".ljust(1)
        else:
            arrow_line_top += "↓".rjust(segment_length)
    result += arrow_line_top + "\n"

    result += "-" * line_length + "\n"

    arrow_line_bottom = ""
    for i in range(len(values)):
        if i == 0:
            arrow_line_bottom += "↑".ljust(1)
        else:
            arrow_line_bottom += "↑".rjust(segment_length)
    result += arrow_line_bottom + "\n"

    value_line = ""
    for i, value in enumerate(values):
        if i == 0:
            value_line += str(value).ljust(1)
        else:
            value_line += str(value).rjust(segment_length)
    result += value_line + "\n"

    highlight_line = " " * highlight_position + "|"
    result += highlight_line + "\n"

    label_line = " " * highlight_position + f"{given_number} ({given_percentage_label})"
    result += label_line + "\n"

    print_rank_0(result)
    return data[int(90 * (n - 1) / 100)]


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


def build_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    dataset_name_or_path,
    args: DataArguments = None,
) -> Dict:
    preparedataset = PrepareDataset(tokenizer=tokenizer)

    if args is None:
        args = DataArguments(dataset_name_or_path=dataset_name_or_path)

    dataset = preparedataset.prepare(dataset_name_or_path)
    print_rank_0(dataset)

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

    train_dataset = dataset["train"]
    if (
        args.max_train_samples is not None
        and len(train_dataset) > args.max_train_samples
    ):
        train_dataset = train_dataset.select(range(args.max_train_samples))
    train_dataset = train_dataset.map(
        lambda x: {
            "input_length": len(tokenizer(x["input"])["input_ids"]),
            "output_length": len(tokenizer(x["output"])["input_ids"]),
            "length": len(tokenizer(x["input"])["input_ids"])
            + len(tokenizer(x["output"])["input_ids"]),
        },
        num_proc=gsi.info["n_cpus"],
    )
    longest_sequence = max(train_dataset, key=lambda x: x["length"])
    longest_sequence_length = longest_sequence["length"]
    if (
        longest_sequence_length >= args.source_max_len + args.target_max_len
        and not args.hard_padding
    ):
        print_rank_0(
            f"WARNING: You choose not to pad all sequences to the max same length (max_input_token = source_max_len + target_max_len = {args.source_max_len + args.target_max_len}) since hard_padding is False. However, at least 1 sequence in the dataset has exceeded the max length ({longest_sequence_length}), which may ultimately cause OOM during the training. To avoid OOM, try few steps with --hard_padding True before training."
        )
    input_length = train_dataset["input_length"]
    output_length = train_dataset["output_length"]
    length = train_dataset["length"]
    recmd_input_length = draw_number_line_with_highlight(
        "input", input_length, args.source_max_len
    )
    recmd_output_length = draw_number_line_with_highlight(
        "output", output_length, args.target_max_len
    )
    recmd_seq_length = draw_number_line_with_highlight(
        "sequence", length, args.source_max_len + args.target_max_len
    )
    print_rank_0(
        f"To cover 90% of input, output and overall sequence length, you may consider setting source_max_len >= {recmd_input_length}, target_max_len >= {recmd_output_length}, and source_max_len + target_max_len >= {recmd_seq_length} respectively."
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
