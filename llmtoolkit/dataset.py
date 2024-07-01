import copy
import os
import re
from typing import Dict, Sequence
from dataclasses import dataclass
import sys
import pandas as pd
import random

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import datasets
from datasets import load_dataset, Dataset

from .utils import (
    print_rank_0,
)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<[PAD]>"

"""
Below is the train prompt and preprocess functions for generating train dataset.
Most of the training prompts are aligned with lm-eval, which is the same as 🤗 Open LLM Leaderboard.
"""

@dataclass
class SFTPrompt:
    default_input: str = ("Question: {question}\nAnswer: ")
    default_output: str = ("{answer}\n\n")
    alpaca_input: str = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n#### Input:\n{input}\n\n#### Response: "
            )
    alpaca_noinput: str = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n#### Response: "
            )
    truthfulqa_6shot:str = ('\
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
        \n\nQ: {question}\nA:')
    math_input: str = ("Question: {instruction}\nAnswer: ")
    math_output: str = ("{output}\n\n")
    
def extract_super_natural_instructions_data(examples, extract_reformulations=False):
    out = {
        'input': [],
        'output': [],
    }
    print_rank_0(examples)
    for instance in examples:
        out['input'].append(instance['input'])
        out['output'].append(instance['output'])
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    print_rank_0(out)
    return out

def preprocess_alpaca(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        if example.get("input", "") != "":
            prompt_format = SFTPrompt.alpaca_input
        else:
            prompt_format = SFTPrompt.alpaca_noinput
        return {'input': prompt_format.format(**example)}
    return dataset.map(_preprocess_doc)

def preprocess_gsm8k(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        prompt_format_input = SFTPrompt.default_input
        prompt_format_output = SFTPrompt.default_output
        return {'input': prompt_format_input.format(**example), 'output': prompt_format_output.format(**example)}
    return dataset.map(_preprocess_doc)

def preprocess_truthfulqa_mc1(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        prompt_format_input = SFTPrompt.default_input
        prompt_format_output = SFTPrompt.default_output
        return {'input': prompt_format_input.format(**example), 'output': prompt_format_output.format(**example)}
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
        if len(example["text"])>1:
            data_slice = {}
            data_slice["input"] = example["text"]
            data_slice["output"] = ""
            new_dataset.append(data_slice)
    dataset.map(_preprocess_doc)
    new_dataset = Dataset.from_list(new_dataset).train_test_split(test_size=0.2)
    return new_dataset

def preprocess_e2e(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        return {'input': example['meaning_representation'], 'output': example['target']}
    return dataset.map(_preprocess_doc)

def preprocess_math(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        return {'input': SFTPrompt.math_input.format(**example), 'output': SFTPrompt.math_output.format(**example)}
    return dataset.map(_preprocess_doc)

def preprocess_commonsense(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        return {'input': example['instruction'], 'output': example['output']}
    return dataset.map(_preprocess_doc)

def preprocess_chip2(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        return {'input': example['text'].split('\n<bot>: ')[0].replace('<human>: ', ''), 'output': example['text'].split('\n<bot>: ')[1]}
    return dataset.map(_preprocess_doc)

def preprocess_selfinstruct(dataset: datasets.Dataset) -> datasets.Dataset:
    for old, new in [["prompt", "input"], ["completion", "output"]]:
        dataset = dataset.rename_column(old, new)
    return dataset

def preprocess_hhrlhf(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        return {'input': '', 'output': example['chosen']}
    return dataset.map(_preprocess_doc)

def preprocess_oasst1(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        return {'input': '', 'output': example['text']}
    return dataset.map(_preprocess_doc)

def preprocess_oasst1(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess_doc(example):
        return {'input': example['inputs'], 'output': example['targets']}
    return dataset.map(_preprocess_doc)


def local_dataset(dataset_name):
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset

"""
Below is for constructing dataset and datacollator module.
"""

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool
    hard_padding: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            padding='max_length' if self.hard_padding else False,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            padding='max_length' if self.hard_padding else False,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None

        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - vicuna
    """
    DATASETS_ARGS = {
        'alpaca': ("tatsu-lab/alpaca", {}),
        'alpaca-dummy': ("Lohse/alpaca-dummy", {}),
        'alpaca-clean': ("yahma/alpaca-cleaned", {}),
        'alpaca-gpt4': ("vicgalle/alpaca-gpt4", {}),
        'flanv2': ("conceptofmind/FLAN_2022", {}),
        'chip2': ("laion/OIG", {'data_files': 'unified_chip2.jsonl'}),
        'self-instruct': ("yizhongw/self_instruct", {'name': 'self_instruct'}),
        'hh-rlhf': ("Anthropic/hh-rlhf", {}),
        'longform': ("akoksal/LongForm", {}),
        'oasst1': ("timdettmers/openassistant-guanaco", {}),
        'gsm8k': ("gsm8k", {'name': "main"}),
        'hellaswag': ("Rowan/hellaswag", {}),
        'wikitext2': ("wikitext", {'name': "wikitext-2-raw-v1"}),
        'e2e': ("GEM/e2e_nlg", {}),
        'math': ("Lohse/math", {}),
        'commonsense': ("Lohse/commonsense", {}),
    }

    FORMAT_FUNCTIONS = {
        'input-output': lambda x: x,
        'alpaca': preprocess_alpaca,
        'alpaca-clean': preprocess_alpaca,
        'alpaca-gpt4': preprocess_alpaca,
        'alpaca-dummy': preprocess_alpaca,
        'gsm8k': preprocess_gsm8k,
        'hellaswag': preprocess_hellaswag,
        'chip2': preprocess_chip2,
        'self-instruct': preprocess_selfinstruct,
        'hh-rlhf': preprocess_hhrlhf,
        'oasst1': preprocess_oasst1,
        'wikitext2': preprocess_wikitext2,
        'e2e': preprocess_e2e,
        'math': preprocess_math,
        'commonsense': preprocess_commonsense,
    }

    def load_data(dataset_name, local_data_path=None):
        """
        3 ways to load dataset:
        (recommend) 1. load online (local_data_path is not given).
        2. load offline from json. *Note that not all datasets are avaliable in local json.
        3. directly load dataset when 'dataset_path/dataset_name' is exists.
        """
        if dataset_name in DATASETS_ARGS:
            dataset_info, kwargs = DATASETS_ARGS[dataset_name]
            if local_data_path is None:
                return load_dataset(dataset_info, **kwargs)
            else:
                return load_dataset('json', data_dir=os.path.join(local_data_path, dataset_name), **kwargs)
        else:
            print_rank_0(f"The dataset {dataset_name} is not supported by llmtoolkit, use at your own risk.")
            dataset_path = os.path.join("" if local_data_path is None else local_data_path, dataset_name)
            if os.path.exists(dataset_path):
                try:
                    args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                    full_dataset = local_dataset(dataset_path)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset '{dataset_name}' from {dataset_path}")
            else:
                raise FileNotFoundError(f"Dataset '{dataset_name}' not found, the path {local_dataset_path} doesn't exist.")

    def format_dataset(dataset_name, dataset_format, dataset):
        if dataset_name in FORMAT_FUNCTIONS:
            dataset = FORMAT_FUNCTIONS[dataset_name](dataset)
        elif dataset_format in FORMAT_FUNCTIONS:
            dataset = FORMAT_FUNCTIONS[dataset_format](dataset)
        else:
            raise NotImplementedError(f"Dataset '{dataset_name}' or dataset format '{dataset_format}' is not supported.")
        return dataset

    dataset = load_data(args.dataset, args.local_data_path)
    dataset = format_dataset(args.dataset, args.dataset_format, dataset)

    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        elif 'test' in dataset:
            eval_dataset = dataset['test']
        else:
            print_rank_0('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
            
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    if args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
        for index in random.sample(range(len(train_dataset)), 3):
            print_rank_0(f"Sample {index} of the training set:\n{train_dataset[index]}.")

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
        hard_padding=args.hard_padding,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )
