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
DEFAULT_PAD_TOKEN = "[PAD]"

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

def preprocess_alpaca(example):
    if example.get("input", "") != "":
        prompt_format = SFTPrompt.alpaca_input
    else:
        prompt_format = SFTPrompt.alpaca_noinput
    return {'input': prompt_format.format(**example)}

def preprocess_gsm8k(example):
    prompt_format_input = SFTPrompt.default_input
    prompt_format_output = SFTPrompt.default_output
    return {'input': prompt_format_input.format(**example), 'output': prompt_format_output.format(**example)}

def preprocess_truthfulqa_mc1(example):
    
    prompt_format_input = SFTPrompt.default_input
    prompt_format_output = SFTPrompt.default_output
    return {'input': prompt_format_input.format(**example), 'output': prompt_format_output.format(**example)}

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

def preprocess_wikitext2(example):
    return {'input': example["text"],'output':""}

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
    def load_data(dataset_name, data_path=None):
        """
        3 ways to load dataset:
        (recommend) 1. load online (data_path is not given).
        (recommend) 2. load offline from json. *Note that not all datasets are avaliable in local json.
        3. directly load dataset when 'dataset_name' is exists.
        """
        if data_path == None:
            if dataset_name == 'alpaca':
                return load_dataset("tatsu-lab/alpaca")
            elif dataset_name == 'alpaca-dummy':
                return load_dataset("Lohse/alpaca-dummy")
            elif dataset_name == 'alpaca-clean':
                return load_dataset("yahma/alpaca-cleaned")
            elif dataset_name == 'flanv2':
                return load_dataset("conceptofmind/FLAN_2022")
            elif dataset_name == 'chip2':
                return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
            elif dataset_name == 'self-instruct':
                return load_dataset("yizhongw/self_instruct", name='self_instruct')
            elif dataset_name == 'hh-rlhf':
                return load_dataset("Anthropic/hh-rlhf")
            elif dataset_name == 'longform':
                return load_dataset("akoksal/LongForm")
            elif dataset_name == 'oasst1':
                return load_dataset("timdettmers/openassistant-guanaco")
            elif dataset_name == 'gsm8k':
                return load_dataset("gsm8k","main")
            elif dataset_name == 'hellaswag':
                return load_dataset("Rowan/hellaswag")
            elif dataset_name == 'wikitext2':
                return load_dataset("wikitext", name="wikitext-2-raw-v1")
            else:
                print_rank_0(f"The dataset {dataset_name} is not officially supported by llmtoolkit, use at your own risk.")
        else:
            if dataset_name in ['alpaca','alpaca-dummy','alpaca-clean','flanv2','hh-rlhf','longform','oasst1']:
                return load_dataset('json', data_dir=os.path.join(data_path,dataset_name))
            elif dataset_name == 'chip2':
                return load_dataset('json', data_dir=os.path.join(data_path,dataset_name), data_files='unified_chip2.jsonl')
            elif dataset_name == 'self-instruct':
                return load_dataset('json', data_dir=os.path.join(data_path,dataset_name), name='self_instruct')
            elif dataset_name == 'super-natural':
                return load_dataset('json', data_dir=os.path.join(data_path,dataset_name))
            else:
                print_rank_0(f"The dataset {dataset_name} is not officially supported by llmtoolkit, use at your own risk.")

        if os.path.exists(dataset_name):
            try:
                args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                full_dataset = local_dataset(dataset_name)
                return full_dataset
            except:
                raise ValueError(f"Error loading dataset from '{dataset_name}'")
        else:
            raise NotImplementedError(f"Error loading dataset from '{dataset_name}', the path dosen't exist.")

    def format_dataset(dataset, dataset_format):
        if (dataset_format in ['alpaca','alpaca-clean','alpaca-dummy'] or (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean', 'alpaca-dummy'])):
            dataset = dataset.map(preprocess_alpaca)
        elif (dataset_format in ['gsm8k'] or (dataset_format is None and args.dataset in ['gsm8k'])):
            dataset = dataset.map(preprocess_gsm8k)
        elif (dataset_format in ['hellaswag'] or (dataset_format is None and args.dataset in ['hellaswag'])):
            dataset = preprocess_hellaswag(dataset)
        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            dataset = dataset.map(
                lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
                })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif dataset_format == 'flanv2' or (dataset_format is None and args.dataset == 'flanv2'):
            dataset = dataset.map(lambda x: {'input': x['inputs'],'output': x['targets'],})
        elif dataset_format =='super-natural' or (dataset_format is None and args.dataset == 'super-natural'):
            dataset = dataset.map(remove_columns=['id'])
            # dataset = extract_super_natural_instructions_data(dataset)
            # dataset = Dataset.from_dict(dataset)
        elif (dataset_format in ['wikitext2'] or (dataset_format is None and args.dataset in ['wikitext2'])):
            dataset = dataset.map(preprocess_wikitext2)
        elif dataset_format == 'input-output':
            # leave as is
            pass
        # Remove unused columns.
        # dataset = dataset.remove_columns(
        #     [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
        # )
        # print(dataset)
        return dataset

    dataset = load_data(args.dataset, args.data_path)
    dataset = format_dataset(dataset, args.dataset_format)

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
