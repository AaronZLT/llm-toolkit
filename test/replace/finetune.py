import torch

import llmtoolkit
from llmtoolkit import (
    get_args,
    get_accelerate_model,
    build_data_module,
    get_unique_key,
    train,
    TrainingArguments,
    ModelArguments,
)
import bitsandbytes as bnb

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

def quant_lora_to_4bit_training(model):
    key_list = [key for key, _ in model.named_modules() if key.endswith(('lora_A.default', 'lora_B.default'))]
    for key in key_list:
        parent, child, child_name = _get_submodules(model, key)
        in_features = child.in_features
        out_features = child.out_features

        bnb_4bit_compute_dtype = torch.bfloat16
        bnb_4bit_use_double_quant = False
        bnb_4bit_quant_type = 'nf4'
        bnb_4bit_quant_storage = torch.bfloat16

        new_module = bnb.nn.Linear4bit(
            in_features,
            out_features,
            child.bias is not None,
            bnb_4bit_compute_dtype,
            compress_statistics=bnb_4bit_use_double_quant,
            quant_type=bnb_4bit_quant_type,
            quant_storage=bnb_4bit_quant_storage,
        )
        new_value_weight = bnb.nn.Params4bit(child._parameters["weight"], requires_grad=False, quant_storage=bnb_4bit_quant_storage).to(child._parameters["weight"].device)
        new_module.weight = new_value_weight
        if child.bias is not None:
            new_value_bias = bnb.nn.Params4bit(child._parameters["bias"], requires_grad=False, quant_storage=bnb_4bit_quant_storage).to(child._parameters["bias"].device)
            new_module.bias = new_value_bias
            
        if getattr(child, "state", None) is not None:
            new_module.state = child.state
        
        new_module.requires_grad_(False)
        setattr(parent, child_name, new_module)
        new_module.to(child.weight.device)
        meta = torch.device("meta")
        for name, module in new_module.named_modules():
            if not any(p.device == meta for p in module.parameters()):
                module.to(child.weight.device)
                
    for name, param in model.named_parameters():
        if "lora_A.default.weight" in name or "lora_B.default.weight" in name:
            print(name)
            param.requires_grad = True

model_args, data_args, training_args = get_args()
model, tokenizer = get_accelerate_model(model_args, training_args)
quant_lora_to_4bit_training(model)
print(model)

data_module = build_data_module(tokenizer, "gsm8k")
train(model, tokenizer, data_module["train_dataset"], data_module["eval_dataset"], data_module["data_collator"], training_args, get_unique_key(model_args, data_args, training_args))
