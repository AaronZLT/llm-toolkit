import re
import os
from tqdm import tqdm

import llmtoolkit
from llmtoolkit import (
    get_args,
    flexible_load,
    build_data_module,
    single_inference,
    print_rank_0,
    safe_dict2file,
)

def extract_num(text):
    # Regex pattern to find the numbers following '####'
    pattern = r'#### (-?[0-9.,]+)'
    # Using re.findall to find all matches
    matches = re.findall(pattern, text)
    if matches:
        result = matches[-1]
    else:
        result = ""
    result = result.replace(',', '')
    try:
        return int(result)
    except ValueError:
        try:
            return float(result)
        except ValueError:
            print(f"'{result}' 不是有效的数字")
            return 0

args,args_dict = get_args()

model, tokenizer = flexible_load(args)

print(model)

data_module = build_data_module(tokenizer=tokenizer, args=args)


all = 0
correct = 0
t = tqdm(data_module["eval_dataset"])
for example in t:
    pred_text = single_inference(model, tokenizer, example['input'])
    golden_num = extract_num(example['output'])
    pred_num = extract_num(pred_text)
    correct += int(golden_num==pred_num)
    all += 1
    t.set_description(f"Accuracy: {correct/all*100:02f}%")
    
print("Acc:", correct/all)

eval_info = {}
eval_info["model"] = args.model_name_or_path
eval_info["peft"] = args.peft_name_or_path
eval_info["bits"] = args.bits
eval_info["mode"] = "merge" if args.unify_load else "mix"
eval_info["GSM8K Acc"] = correct/all

safe_dict2file(eval_info, "eval_info.txt")
