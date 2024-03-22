import json
import random

def random_str(length: int):
    strings="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890"
    result = random.choices(strings, k=length)
    new_str = ''.join(result)
    return new_str

with open('alpaca_data_en_52k.json', 'r') as file:
    data = json.load(file)

instructions = []
inputs = []
outputs = []

for block in data:
    instruction = block['instruction']
    input = block['input']
    output = block['output']

    instructions.append(instruction)
    inputs.append(input)
    outputs.append(output)

ins_length_avg = 0
inp_length_avg = 0
out_length_avg = 0

for index in instructions:
    ins_length_avg = ins_length_avg + len(index)
ins_length_avg = int(ins_length_avg/len(instructions))

for index in inputs:
    inp_length_avg = inp_length_avg + len(index)
inp_length_avg = int(inp_length_avg/len(inputs))

for index in outputs:
    out_length_avg = out_length_avg + len(index)
out_length_avg = int(out_length_avg/len(outputs))

new_data = []

tmp_block = {}
tmp_block["instruction"] = random_str(ins_length_avg)
tmp_block["input"] = random_str(inp_length_avg)
tmp_block["output"] = random_str(out_length_avg)

for i in range(len(instructions)):
    new_data.append(tmp_block)

with open('alpaca_data_en_52k_dummy.json', 'w') as file:
    json.dump(new_data, file, indent=4)