import llmtoolkit

from llmtoolkit import simple_eval


lora3_100 = "/hpc2hdd/home/lzhang330/llm-toolkit/examples/Finetuning/gsm8k_3e-4_lora/checkpoint-100"
lorafa4_10000 = "/hpc2hdd/home/lzhang330/llm-benchmark/output/lorafa4/checkpoint-10000"

def eval_gsm8k():
    for i in range(10,410,10):
        simple_eval("/hpc2hdd/home/lzhang330/ssd_workspace/models/Llama-2-7b-hf", f"/hpc2hdd/home/lzhang330/llm-toolkit/examples/Finetuning/gsm8k/lora/gsm8k_3e-4_lora/checkpoint-{i}", ["gsm8k"], output_dir=f"result_gsm8k_3e-4_lora_ckpt_{i}")

def eval_truthfulqa():
    for i in range(100,110,10):
        simple_eval("/hpc2hdd/home/lzhang330/ssd_workspace/models/Llama-2-7b-hf", f"/hpc2hdd/home/lzhang330/llm-toolkit/examples/Finetuning/gsm8k/lora/gsm8k_3e-4_lora/checkpoint-{i}", ["truthfulqa_mc1"], all_output = True)

def eval_hellaswag():
    for i in range(20,380,20):
        simple_eval("/hpc2hdd/home/lzhang330/ssd_workspace/models/Llama-2-7b-hf", f"/hpc2hdd/home/lzhang330/llm-toolkit/examples/Finetuning/hellaswag_3e-4_lora/checkpoint-{i}", ["hellaswag"], output_dir=f"eval_hellaswag/lora/result_hellaswag_3e-4_lora_ckpt_{i}")

eval_hellaswag()