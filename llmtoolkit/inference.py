import torch

from .utils import (
    require_lib,
)


def single_inference(model, tokenizer, input: str, task_type: str = "CausalLM", source_max_len: str = 512, target_max_len: str = 512):
    if task_type == 'CausalLM':
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


def vllm_inference(prompts: list, model_name_or_path: str, peft_name_or_path: str = None, max_lora_rank: int = 128, source_max_len: int = 512, target_max_len: int = 512) -> list:
    require_lib("vllm")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    max_tokens = source_max_len + target_max_len
    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=max_tokens)
    results = []

    if peft_name_or_path:
        llm = LLM(model=model_name_or_path, enable_lora=True,
                  max_lora_rank=max_lora_rank)
        outputs = llm.generate(prompts, sampling_params, lora_request=LoRARequest(
            peft_name_or_path, 1, peft_name_or_path))
    else:
        llm = LLM(model=model_name_or_path)
        outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        results.append(
            {"prompt": output.prompt, "response": output.outputs[0].text})

    return results
