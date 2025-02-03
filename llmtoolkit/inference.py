import torch

from .utils import (
    require_lib,
    gsi,
    rank_0,
    print_rank_0,
)

@rank_0
def single_inference(
    model,
    tokenizer,
    input: str,
    task_type: str = "CausalLM",
    source_max_len: str = 512,
    target_max_len: str = 512,
):
    if task_type == "CausalLM":
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
                top_p=0.0,
                temperature=0.1,
            )
        pred_text = tokenizer.decode(
            outputs.sequences[0][len(inputs["input_ids"][0]) :],
            skip_special_tokens=True,
        )
    elif task_type == "ConditionalGeneration":
        inputs = tokenizer(input, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=target_max_len)
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return pred_text


def vllm_inference(
    prompts: list,
    model_name_or_path: str,
    peft_name_or_path: str = None,
    max_lora_rank: int = 128,
    source_max_len: int = 512,
    target_max_len: int = 512,
    load_in_4bit: bool = False,
) -> list:
    require_lib("vllm")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import torch

    max_tokens = source_max_len + target_max_len
    sampling_params = SamplingParams(temperature=0.0, top_p=0.1, max_tokens=max_tokens)

    if gsi.info["ngpu"] >= 2:
        print_rank_0('WARNING: 2 or more gpus are detected, and VLLM will use all gpus to inference. However, a RuntimeError may raised: "An attempt has been made to start a new process before the current process ...". To avoid this error, wrap your code within " if __name__ == "__main__": ". This is a bug in VLLM, an expected behavior when tp >= 2 & ray. For more info please refer to https://github.com/vllm-project/vllm/pull/5669.')
    vllm_kwargs = {
        "model": model_name_or_path,
        "dtype": torch.bfloat16,
        "tensor_parallel_size": gsi.info["ngpu"],
        "gpu_memory_utilization": 0.8,
    }
    if load_in_4bit:
        vllm_kwargs.update(
            {"quantization": "bitsandbytes", "load_format": "bitsandbytes"}
        )
    if peft_name_or_path:
        vllm_kwargs.update({"enable_lora": True, "max_lora_rank": max_lora_rank})

    llm = LLM(**vllm_kwargs)

    generate_kwargs = {}
    if peft_name_or_path:
        generate_kwargs["lora_request"] = LoRARequest(
            peft_name_or_path, 1, peft_name_or_path
        )

    outputs = llm.generate(prompts, sampling_params, **generate_kwargs)

    results = [
        {"prompt": output.prompt, "response": output.outputs[0].text}
        for output in outputs
    ]

    return results
