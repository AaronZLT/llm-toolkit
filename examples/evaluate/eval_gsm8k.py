from llmtoolkit import (
    infly_evaluate,
    safe_dict2file,
)


def eval(model_name_or_path: str):
    acc = infly_evaluate("gsm8k", model_name_or_path)
    results = {}
    results["model"] = model_name_or_path
    results["task"] = "gsm8k"
    results["accuracy"] = acc
    safe_dict2file(results, "eval_result.txt")


def eval_lora(model_name_or_path: str, peft_name_or_path: str):
    acc = infly_evaluate("gsm8k", model_name_or_path, peft_name_or_path)
    results = {}
    results["model"] = model_name_or_path
    results["peft"] = peft_name_or_path
    results["task"] = "gsm8k"
    results["accuracy"] = acc
    safe_dict2file(results, "eval_result.txt")


def eval_qlora(model_name_or_path: str, peft_name_or_path: str):
    acc = infly_evaluate(
        "gsm8k", model_name_or_path, peft_name_or_path, load_in_4bit=True
    )
    results = {}
    results["model"] = model_name_or_path
    results["peft"] = peft_name_or_path
    results["bits"] = 4
    results["task"] = "gsm8k"
    results["accuracy"] = acc
    safe_dict2file(results, "eval_result.txt")
