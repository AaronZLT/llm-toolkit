import argparse
import os

from transformers import LlamaForCausalLM, LlamaConfig


def get_config(model_name_or_path: str):
    # for now, we only support:
    model2config = {"llama": (LlamaConfig, LlamaForCausalLM)}
    for name in model2config.keys():
        if name in model_name_or_path.lower():
            return model2config[name]
    return None


def list_directories(path="."):
    items = os.listdir(path)
    directories = [item for item in items if os.path.isdir(os.path.join(path, item))]
    return directories


def get_model_file(path="."):
    items = os.listdir(path)
    extensions = (".safetensors", ".bin")
    files_with_extensions = [
        item
        for item in items
        if os.path.isfile(os.path.join(path, item)) and item.endswith(extensions)
    ]
    return files_with_extensions


def add_args(parser):
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["all"],
        help="Which model shall be generated? When set to 'all', all sub-folders will be view as model_name_or_path to be generated. Default = all.",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        type=bool,
        default=False,
        help="Overwrite target dir when there already exists model files (.safetensors, .bin). Default = False.",
    )

    return parser


def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    models_to_save = []

    if "all" in args.model:
        models_to_save = list_directories()
    else:
        for i in args.model:
            models_to_save.append(i)

    print(f"Generating target models: {models_to_save}. Check if this is right.")
    for m in models_to_save:
        if not any(
            os.path.isdir(os.path.join(".", item)) and item == m
            for item in os.listdir(".")
        ):
            print(
                f"You choose to generate model {m}, however the config folder doesn't exist. You can download the configs from huggingface. Skipping."
            )
            continue

        config = get_config(m)
        if config is None:
            print(f"We do not support generating dummy {m} for now. Skipping.")

        model_file = get_model_file()
        if len(model_file) != 0:
            if not args.overwrite_output_dir:
                print(f"Model {m} already exists! Skipping.")
                continue
            else:
                print(
                    f"Model {m} already exists! However you turned on overwrite_output_dir, generating will continue and overwrite the {m}."
                )

        print(f"Generating dummy checkpoint of {m}")
        cfg = config[0].from_pretrained(m)
        model = config[1](cfg)
        print(f"Saving dummy checkpoint to {m}")
        model.save_pretrained(m)


if __name__ == "__main__":
    main()
