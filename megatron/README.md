# Megatron-LM TP + PP

*** we only support DP + TP + PP in pretraining Llama2-7b for now.

## Before start

### 0. Nvidia docker

```bash
# pull and run docker
docker pull nvcr.io/nvidia/pytorch:23.08-py3
docker run --gpus all -it --shm-size="384g" --rm -v /path/to/megatron:/workspace/megatron -v /path/to/dataset:/workspace/dataset -v /path/to/checkpoints:/workspace/checkpoints nvcr.io/nvidia/pytorch:23.08-py3

#prepare env
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install accelerate transformers sentencepiece flash-attn nltk deepspeed
```

Note that since now we are in docker.

### 1. git clone Megatron-LLaMA

```bash
cd /workspace/megatron
git clone https://github.com/AaronZLT/Megatron-LLaMA.git
cd Megatron-LLaMA
```

### 2. convert model and build dataset

```bash
# convert HF model to Megatron model
./hf_to_megatron.sh

# build the dataset using tokenizer.model
./build_dataset.sh
```

## Run pretrain

```bash
./llama2.sh
```
