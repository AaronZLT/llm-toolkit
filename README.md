# llm-toolkit

## Before start

### 0. git clone this repo (llm-toolkit)

```bash
git clone https://github.com/AaronZLT/llm-toolkit.git
```

### 1. install llmtoolkit

```bash
# (option) conda & pypi mirror site
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# install llmtoolkit
cd llm-toolkit
pip install .

# install flash-attn
pip install flash-attn --no-build-isolation
```

## Examples

### 2.1 from llm-toolkit/examples
```bash
cd llm-toolkit/examples/Benchmark
```
Modify LLM_BENCHMARK_PATH in the cmds.sh to the path to llm-toolkit, then:
```bash
./cmds.sh
```

### 2.2 from your code
```python
import llmtoolkit
llmtoolkit.train()
```

## Citation
```bibtex
@misc{zhang2023dissectingruntimeperformancetraining,
      title={Dissecting the Runtime Performance of the Training, Fine-tuning, and Inference of Large Language Models}, 
      author={Longteng Zhang and Xiang Liu and Zeyu Li and Xinglin Pan and Peijie Dong and Ruibo Fan and Rui Guo and Xin Wang and Qiong Luo and Shaohuai Shi and Xiaowen Chu},
      year={2023},
      eprint={2311.03687},
      archivePrefix={arXiv},
      primaryClass={cs.PF},
      url={https://arxiv.org/abs/2311.03687}, 
}

@misc{zhang2023lorafamemoryefficientlowrankadaptation,
      title={LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning}, 
      author={Longteng Zhang and Lin Zhang and Shaohuai Shi and Xiaowen Chu and Bo Li},
      year={2023},
      eprint={2308.03303},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2308.03303}, 
}
```

