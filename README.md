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
