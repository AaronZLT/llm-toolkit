# Copyright 2024 aaron_zlt@outlook.com. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools
import re

with open("README.md", "r") as fh:
    long_description = fh.read()

_deps = [
    "bitsandbytes",
    "datasets",
    "evaluate",
    "InquirerPy",
    "numpy",
    "nvidia-ml-py",
    "packaging",
    "pandas",
    "peft",
    "torch",
    "tqdm",
    "transformers",
    "SentencePiece",
    "protobuf",
    "scikit-learn",
    "matplotlib",
    "lm-eval",
    "wandb",
    "accelerate",
]

deps = {
    b: a
    for a, b in (re.findall(r"^(([^!=<>~ ]+)(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)
}

install_requires = [
    deps["bitsandbytes"],
    deps["datasets"],
    deps["evaluate"],
    deps["InquirerPy"],
    deps["numpy"],
    deps["nvidia-ml-py"],
    deps["packaging"],
    deps["peft"],
    deps["torch"],
    deps["tqdm"],
    deps["transformers"],
    deps["pandas"],
    deps["SentencePiece"],
    deps["protobuf"],
    deps["scikit-learn"],
    deps["matplotlib"],
    deps["lm-eval"],
    deps["wandb"],
    deps["accelerate"],
]

setuptools.setup(
    name="llmtoolkit",
    version="0.0.1",
    author="aaron",
    author_email="aaron_zlt@outlook.com",
    description="This is a package for benchmarking, fine-tuning, evaluating Large Language Models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AaronZLT/llm-toolkit",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6, <3.11",
)
