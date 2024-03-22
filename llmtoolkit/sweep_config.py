import os

from enum import Enum
from typing import Dict, List
from itertools import combinations, product
from collections import Counter

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.validator import PathValidator

from .utils import (
    print_rank_0,
    run_once,
)

os.environ["INQUIRERPY_STYLE_POINTER"] = "#82f24a"
os.environ["INQUIRERPY_STYLE_ANSWER"] = "#82f24a"
os.environ["INQUIRERPY_STYLE_CHECKBOX"] = "#82f24a"
os.environ["INQUIRERPY_STYLE_INPUT"] = "#82f24a"
os.environ["INQUIRERPY_STYLE_QUESTIONMARK"] = "#E3170D"

@run_once
def print_navigation():
    print_rank_0(
        '''
        Navigation
        ↑↓      select
        space   selecting
        enter   comfirm
        Ctrl+A  select all
        '''
    )
    
class ExplicitEnum(str, Enum):
    '''
    from huggingface transformers
    '''
    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )
        
class ConfigType(ExplicitEnum):
    GPU_NAME = "gpu_name"
    GPU_MEMORY = "gpu_mem"
    TASK = "task"
    WORK_PATH = "work_path"
    MAIN_FILE_PATH = "main_file_path"
    MODELS_PATH = "models_path"
    DATASET_PATH = "dataset_path"
    OUTPUT_PATH = "output_path"
    NGPUS = "n_gpus"
    MODEL = "model"
    OPTIMIZATION_TECHNIQUES = "optimization_techniques"
    PEFT = "peft"
    ALLOW_MIX = "allow_mix"
    BATCHSIZE = "batchsize"
    SEQUENCE_LENGTH = "sequence_length"
    
class TaskType(ExplicitEnum):
    PRETRAIN = "pre-train"
    FINETUNE = "fine-tune"
    INFERENCE = "inference"

class ModelType(ExplicitEnum):
    LLAMA_2_1_3B = "Llama-2-1.3b-hf"
    LLAMA_2_7B = "Llama-2-7b-hf"
    LLAMA_2_13B = "Llama-2-13b-hf"
    LLAMA_2_70B = "Llama-2-70b-hf"

class OptimizationTechniquesType(ExplicitEnum):
    RECOMPUTATION = "R"
    FLASHATTENTION = "F"
    ZERO2 = "Z2"
    ZERO3 = "Z3"
    ZERO2_OFFLOAD = "Z2O"
    ZERO3_OFFLOAD = "Z3O"
    QUANTIZATION = "Q"
    LORA = "L"
    QLORA = "QL"

class Action:
    def __init__(self, action, key) -> None:
        self.key = key
        self.init_action(action)
    def init_action(self, action):
        self.action = action
    def execute(self):
        if self.action:
            # self.result = self.action.execute()
            return(self.action.execute())

class CheckboxAction(Action):
    def __init__(self, action, key:str = "default_key") -> None:
        super().__init__(action, key)
        
class InputAction(Action):
    def __init__(self, action, key:str = "default_key") -> None:
        super().__init__(action, key)
        
class SelectAction(Action):
    def __init__(self, action, key:str = "default_key") -> None:
        super().__init__(action, key)

taskAction = CheckboxAction(
    action=inquirer.checkbox(
        message="Which task to benchmark?",
        choices=[
            Choice(TaskType.PRETRAIN, name="Pre-train"),
            Choice(TaskType.FINETUNE, name="Fine-tune"),
            Choice(TaskType.INFERENCE, name="Inference"),
            ],
        ),
    key=ConfigType.TASK
    )

batchsizeAction = CheckboxAction(
    action=inquirer.checkbox(
        message="Batchsize:",
        choices=[1,2,4,8,16,32,48,64,80,96,112,128],
        validate=lambda result: len(result) >= 1,
        invalid_message="should be at least 1 selection",
        instruction="(select at least 1)",
        ),
    key=ConfigType.BATCHSIZE
    )

sequence_lengthAction = CheckboxAction(
    action=inquirer.checkbox(
        message="Sequence length:",
        choices=[512,1024,2048,4096],
        validate=lambda result: len(result) >= 1,
        invalid_message="should be at least 1 selection",
        instruction="(select at least 1)",
        ),
    key=ConfigType.SEQUENCE_LENGTH
    )

modelAction = CheckboxAction(
    action=inquirer.checkbox(
        message="Model:",
        choices=[
            Choice(ModelType.LLAMA_2_1_3B, name="Llama-2-1.3b-hf"),
            Choice(ModelType.LLAMA_2_7B, name="Llama-2-7b-hf (default)", enabled=True),
            Choice(ModelType.LLAMA_2_13B, name="Llama-2-13b-hf"),
            Choice(ModelType.LLAMA_2_70B, name="Llama-2-70b-hf"),
            ],
        validate=lambda result: len(result) >= 1,
        invalid_message="should be at least 1 selection",
        instruction="(select at least 1)",
        ),
    key=ConfigType.MODEL
    )

optimization_techniquesAction = CheckboxAction(
    action=inquirer.checkbox(
        message="Optimization techniques:",
        choices=[
            Choice(OptimizationTechniquesType.RECOMPUTATION, name="Activation Checkpointing"),
            Choice(OptimizationTechniquesType.FLASHATTENTION, name="Flash Attention"),
            Choice(OptimizationTechniquesType.ZERO2, name="ZeRO-2"),
            Choice(OptimizationTechniquesType.ZERO3, name="ZeRO-3"),
            Choice(OptimizationTechniquesType.ZERO2_OFFLOAD, name="ZeRO-2 + Offload"),
            Choice(OptimizationTechniquesType.ZERO3_OFFLOAD, name="ZeRO-3 + Offload"),
            Choice(OptimizationTechniquesType.QUANTIZATION, name="Quantization"),
            ],
        validate=lambda result: len(result) >= 0,
        invalid_message="",
        instruction="(select 0 or more techniques to benchmark)",
        ),
    key=ConfigType.OPTIMIZATION_TECHNIQUES
    )

allow_mixAction = SelectAction(
    action = inquirer.select(
        message="Allow additive mix the techniques above?",
        choices=[
            Choice(True, name="mix (default)", enabled=True),
            Choice(False, name="no-mix"),
        ],
        multiselect=False,
    ),
    key=ConfigType.ALLOW_MIX
)

peftAction = CheckboxAction(
    action=inquirer.checkbox(
        message="Which PEFT method to benchmark?",
        choices=[
            Choice(OptimizationTechniquesType.LORA, name="LoRA"),
            Choice(OptimizationTechniquesType.QLORA, name="QLoRA"),
            ],
        validate=lambda result: len(result) >= 1,
        invalid_message="should be at least 1 selection",
        instruction="(select PEFT method to benchmark)",
        ),
    key=ConfigType.PEFT
    )

class SerialAction:
    def __init__(self,actions:List[Action]) -> None:
        self.actions = []
        for i in actions:
            self.actions.append(i)
            
    def execute(self):
        print_rank_0("")
        results={}
        for i in self.actions:
            results[i.key]=i.execute()
        return results

def get_gpu_candidate(n_gpus:int) -> List:
    candidates = []
    iter = 1
    while iter <= n_gpus:
        candidates.append(iter)
        iter *= 2
    if iter // 2 < n_gpus:
        candidates.append(n_gpus)
    return candidates[::-1]

def get_n_gpus_Action(n_gpus:int):
    return(
        CheckboxAction(
        action=inquirer.select(
            message="How many GPU(s) to use in benchmark?",
            choices=get_gpu_candidate(n_gpus),
            multiselect=False,
            ),
        key=ConfigType.NGPUS
        ))

key2cmd = {
    OptimizationTechniquesType.FLASHATTENTION:"--flash_attn True",
    OptimizationTechniquesType.RECOMPUTATION:"--gradient_checkpointing True",
    OptimizationTechniquesType.ZERO2:"--deepspeed zero2.json",
    OptimizationTechniquesType.ZERO3:"--deepspeed zero3.json",
    OptimizationTechniquesType.ZERO2_OFFLOAD:"--deepspeed zero2off.json",
    OptimizationTechniquesType.ZERO3_OFFLOAD:"--deepspeed zero3off.json",
    OptimizationTechniquesType.QUANTIZATION:"--quant True --double_quant True --bits 4",
    OptimizationTechniquesType.LORA:"--use_lora True --lora_r 64 --lora_modules all",
    OptimizationTechniquesType.QLORA:"--quant True --double_quant True --bits 4 --use_lora True --lora_r 64 --lora_modules all",
}

class BenchmarkConfig:
    def __init__(self, config:Dict = None) -> None:
        self.config = config

    def get_optimization_techniques_cmd(self,config:Dict,optimization_techniques:List):
        cmds = []
        combos = []
        
        if config[ConfigType.ALLOW_MIX] == False:
            cmds = [key2cmd[i] for i in optimization_techniques]
            return cmds
        
        for i in range(1, len(optimization_techniques)+1):
            for combo in combinations(optimization_techniques, i):
                if self.mix_rules(combo):
                    combos.append(combo)      
        for combo in combos:
            optimization_techniques_cmd_combo = [key2cmd[i] for i in combo]
            cmds.append(' '.join(optimization_techniques_cmd_combo))
        return cmds
    
    def mix_rules(self,combo:List) -> bool:
        counts = Counter(combo)
        
        if(counts[OptimizationTechniquesType.ZERO2]+counts[OptimizationTechniquesType.ZERO2_OFFLOAD]+counts[OptimizationTechniquesType.ZERO3]+counts[OptimizationTechniquesType.ZERO3_OFFLOAD]>1):
            return False
        if(counts[OptimizationTechniquesType.QUANTIZATION]+counts[OptimizationTechniquesType.LORA]+counts[OptimizationTechniquesType.QLORA]):
            return False
        
        return True
    
    def sweep(self,config:Dict = None):
        if config == None:
            config = self.config
        if config == None:
            return ValueError("In sweep, config must be initialized.")
        
        cmds = []
        memory = config[ConfigType.GPU_MEMORY]
        main_file_path = config[ConfigType.MAIN_FILE_PATH]
        models_path = config[ConfigType.MODELS_PATH]
        dataset_path = config[ConfigType.DATASET_PATH]
        output_path = config[ConfigType.OUTPUT_PATH]
        
        if(TaskType.PRETRAIN in config):
            pre_train_config = config[TaskType.PRETRAIN]
            
            # Without opti-tech
            for i in list(product(pre_train_config[ConfigType.MODEL],pre_train_config[ConfigType.SEQUENCE_LENGTH],pre_train_config[ConfigType.BATCHSIZE])):
                cmds.append(self.BenchmarkCMD(pre_train_config[ConfigType.NGPUS],memory,main_file_path,models_path,dataset_path,output_path,i[0],"",i[1],i[2]))
                
            for i in list(product(pre_train_config[ConfigType.MODEL],self.get_optimization_techniques_cmd(pre_train_config,pre_train_config[ConfigType.OPTIMIZATION_TECHNIQUES]),pre_train_config[ConfigType.SEQUENCE_LENGTH],pre_train_config[ConfigType.BATCHSIZE])):
                # With opti-tech
                cmds.append(self.BenchmarkCMD(pre_train_config[ConfigType.NGPUS],memory,main_file_path,models_path,dataset_path,output_path,i[0],i[1],i[2],i[3]))
                
        if(TaskType.FINETUNE in config):
            fine_tune_config = config[TaskType.FINETUNE]
            
            for i in list(product(fine_tune_config[ConfigType.MODEL],self.get_optimization_techniques_cmd(pre_train_config,fine_tune_config[ConfigType.PEFT]+fine_tune_config[ConfigType.OPTIMIZATION_TECHNIQUES]),fine_tune_config[ConfigType.SEQUENCE_LENGTH],fine_tune_config[ConfigType.BATCHSIZE])):
                cmds.append(self.BenchmarkCMD(fine_tune_config[ConfigType.NGPUS],memory,main_file_path,models_path,dataset_path,output_path,i[0],i[1],i[2],i[3]))
                
        return cmds

    def BenchmarkCMD(self, n_gpus:int, memory:int, main_file_path:str, models_path:str, dataset_path:str, output_path:str, model:str, optimization_techniques:str, sequence_length:int, batchsize:int):
        '''
        A default cmd in training should be like this:
        
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DS_SKIP_CUDA_CHECK=1 accelerate launch $WORK_PATH/run_llama.py --model_name_or_path $MODEL_PATH/$MODEL_NAME --data_path $WORK_PATH/data --dataset $DATASET --metrics_path $WORK_PATH/metrics --output_dir $WORK_PATH/output/ --num_train_epochs 15 --learning_rate 0.0002 --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --evaluation_strategy no --eval_steps 1 --eval_dataset_size 10 --max_eval_samples 10 --per_device_eval_batch_size 1 --max_new_tokens 32 --dataloader_num_workers 1 --remove_unused_columns False --do_train --do_eval --source_max_len 256 --target_max_len 256 --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler_warmup_step 2 --max_steps 4 --profiler pytorch --max_memory_MB 24000 --per_device_train_batch_size 1
        
        For benchmark, some args (dataset, learning_rate, etc.) are set to default value, as they won't influence the result.
        
        '''
        
        DS_SKIP_CUDA_CHECK = 1
        # if DS_SKIP_CUDA_CHECK == 1:
        #     print_rank_0("We are ingoring the version check of compiled torch and your cuda version. Proceed with caution.")
            
        cmd = f"CUDA_VISIBLE_DEVICES={','.join(str(i) for i in range(n_gpus))} DS_SKIP_CUDA_CHECK={DS_SKIP_CUDA_CHECK} accelerate launch {main_file_path} --data_path {dataset_path} --dataset alpaca-dummy --output_dir {output_path} --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --evaluation_strategy no --eval_steps 1 --eval_dataset_size 10 --max_eval_samples 10 --per_device_eval_batch_size 1 --max_new_tokens 32 --dataloader_num_workers 1 --remove_unused_columns False --do_train --do_eval --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler_warmup_step 5 --max_steps 10 --model_name_or_path {os.path.join(models_path,model)} --per_device_train_batch_size {batchsize} --source_max_len {int(sequence_length/2)} --target_max_len {int(sequence_length/2)} --max_memory_MB {memory} {optimization_techniques}"
        
        return cmd
    
def get_path_SerialAction(path:str):
    home_path = "~/" if os.name == "posix" else "C:\\"
    
    work_path = InputAction(
        action=inquirer.filepath(
        message="Enter the path to llm-benchmark:",
        default=path,
        validate=PathValidator(is_dir=True, message="Input is not a directory"),
        only_directories=True,),
        key=ConfigType.WORK_PATH
        )
    main_file_path = InputAction(
        action=inquirer.filepath(
        message="Enter the path to benchmark.py (or the python script you want to run):",
        default=os.path.join(path,"benchmark.py"),
        validate=PathValidator(is_file=True, message="Input is not a file"),),
        key=ConfigType.MAIN_FILE_PATH
        )
    models_path = InputAction(
        action=inquirer.filepath(
        message="Enter the path to models (leave blank to download from huggingface):",
        default=os.path.join(path,"models"),
        validate=PathValidator(is_dir=True, message="Input is not a directory"),
        only_directories=True,),
        key=ConfigType.MODELS_PATH
        )
    dataset_path = InputAction(
        action=inquirer.filepath(
        message="Enter the path to datasets (leave blank to download from huggingface):",
        default=os.path.join(path,"datasets"),
        validate=PathValidator(is_dir=True, message="Input is not a directory"),
        only_directories=True,),
        key=ConfigType.DATASET_PATH
        )
    output_path = InputAction(
        action=inquirer.filepath(
        message="Enter the output path:",
        default=os.path.join(path,"output"),),
        key=ConfigType.OUTPUT_PATH
        )
    return SerialAction([work_path,main_file_path,models_path,dataset_path,output_path])
