import os
import datetime
from pathlib import Path
from typing import Dict, List

from .sweep_config import (
    print_navigation,
    get_path_SerialAction,
    get_n_gpus_Action,
    SerialAction,
    taskAction,
    modelAction,
    optimization_techniquesAction,
    allow_mixAction,
    batchsizeAction,
    sequence_lengthAction,
    peftAction,
    TaskType,
    ConfigType,
    BenchmarkConfig,
)
from .utils import (
    safe_list2file,
    safe_dict2file,
    safe_readjson,
    hardware_info,
    print_rank_0,
)
def save_cmds_config(cmds:List, config:Dict):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        
    os.makedirs(f"benchmark_{formatted_time}")
    print_rank_0(f"Saving your cmds and config to benchmark_{formatted_time}/cmds and benchmark_{formatted_time}/config ...")
    safe_list2file(cmds, os.path.join(f"benchmark_{formatted_time}","cmds.sh"))
    safe_dict2file(config,os.path.join(f"benchmark_{formatted_time}","config.json"))
    
def AutoConfig():
    config = {}
    info = hardware_info()
    if info.n_gpus <= 0:
        return NotImplementedError("0 GPUs have been detected. We only support GPU LLM benchmark for now.")
    
    config[ConfigType.NGPUS] = info.n_gpus
    config[ConfigType.GPU_NAME] = info.gpu_info[0]["name"]
    config[ConfigType.GPU_MEMORY] = info.gpu_info[0]["total_memory"]
    
    print_navigation()

    path_serialAction = get_path_SerialAction(str(Path(os.getcwd()).resolve().parent.parent))
    paths = path_serialAction.execute()
    config.update(paths)
    
    gpus_config = get_n_gpus_Action(n_gpus=info.n_gpus)
    pretrain = SerialAction([gpus_config,modelAction,optimization_techniquesAction,allow_mixAction,batchsizeAction,sequence_lengthAction])
    finetune = SerialAction([gpus_config,modelAction,peftAction,optimization_techniquesAction,allow_mixAction,batchsizeAction,sequence_lengthAction])
        
    benchmark_task = taskAction.execute()
    if len(benchmark_task) == 0:
        print("No task to benchmark, exiting.")
        exit()
    for t in benchmark_task:
        if t==TaskType.PRETRAIN:
            print("\nPre-train Configure")
            config[TaskType.PRETRAIN]=pretrain.execute()
        elif t==TaskType.FINETUNE:
            print("\nFine-tune Configure")
            config[TaskType.FINETUNE]=finetune.execute()
        else:
            return NotImplementedError("AutoConfig and Sweep only support pre-train and fine-tune currently.")
        
    benchmark = BenchmarkConfig(config)
    cmds = benchmark.sweep()
    
    save_cmds_config(cmds,config)
    
def load_config_from_disk(config_file_path:str):
    config = safe_readjson(config_file_path)
    benchmark = BenchmarkConfig(config)
    cmds = benchmark.sweep()
    
    save_cmds_config(cmds,config)
