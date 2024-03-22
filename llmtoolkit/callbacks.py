import os
import time
import numpy as np
from tqdm import tqdm

import torch
import transformers
import evaluate

from .utils import (
    print_rank_0,
    safe_dict2file,
)
from .dataset import (
    IGNORE_INDEX,
)

class MMLUEvalCallback(transformers.TrainerCallback):
    def __init__(self, args, key, trainer, tokenizer, mmlu_dataset):
        '''
        trainer, tokenizer, mmlu_dataset must be initialized.
        '''
        self.key = key
        self.lr = args.learning_rate
        self.r = args.r
        self.trainer = trainer
        self.mmlu_dataset = mmlu_dataset
        self.abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
            ]
        self.accuracy = evaluate.load("accuracy" if args.metrics_path is None else os.path.join(args.metrics_path,"accuracy"))

    
    def on_evaluate(self, args, state, control, model, **kwargs):
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
        
        data_loader = self.trainer.get_eval_dataloader(self.mmlu_dataset)
        source_max_len = self.trainer.data_collator.source_max_len
        self.trainer.data_collator.source_max_len = args.mmlu_source_max_len
        self.trainer.model.eval()
        preds, refs = [], []
        loss_mmlu = 0
        for batch in tqdm(data_loader, total=len(data_loader)):
            (loss, logits, labels) = self.trainer.prediction_step(self.trainer.model,batch,prediction_loss_only=False,)
            # There are two tokens, the output, and eos token.
            for i, logit in enumerate(logits):
                label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                logit_abcd = logit[label_non_zero_id-1][self.abcd_idx]
                preds.append(torch.argmax(logit_abcd).item())
            labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
            refs += [self.abcd_idx.index(label) for label in labels.tolist()]
            loss_mmlu += loss.item()
        # Extract results by subject.
        results = {'mmlu_loss':loss_mmlu/len(data_loader)}
        subject = self.mmlu_dataset['subject']
        subjects = {s:{'refs':[], 'preds':[]} for s in set(subject)}
        for s,p,r in zip(subject, preds, refs):
            subjects[s]['preds'].append(p)
            subjects[s]['refs'].append(r)
        subject_scores = []
        for subject in subjects:
            if len(subjects[subject]['refs']) !=0 and len(subjects[subject]['preds']) !=0:
                subject_score = self.accuracy.compute(
                    references=subjects[subject]['refs'],
                    predictions=subjects[subject]['preds']
                )['accuracy']
                results[f'mmlu_{args.mmlu_split}_accuracy_{subject}'] = subject_score
                subject_scores.append(subject_score)
        results[f'mmlu_{args.mmlu_split}_accuracy'] = np.mean(subject_scores)
        
        mmlu_accuracy_dict={}
        mmlu_accuracy_dict["key"]=self.key
        mmlu_accuracy_dict["lr"]=self.lr
        mmlu_accuracy_dict["r"]=self.r
        mmlu_accuracy_dict[f'mmlu_{args.mmlu_split}_accuracy'] = results[f'mmlu_{args.mmlu_split}_accuracy']
        
        safe_dict2file(mmlu_accuracy_dict,"accuracy.txt")
        
        self.trainer.log(results)
        self.trainer.data_collator.source_max_len = source_max_len
                    
class EmptycacheCallback(transformers.TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        print_rank_0("Cache cleared [after step].")
    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        print_rank_0("Cache cleared [before train].")
    def on_init_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        print_rank_0("Cache cleared [after init].")

class PT_ProfCallback(transformers.TrainerCallback):
    def __init__(self, warmup_step, key, output_dir:str = ""):
        self.prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=warmup_step, active=10, repeat=1),
            profile_memory=True,
            with_stack=True,
            record_shapes=True
            )
        self.warmup_step = warmup_step
        self.key = key
        self.output_dir = output_dir

    def on_train_begin(self, args, state, control, **kwargs):
        self.prof.start()

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()

    def on_train_end(self, args, state, control, **kwargs):
        self.prof.stop()
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                self.prof.export_chrome_trace(os.path.join(self.output_dir,f"Trace_{self.key}_step_{self.warmup_step}_to_{self.prof.step_num}.json"))
        else:
            self.prof.export_chrome_trace(os.path.join(self.output_dir,f"Trace_{self.key}_step_{self.warmup_step}_to_{self.prof.step_num}.json"))           

class StepInfoCallback(transformers.TrainerCallback):
    def __init__(self, warmup_step, key, token_per_step, output_dir:str = ""):
        self.warmup_step = warmup_step
        self.key = key
        self.token_per_step = token_per_step
        self.step_times = []
        self.output_dir = output_dir

    def on_step_begin(self, args, state, control, **kwargs):  
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        self.end_time = time.time()
        self.step_times.append(self.end_time-self.start_time)

    def on_train_end(self, args, state, control, **kwargs):
        mean_step_time = round(np.mean(self.step_times[self.warmup_step-1:-1]),3)
        std_step_time = round(np.std(self.step_times[self.warmup_step-1:-1]),3)
        profile_dict={}
        profile_dict["key"] = self.key
        profile_dict["step_time (s)"] = mean_step_time
        profile_dict["step_time_std (s)"] = std_step_time
        # profile_dict["step_time (s)"] = f"{mean_step_time} s"
        # profile_dict["step_time_std (s)"] = f"{std_step_time} s"
        profile_dict["token/s"] = round(self.token_per_step/mean_step_time,2)
        profile_dict["mem (GB)"] = round((torch.cuda.mem_get_info(device=None)[1]-torch.cuda.mem_get_info(device=None)[0])/1024/1024/1024,2)
        safe_dict2file(profile_dict, os.path.join(self.output_dir,"profiler.txt"))