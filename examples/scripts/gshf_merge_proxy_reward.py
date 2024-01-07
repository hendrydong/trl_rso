from datasets import load_dataset
import torch
from transformers import (
    default_data_collator,
    pipeline,
    set_seed,
    AutoTokenizer,
    HfArgumentParser
)
from typing import Optional, List
import sys
import os
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from accelerate import Accelerator
from accelerate.state import AcceleratorState
import torch.distributed as dist
from tqdm import tqdm

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    json_path: Optional[str] = field(
        default="1000.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="gen.json",
        metadata={"help": "the location of the output file"},
    )
    #max_length: Optional[int] = field(
    #    default=2048,
    #    metadata={"help": "the maximum length of the prompt"},
    #)
    merge_type: Optional[str] = field(
        default="min",
        metadata={"help": "the delimiter between input and output"},
    )







parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]




accelerator = Accelerator()



#####
# This script takes a dataset as the input, where each sample is {"input": "the pormpt", "output": ["response1", "response2", "response3", ...]}
# The script will compute the reward for each input-output pair, and eventually output a new dataset, where each sample contains {"input": "the pormpt", "output": ["response1", "response2", "response3", ...], "rewards": [reward1, reward2, ...]}
# Due to memory constraint, we will set the reward of the input+output that is longer than 800 tokens as -999999, which should be discarded in later processing. It should be at most ~2% samples that are discarded.
#####

#Parameters
#
#
#

ds_dir = script_args.json_path#"/home/xiongwei/over_opt/LMFlow_RAFT_Dev/output_models/forgetting_proj/over_opt_raft3b_get_samples_by_if_model_max512/model0/infer_set/my_infer_set.json"
output_dir = script_args.output_dir#"/home/xiongwei/gshf_gen_data/LMFlow_RAFT_Dev/data/my_filtered_set.json"
reward_model = script_args.proxy_reward_name_or_path#"/home/xiongwei/rm_study/LMFlow/output_models/gold_rm_7b_lora_1e4_bz16_with_2epoch_sft_boundary_loss5/merged_rm"
from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaTokenizer
import torch.nn as nn
import torch


device = accelerator.device

#ds = load_dataset("json", data_files=config.dataset_path, split="train", field="instances")

world_size = int(os.getenv("WORLD_SIZE", "1"))
####

ds = load_dataset("json", data_files=ds_dir, split="train", field="instances")

data_size0 = len(ds['input'])

#ds = ds.map(tokenize, batched=False)
#ds = ds.filter(lambda x: len(x["input"]) + len(x["output"]) <= script_args.max_length)


local_rank = Accelerator().local_process_index

data_size = len(ds['input'])
print("data_size:", data_size, "data_size0:", data_size0, "local_rank:", local_rank, "world_size:", world_size)

share = int(data_size / world_size) 
ds = ds.select(np.arange(local_rank * share, (local_rank + 1)*share))
responses_pos = [sample['input'] + sample['output'][0] for sample in ds]
#responses_neg =  [sample['input'] + sample['output'][1] for sample in ds]
#N = 2000
#print(len(responses_pos), len(responses_neg))
N = len(responses_pos)




scores = []
data = []


cnt = 0

# tqdm is used to show the progress bar

if script_args.merge_type == 'min':
    merge_func = lambda x: min(x)
elif script_args.merge_type == 'max':
    merge_func = lambda x: max(x)
elif script_args.merge_type == 'mean':
    merge_func = lambda x: sum(x) / len(x)
elif script_args.merge_type == 'median':
    merge_func = lambda x: np.median(x)
elif script_args.merge_type == 'sum':
    merge_func = lambda x: sum(x)
else:
    raise NotImplementedError


has_rewards = 'rewards' in ds.features
assert has_rewards
with torch.no_grad():
    for sample in tqdm(ds):
        test_texts = [sample['input'] + script_args.input_output_delimiter + tmp_output for tmp_output in sample['output']]
        rewards = sample['rewards']
        new_rewards = []
        idx = 0
        for r in rewards:
            new_r = merge_func(r)
            new_rewards.append(new_r)
            idx+=1
        data.append({"input": sample['input'], "output": sample['output'], "rewards": new_rewards})


        


print("mean scores", np.mean(scores))

#### Send the data to other GPUs
world_size = int(os.getenv("WORLD_SIZE", "1"))
all_process_list =[{}] * world_size

data_to_send = {
    'data': [[data[i]] for i in range(len(data))]
}
torch.distributed.barrier()
dist.all_gather_object(all_process_list, data_to_send)
gathered_data = []


for i in range(world_size):
    tmp_data = [tmp[0] for tmp in all_process_list[i]['data']]
    gathered_data.extend(tmp_data)   
    
output_eval_dataset = {}
output_eval_dataset['type'] = 'text_only'
output_eval_dataset['instances'] = gathered_data
import json


if local_rank == 0:
    with open(output_dir, 'w', encoding='utf8') as f:
        json.dump(output_eval_dataset, f, ensure_ascii=False)
