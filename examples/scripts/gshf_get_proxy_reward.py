from datasets import load_dataset
import torch
from transformers import (
    default_data_collator,
    pipeline,
    set_seed,
    AutoTokenizer
)
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
# 获取命令行参数列表
from accelerate import Accelerator
from accelerate.state import AcceleratorState
import torch.distributed as dist

accelerator = Accelerator()

AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 4


#####
# This script takes a dataset as the input, where each sample is {"input": "the pormpt", "output": ["response1", "response2", "response3", ...]}
# The script will compute the reward for each input-output pair, and eventually output a new dataset, where each sample contains {"input": "the pormpt", "output": ["response1", "response2", "response3", ...], "rewards": [reward1, reward2, ...]}
# Due to memory constraint, we will set the reward of the input+output that is longer than 800 tokens as -999999, which should be discarded in later processing. It should be at most ~2% samples that are discarded.
#####

#Parameters
#
#
#

ds_dir = "/home/xiongwei/over_opt/LMFlow_RAFT_Dev/output_models/forgetting_proj/over_opt_raft3b_get_samples_by_if_model_max512/model0/infer_set/my_infer_set.json"
output_dir = "/home/xiongwei/gshf_gen_data/LMFlow_RAFT_Dev/data/my_filtered_set.json"
reward_model = "/home/xiongwei/rm_study/LMFlow/output_models/gold_rm_7b_lora_1e4_bz16_with_2epoch_sft_boundary_loss5/merged_rm"
from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaTokenizer
import torch.nn as nn
import torch
from typing import Optional, List

device = accelerator.device


rm_tokenizer = AutoTokenizer.from_pretrained(reward_model)

rm_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model,
    device=device,
    tokenizer=rm_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16}
)

pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 1
}



world_size = int(os.getenv("WORLD_SIZE", "1"))
####

ds = load_dataset("json", data_files=ds_dir, split="train", field="instances")
zz = 0
for sample in ds:
    print(sample['input'] + sample['output'][0])
    zz += 1
    if zz > 10:
        break
local_rank = Accelerator().local_process_index

data_size = len(ds['input'])
share = int(data_size / world_size) 
ds = ds.select(np.arange(local_rank * share, (local_rank + 1)*share))
responses_pos = [sample['input'] + sample['output'][0] for sample in ds]
responses_neg =  [sample['input'] + sample['output'][1] for sample in ds]
#N = 2000
print(len(responses_pos), len(responses_neg))
N = len(responses_neg)


def get_reward(test_texts):

    pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
    rewards = [output[0]["score"] for output in pipe_outputs]

    return rewards

scores = []
data = []


cnt = 0

for sample in ds:
    test_texts = [sample['input'] +  tmp_output for tmp_output in sample['output']]
    rewards = get_reward(test_texts)
    data.append({"input": sample['input'], "output": sample['output'], "rewards": rewards})
    cnt += 1
    if (cnt + 1) % 100 == 0:
        print(cnt)
    if rewards[0] > -1000:
        scores.append(rewards[0])


print("mean scores", np.mean(scores))

#### Send the data to other GPUs
world_size = int(os.getenv("WORLD_SIZE", "1"))
all_process_list =[{}] * world_size

data_to_send = {
    'data': [[data[i]] for i in range(len(data))]
}
dist.all_gather_object(all_process_list, data_to_send)
gathered_data = []


for i in range(world_size):
    tmp_data = [tmp[0] for tmp in all_process_list[i]['data']]
    gathered_data.extend(tmp_data)   
    
output_eval_dataset = {}
output_eval_dataset['type'] = 'text_only'
output_eval_dataset['instances'] = data
import json


if local_rank == 0:
    with open(output_dir, 'w', encoding='utf8') as f:
        json.dump(output_eval_dataset, f, ensure_ascii=False)
