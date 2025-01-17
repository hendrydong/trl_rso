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
    proxy_reward_name_or_path: Optional[str] = field(
        default="relabel_by_gold13b_genby3b_if_rm_open_llama_3b_v2_if_1epoch_hh_2e5_2epoch_exp1",
        metadata={"help": "the name of the gold reward model"},
    )
    proxy_reward_tokenizer_name_or_path: Optional[str] = field(
        default=""
    )
    input_output_delimiter: Optional[str] = field(
        default=" ",
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

if script_args.proxy_reward_tokenizer_name_or_path == "":
    rm_tokenizer = AutoTokenizer.from_pretrained(reward_model)
else:
    rm_tokenizer = AutoTokenizer.from_pretrained(script_args.proxy_reward_tokenizer_name_or_path)

rm_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model,
    device=device,
    tokenizer=rm_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    truncation=True
)

pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 1,
}

def tokenize(sample):
    tokenized_input = rm_tokenizer(sample['input'])
    tokenized_output = rm_tokenizer(sample['output'])
    sample["input_ids"] = tokenized_input["input_ids"]
    sample["output_ids"] = tokenized_output["input_ids"]
    return sample

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


def get_reward(test_texts):
    pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
    rewards = [output[0]["score"] for output in pipe_outputs]
    return rewards

scores = []
data = []


cnt = 0

# tqdm is used to show the progress bar

has_rewards = 'rewards' in ds.features

with torch.no_grad():
    for sample in tqdm(ds):
        test_texts = [sample['input'] + script_args.input_output_delimiter + tmp_output for tmp_output in sample['output']]
        try:
            rewards = get_reward(test_texts)
            if has_rewards:
                new_rewards = []
                idx = 0
                for r in rewards:
                    new_r = rewards[idx]
                    if type(r) == list:
                        new_rewards.append(r+[new_r])
                    elif type(r) == float:
                        new_rewards.append([r,new_r])
                    idx+=1
                data.append({"input": sample['input'], "output": sample['output'], "rewards": new_rewards})
            else:
                data.append({"input": sample['input'], "output": sample['output'], "rewards": rewards})
                cnt += 1
                if rewards[0] > -1000:
                    scores.append(rewards[0])
        except RuntimeError:
            with open(f"error_{cnt}.txt", "w") as f:
                for tmp in test_texts:
                    f.write(tmp + "\n")
        


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
