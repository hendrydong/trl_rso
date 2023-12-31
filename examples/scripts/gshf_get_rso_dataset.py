# Standard library imports
import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import random
# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    HfArgumentParser,
    LlamaConfig,
    LlamaModel,
    LlamaTokenizer,
    pipeline,
    PreTrainedModel,
    set_seed
)
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.state import AcceleratorState

# Specific imports from a package
from trl.trainer.utils import generate
from trl.trainer.utils import conduct_rejection_sampling, compute_reward_score



# Additional setup
tqdm.pandas()
set_seed(42)

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
        default="rso_gen.json",
        metadata={"help": "the location of the output file"},
    )
    num_samples_per_prompt: Optional[int] = field(
        default=2,
        metadata={"help": "the number of samples to generate per prompt"},
    )
    ranking_type: Optional[str] = field(
        default="first_round",
        metadata={"help": "the ranking method"},
    )
    beta: Optional[float] = field(
        default=0.5,
        metadata={"help": "the beta value for rejection sampling"},
    )
    input_output_delimiter: Optional[str] = field(
        default=" ",
        metadata={"help": "the delimiter between input and output"},
    )


def first_round_ranking(responses: List[str], rewards: List[float]) -> Tuple[List[str], List[str]]:
    """Conducts first round ranking. Starts from n responses and construct n/2 pairs to be assigned
    to chosen or rejected based on there rewards.
    
    Args:
        responses: accecpted candidates from rejection sampling
        rewards: response rewards.
        
    Returns:
        chosen: chosen samples.
        rejected: rejected samples.
    """
    
    chosen = []
    rejected = []
    
    def pick(responses):
        selected = random.randrange(len(responses))
        return responses.pop(selected)
    
    responses = [(response, reward) for response, reward in zip(responses,rewards)]
    while responses:
        selected1 = pick(responses)
        selected2 = pick(responses)
        if selected1[1]>selected2[1]:
            chosen.append(selected1[0])
            rejected.append(selected2[0])
        else:
            chosen.append(selected2[0])
            rejected.append(selected1[0])
            
    return chosen, rejected


def tournament_ranking(responses: List[str], rewards: List[float]):
    """Conducts tournament ranking. Starts from n responses and construct n-1 pairs to be assigned
    to chosen or rejected based on there rewards.
    
    Args:
        responses: accecpted candidates from rejection sampling.
        rewards: response rewards.
        
    Returns:
        chosen: chosen samples.
        rejected: rejected samples.
    """
    sorted_responses = [response for _, response in sorted(zip(rewards, responses), reverse=True)]
    
    chosen = [sorted_responses[i] for i in range(0, len(responses), 2)]
    rejected =[sorted_responses[i] for i in range(1, len(responses), 2)]
    
    return chosen, rejected

def random_ranking(responses: List[str], rewards: List[float]):
    """Conducts random ranking. 
    
    Args:
        responses: accecpted candidates from rejection sampling.
        rewards: response rewards.
        
    Returns:
        chosen: chosen samples.
        rejected: rejected samples.
    """
    chosen = []
    rejected = []
    
    def pick(responses):
        selected = random.randrange(len(responses))
        return responses.pop(selected)
    
    responses = [(response, reward) for response, reward in zip(responses,rewards)]
    while responses:
        selected1 = pick(responses)
        selected2 = pick(responses)
        
        chosen.append(selected1[0])
        rejected.append(selected2[0])
            
    return chosen, rejected


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

#

ds_dir = script_args.json_path# "/home/xiongwei/gshf_gen_data/LMFlow_RAFT_Dev/data/my_filtered_set.json"
output_dir = script_args.output_dir#"/home/xiongwei/gshf_gen_data/LMFlow_RAFT_Dev/data/comp.json"

ds = load_dataset("json", data_files=ds_dir, split="train", field="instances")


######## It is recommeded to clean the dataset
def check(x):
    # delete short response
    if len(x) < 5:
        return False
    # delete imcomplete response
    if '.' not in x[-7:] and '?' not in x[-7:] and '!' not in x[-7:]:
        return False
    return True

print(len(ds))
ds = ds.filter(lambda x: "#" not in x['output'][0] and "#" not in x['output'][1] )
print("After filtering samples with #", len(ds))
ds = ds.filter(lambda x: "Human:" not in x['output'][0] and "Human:" not in x['output'][1] )
print("After filtering samples with Human",len(ds))
ds = ds.filter(lambda x: "Assistant:" not in x['output'][0] and "Assistant:" not in x['output'][1])
print("After filtering samples with Assistant#", len(ds))
ds = ds.filter(lambda x:  check(x['output'][0]) or check(x['output'][1]))
print("After filtering incomplete and short responses", len(ds))
#######

data = []
cnt = 0
for sample in ds:
    if np.min(sample['rewards']) < -999 or len(set(sample['rewards']))<script_args.num_samples_per_prompt:
        cnt += 1
        continue
    if len(sample["output"])>=script_args.num_samples_per_prompt:
        #print(len(sample["rewards"]),sample["rewards"])
        accepted, rewards = conduct_rejection_sampling(sample["output"],
                                    sample["rewards"], 
                                    script_args.num_samples_per_prompt, 
                                    script_args.beta)
        if script_args.ranking_type == "first_round":
            ranking_fn = first_round_ranking
        elif script_args.ranking_type == "tournament":
            ranking_fn = tournament_ranking
        elif script_args.ranking_type == "random":
            ranking_fn = random_ranking
        chosen, rejected = ranking_fn(accepted, rewards)

        assert len(chosen) == len(rejected)

        for i in range(len(chosen)):
            data.append({"positive": sample['input'] + script_args.input_output_delimiter + chosen[i],
                        "negative": sample['input'] + script_args.input_output_delimiter + rejected[i]})

    #if sample['rewards'][0] > sample['rewards'][1]:
    #    data.append({"positive": sample['input'] + sample['output'][0], "negative": sample['input'] + sample['output'][1]})
    #else:
    #    data.append({"positive": sample['input'] + sample['output'][1], "negative": sample['input'] + sample['output'][0]})

print("Some samples are too long so deleted", cnt)
print("We collect ", len(data), " comparison pairs")

output_eval_dataset = {}
output_eval_dataset['type'] = 'text_only'
output_eval_dataset['instances'] = data
import json

with open(output_dir, 'w', encoding='utf8') as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)

