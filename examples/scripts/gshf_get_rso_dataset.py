# Standard library imports
import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional, List

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
    if sample['rewards'][0] < -999 or sample['rewards'][1] < -999:
        cnt += 1
        continue
    if sample['rewards'][0] > sample['rewards'][1]:
        data.append({"positive": sample['input'] + sample['output'][0], "negative": sample['input'] + sample['output'][1]})
    else:
        data.append({"positive": sample['input'] + sample['output'][1], "negative": sample['input'] + sample['output'][0]})

print("Some samples are too long so deleted", cnt)
print("We collect ", len(data), " comparison pairs")

output_eval_dataset = {}
output_eval_dataset['type'] = 'text_only'
output_eval_dataset['instances'] = data
import json

with open(output_dir, 'w', encoding='utf8') as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)
