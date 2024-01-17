from transformers import pipeline, AutoTokenizer
from transformers import GenerationConfig, AutoModelForCausalLM
import time
from torch.utils.data import DataLoader

import os

from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import json
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, DataCollatorForSeq2Seq
from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaTokenizer
import torch.nn as nn
import torch
from typing import Optional, List
import numpy as np
import pandas as pd
tqdm.pandas()
from accelerate import Accelerator
from accelerate.state import AcceleratorState

from trl.trainer.utils import generate


from collections import defaultdict


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    json_path: Optional[str] = field(
        default="/home/xiongwei/gshf_gold_test/LMFlow_RAFT_Dev/output_models/online_dpo/iter1",
        metadata={"help": "the location of the dataset name or path"},
    )
    base_json_path: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )



parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

def get_win_rate_with_sft(ds1, ds2):
    data_comp = defaultdict(list)
    win = 0
    total = 0
    tie = 0
    lose = 0
    for sample in ds1:
        if len(data_comp[sample['input']]) == 0:
            data_comp[sample['input']].append(sample['rewards'][0])
            total += 1
    for sample in ds2:
        #print(sample['input'])
        if len(data_comp[sample['input']]) == 0:
            continue
        if len(data_comp[sample['input']]) == 1:
            if sample['rewards'][0] > data_comp[sample['input']][0]:
                win += 1
            elif sample['rewards'][0] == data_comp[sample['input']][0]:
                tie += 1
            else:
                lose += 1
            data_comp[sample['input']].append(sample['rewards'][0])
    if tie > 0:
        print("tie : ", tie)
    print(win, tie, lose, total)
    return 1.0 * (win+tie*0.5) / total, 1.0 * lose / total

base_dir = script_args.json_path
ds_dir = script_args.base_json_path
sft = load_dataset("json", data_files=base_dir, split="train", field="instances")
sft2 = load_dataset("json", data_files=ds_dir, split="train", field="instances")
win_rate, lose_rate = get_win_rate_with_sft(sft, sft2)
print(win_rate, lose_rate)