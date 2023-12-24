from datasets import load_dataset
import torch
from transformers import (
    default_data_collator,
    pipeline,
    set_seed,
    AutoTokenizer
)
import sys

import numpy as np
import matplotlib.pyplot as plt
# 获取命令行参数列表



#

ds_dir = "/home/xiongwei/gshf_gen_data/LMFlow_RAFT_Dev/data/my_filtered_set.json"
output_dir = "/home/xiongwei/gshf_gen_data/LMFlow_RAFT_Dev/data/comp.json"

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

