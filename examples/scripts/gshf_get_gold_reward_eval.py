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

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    json_path: Optional[str] = field(
        default="5000_dpo.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    baseline_path: Optional[str] = field(
        default="",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    train_micro_batch_size_per_gpu: Optional[int] = field(
        default=4,
        metadata={"help": "the batch size for inference"},
    )
    max_length: Optional[int] = field(
        default=9999999999,
        metadata={"help": "the maximum length of the prompt"},
    )
    gold_reward_name_or_path: Optional[str] = field(
        default="openbmb/UltraRM-13b",
        metadata={"help": "the name of the gold reward model"},
    )
    input_output_delimiter: Optional[str] = field(
        default=" ",
        metadata={"help": "the delimiter between input and output"},
    )








accelerator = Accelerator()

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

print(script_args)
print("delimiter:'"+script_args.input_output_delimiter+"'")

AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = script_args.train_micro_batch_size_per_gpu






class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward( # args are the same as LlamaForCausalLM
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        transformer_outputs = self.model(
                                input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                inputs_embeds=inputs_embeds,                               
                            )

        hidden_states = transformer_outputs[0]
        rewards = self.regression_head(hidden_states).squeeze(-1)
        
        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1,1)
        rewards = torch.gather(rewards, 1, ends)
        
        return rewards
    


device = accelerator.device

tokenizer = LlamaTokenizer.from_pretrained(script_args.gold_reward_name_or_path)
rm_tokenizer = AutoTokenizer.from_pretrained(script_args.gold_reward_name_or_path)

world_size = int(os.getenv("WORLD_SIZE", "1"))
from collections import defaultdict

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
        assert len(data_comp[sample['input']]) > 0
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
    return 1.0 * win / total, 1.0 * lose / total





####

model = LlamaRewardModel.from_pretrained(script_args.gold_reward_name_or_path).to(torch.bfloat16)
model = model.to(device)


model = accelerator.prepare(model)

if not script_args.baseline_path:
    baseline_path = "eval_plus_gold_reward.json"
else:
    baseline_path = script_args.baseline_path
sft = load_dataset("json", data_files=baseline_path, split="train", field="instances")
#base_dir = "/home/xiongwei/gshf_gold_test/LMFlow_RAFT_Dev/output_models/online_dpo/iter3/1e6/checkpoint-pymodel"

ds_dir = script_args.json_path
#base_dir + str(file_dir) + "/eval_set/eval.json"


output_dir = script_args.output_dir#"/home/xiongwei/gshf_gen_data/LMFlow_RAFT_Dev/data/my_filtered_set.json"

ds = load_dataset("json", data_files=ds_dir, split="train", field="instances")

local_rank = Accelerator().local_process_index

data_size = len(ds['input'])
share = int(data_size / world_size) 
ds = ds.select(np.arange(local_rank * share, (local_rank + 1)*share))



#N = len(responses_neg)

def get_reward(texts):
    scores = []
    for txt in texts:
        inputs = tokenizer(txt, return_tensors="pt").to(device)
        if len(inputs['input_ids']) > 800:
            chosen_reward = -999999
        else:
            with torch.no_grad():
                chosen_reward = model(**inputs).item()
        scores.append(chosen_reward)
        #del inputs
    return scores





scores = []
data = []

def change_of_format(txt):
    tmp = [y for y in txt.split("###") if y]
    #txt = txt.replace
    new_txt = "\n".join(tmp)
    return new_txt

cnt = 0

for sample in ds:
    len_output = len(sample['output'])
    test_texts = [change_of_format(sample['input'] +script_args.input_output_delimiter+ sample['output'][i]) for i in range(len_output)]
    rewards = get_reward(test_texts)
    data.append({"input": sample['input'], "output": sample['output'], "rewards": rewards})
    cnt += 1
    if (cnt + 1) % 1000 == 0:
        print(cnt+1,np.mean(scores))
    if rewards[0] > -1000:
        scores.append(rewards[0])



#### Send the data to other GPUs
world_size = int(os.getenv("WORLD_SIZE", "1"))
all_process_list =[{}] * world_size

data_to_send = {
    'data': [[data[i]] for i in range(len(data))],
    'scores': scores
}

import torch.distributed as dist

dist.all_gather_object(all_process_list, data_to_send)
gathered_data = []
gathered_scores = []

for i in range(world_size):
    tmp_data = [tmp[0] for tmp in all_process_list[i]['data']]
    gathered_data.extend(tmp_data)   
    gathered_scores += all_process_list[i]['scores']
    
output_eval_dataset = {}
output_eval_dataset['type'] = 'text_only'
output_eval_dataset['instances'] = gathered_data
import json

if local_rank == 0:
    #print("Mean reward: ", np.mean(gathered_scores))
    with open(script_args.output_dir, 'w', encoding='utf8') as f:
        json.dump(output_eval_dataset, f, ensure_ascii=False)        
    sft2 = load_dataset("json", data_files=script_args.output_dir, split="train", field="instances")

    win_rate, lose_rate = get_win_rate_with_sft(sft, sft2)
    print(win_rate, lose_rate, np.mean(gathered_scores))
