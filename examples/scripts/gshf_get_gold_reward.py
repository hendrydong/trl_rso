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
        default="1000.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="gen.json",
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







accelerator = Accelerator()

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = script_args.train_micro_batch_size_per_gpu



ds_dir = script_args.json_path#"/home/xiongwei/over_opt/LMFlow_RAFT_Dev/output_models/forgetting_proj/over_opt_raft3b_get_samples_by_if_model_max512/model0/infer_set/my_infer_set.json"
output_dir = script_args.output_dir#"/home/xiongwei/gshf_gen_data/LMFlow_RAFT_Dev/data/my_filtered_set.json"



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
model = LlamaRewardModel.from_pretrained(script_args.gold_reward_name_or_path).to(torch.bfloat16)
model = model.to(device)

rm_tokenizer = AutoTokenizer.from_pretrained(script_args.gold_reward_name_or_path)

world_size = int(os.getenv("WORLD_SIZE", "1"))
####

ds = load_dataset("json", data_files=ds_dir, split="train", field="instances")

local_rank = Accelerator().local_process_index

data_size = len(ds['input'])
share = int(data_size / world_size) 
ds = ds.select(np.arange(local_rank * share, (local_rank + 1)*share))
responses_pos = [sample['input'] + sample['output'][0] for sample in ds]
responses_neg =  [sample['input'] + sample['output'][1] for sample in ds]


print(len(responses_pos), len(responses_neg))
N = len(responses_neg)


model, optimizer = accelerator.prepare(
    model
)



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
    test_texts = [change_of_format(sample['input'] + sample['output'][0]), change_of_format(sample['input'] + sample['output'][1])]
    rewards = get_reward(test_texts)
    data.append({"input": sample['input'], "output": sample['output'], "rewards": rewards})
    cnt += 1
    if (cnt + 1) % 100 == 0:
        print(cnt)
    if rewards[0] > -1000:
        scores.append(rewards[0])



#### Send the data to other GPUs
world_size = int(os.getenv("WORLD_SIZE", "1"))
all_process_list =[{}] * world_size

data_to_send = {
    'data': [[data[i]] for i in range(len(data))]
}

import torch.distributed as dist

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

