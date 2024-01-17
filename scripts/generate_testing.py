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

import numpy as np
import pandas as pd
tqdm.pandas()
from accelerate import Accelerator

from trl.trainer.utils import generate

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    model_name_or_path: Optional[str] = field(
        default="./openllama3b_dpo_v2/checkpoint-2000/",
        #default="/home/xiongwei/gshf_gold_test/LMFlow_RAFT_Dev/output_models/online_rso/iter1/beta_1.0_8_pick_2/checkpoint-pymodel1784",
        # default="/home/xiongwei/gshf_gold_test/LMFlow_RAFT_Dev/output_models/gshf_online/iter2/inner_1/checkpoint-pymodel1950",
        # default="/home/xiongwei/rm_study/LMFlow/output_models/exp_no_sharegpt/open_llama_3b_v2_instruction_following_1epoch_on_relabel_split_2w_for_sft",
        #default="/home/xiongwei/gshf_gold_test/LMFlow_RAFT_Dev/output_models/online_dpo/iter2/1e6/checkpoint-pymodel1587",
        #default="/home/xiongwei/gshf_gold_test/LMFlow_RAFT_Dev/output_models/online_dpo/new_iter2/checkpoint-pymodel2120",
        #default="",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="/home/share/data/hanze/no_share_gpt_hh/offline/rlhf/rlhf_eval/helpful_5739.json",
        #default="/home/xiongwei/rm_study/LMFlow/data/helpful/rlhf/rlhf_eval/helpful_5739.json",
        #default="/home/xiongwei/rm_study/LMFlow/data/helpful/rlhf/rlhf_prompt/helpful_100000.json",
        #default="/home/xiongwei/gshf_gold_test/LMFlow_RAFT_Dev/output_models/gshf_online/new_iter1/1w5_not_used_prompts.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        #default="/home/xiongwei/gshf_gold_test/LMFlow_RAFT_Dev/output_models/online_dpo/new_iter2/dpo_iter2_gen.json",
        #default="/home/xiongwei/over_opt/LMFlow_RAFT_Dev/output_models/12_30/raft/baseline/iter5/gen.json",
        #default="/home/xiongwei/rm_study/LMFlow/output_models/exp_no_sharegpt/open_llama_3b_v2_instruction_following_1epoch_on_hh_bz12/eval/my_eval.json",
        default="dpo_eval_reward.json",
        metadata={"help": "the location of the output file"},
    )
    batch_size: Optional[int] = field(
        default=24,
        metadata={"help": "the batch size for inference"},
    )
    K: Optional[int] = field( 
        default=1,
        metadata={"help": "the number of generations per prompt"},
    )
    max_length: Optional[int] = field(
        default=400,
        metadata={"help": "the maximum length of the prompt"},
    )
    max_new_tokens: Optional[int] = field(
        default=400,
        metadata={"help": "the maximum length of the new tokens"},
    )





accelerator = Accelerator()

#AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 4
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


ds_dir = script_args.dataset_name_or_path#"/home/xiongwei/rm_study/LMFlow/data/helpful/rlhf/rlhf_eval/helpful_5739.json"
model_name = script_args.model_name_or_path#"/home/xiongwei/LMFlow/DATA/sft_open_llama_3b_1epoch_plus_hh_rlhf_1epoch"
output_dir = script_args.output_dir#"/home/xiongwei/gshf_gen_data/LMFlow_RAFT_Dev/data/my_gen.json"
K = script_args.K

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "temperature": 1.0,
    "num_return_sequences": script_args.K,
    "max_new_tokens": script_args.max_new_tokens
}


def _clean_text(text):
    
    if len(text) == 0:
        return text
    stext = [x for x in text.split("###Human") if x]

    return stext[0].strip().replace("#", "").replace("<s>", "").replace("</s>", "")


def tokenize_fn(samples):
    model_inputs = tokenizer(samples["text"])
    return {
        **model_inputs,
    }

ds = load_dataset("json", data_files=ds_dir, split="train", field="instances")#.select(range(1000))
print(ds)

all_prompts = [sample['text'] for sample in ds]
device = accelerator.device

model = AutoModelForCausalLM.from_pretrained(
    model_name,torch_dtype=torch.bfloat16,trust_remote_code=True
)

model = model.to(device)
model.gradient_checkpointing_disable()
model.config.use_cache = True
local_rank = Accelerator().local_process_index

ds = ds.map(tokenize_fn, batched=True,remove_columns=list(ds.features))
ds.filter(lambda x: len(x["input_ids"])<script_args.max_length )

data_collator = DataCollatorForSeq2Seq(tokenizer, max_length=script_args.max_length, pad_to_multiple_of=1)

dataloader = DataLoader(ds, batch_size=script_args.batch_size, shuffle=False, collate_fn=data_collator)

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)


model, dataloader, opt = accelerator.prepare(model, dataloader, optimizer)

prompts, responses = generate(model, dataloader, tokenizer, accelerator, seed = 0, **generation_kwargs)

####
# We repeat each prompt for K times


generated_dataset = Dataset.from_dict({"prompt": prompts, "response": responses})

assert len(prompts) == len(responses)

gathered_data = []

prompts_set = {}

for i in range(len(prompts)):
    if prompts[i] not in prompts_set:
        tmp_data = {"input": prompts[i], "output": [_clean_text(responses[i])]}
        gathered_data.append(tmp_data)   
        prompts_set[prompts[i]] = len(gathered_data)
    else:
        gathered_data[prompts_set[prompts[i]]-1]["output"].append(_clean_text(responses[i]))
output_eval_dataset = {}
output_eval_dataset['type'] = 'text_only'
output_eval_dataset['instances'] = gathered_data
print("I collect ", len(gathered_data), "samples")

import json

if local_rank == 0:
    with open(output_dir, 'w', encoding='utf8') as f:
        json.dump(output_eval_dataset, f, ensure_ascii=False)
####

'''
data = [
    {"text": "###HUman:adfa. ###Assistant:124124"},
    {"text": "bbbb"},]
    
output_eval_dataset = {}
output_eval_dataset['type'] = 'text_only'
output_eval_dataset['instances'] = data
print("I collect ", len(gathered_data), "samples")

import json

if local_rank == 0:
    with open(output_dir, 'w', encoding='utf8') as f:
        json.dump(output_eval_dataset, f, ensure_ascii=False)
'''