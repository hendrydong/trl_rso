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
from accelerate.state import AcceleratorState

from trl.trainer.utils import generate

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    model_name_or_path: Optional[str] = field(
        default="/home/share/data/hanze/open_llama_3b_v2_if_1epoch_1e5_bz128_block2048_plus_hh_1epoch_1e5_bz6_blocksize2048",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="/home/share/data/hanze/raft/raft_train.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="/home/share/data/hanze/raft/raft_train_gen.json",
        metadata={"help": "the location of the output file"},
    )
    batch_size: Optional[int] = field(
        default=64,
        metadata={"help": "the batch size for inference"},
    )
    K: Optional[int] = field( 
        default=2,
        metadata={"help": "the number of generations per prompt"},
    )
    max_length: Optional[int] = field(
        default=9999999999,
        metadata={"help": "the maximum length of the prompt"},
    )
    max_new_tokens: Optional[int] = field(
        default=10,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
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
    "max_new_tokens": script_args.max_new_tokens,
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

ds = load_dataset("json", data_files=ds_dir, split="train", field="instances")
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

data_collator = DataCollatorForSeq2Seq(tokenizer, max_length=script_args.max_length, pad_to_multiple_of=8)

dataloader = DataLoader(ds, batch_size=script_args.batch_size, shuffle=False, collate_fn=data_collator)

model, dataloader = accelerator.prepare(model, dataloader)

prompts, responses = generate(model, dataloader, tokenizer, accelerator, script_args.seed, **generation_kwargs)

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