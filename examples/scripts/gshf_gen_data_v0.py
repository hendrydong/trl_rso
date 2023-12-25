from transformers import pipeline, AutoTokenizer
from transformers import GenerationConfig, AutoModelForCausalLM
import time


import os

from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import json
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

import numpy as np
import pandas as pd
tqdm.pandas()
from accelerate import Accelerator
from accelerate.state import AcceleratorState


accelerator = Accelerator()

AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 4


ds_dir = "./helpful_5739.json"
model_name = "gpt2"
output_dir = "./my_gen.json"
K = 2
infer_batch_size = 64
max_tokens = 10

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
}


def _clean_text(text):
    if len(text) == 0:
        return text
    stext = [x for x in text.split("###Human") if x]

    return stext[0].strip().replace("#", "").replace("<s>", "").replace("</s>", "")

ds = load_dataset("json", data_files=ds_dir, split="train", field="instances")
print(ds)
#print(ds)
all_prompts = [sample['text'] for sample in ds]
device = accelerator.device

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #load_in_8bit=True,
    #peft_config=lora_config,
    #device_map=device,
    #device_map={"": gpu_id}
)

model = model.to(device)
model.gradient_checkpointing_disable()
model.config.use_cache = True


world_size = int(os.getenv("WORLD_SIZE", "1"))
####
local_rank = Accelerator().local_process_index
data_size = len(ds['text'])
share = int(data_size / world_size) 
ds = ds.select(np.arange(local_rank * share, (local_rank + 1)*share))
####

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

model, optimizer = accelerator.prepare(
    model, optimizer
)



####
# We repeat each prompt for K times
querys = []
for prompt in all_prompts:
    querys.extend([prompt for _ in range(K)])
data_size = len(querys)
assert data_size == K * len(all_prompts)

input_texts = []

all_texts_record = []
all_responses_record = []
start_time = time.time()
for i, query in enumerate(querys):
    input_texts.append(query)
    if (i + 1) % infer_batch_size == 0 or (i+1 == data_size):
        print(i, time.time()-start_time)


        generation_kwargs["max_new_tokens"] = max_tokens
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        generated_texts = []
        output_ids = outputs
        input_ids = inputs['input_ids']
        print(i, time.time()-start_time)
        for j, output in enumerate(output_ids):
            prompt_length = len(input_ids[j])
            tgenerated_text = output[prompt_length:]
            tgenerated_text = tokenizer.decode(tgenerated_text, skip_special_tokens=True)
            generated_texts.append(tgenerated_text)
        print(i, time.time()-start_time)
        generated_texts = [
            _clean_text(generated_text) for generated_text in generated_texts
        ]
        input_texts = [txt.replace("<s>", "").replace("</s>", "") for txt in input_texts]
        all_texts_record.extend(input_texts)
        all_responses_record.extend(generated_texts)
        input_texts = []
        
assert len(all_responses_record) == data_size

data = []
for i in range(len(all_prompts)):
    data.append({"input": all_texts_record[i * K], "output": all_responses_record[i * K : (i+1) * K]})

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
output_eval_dataset['instances'] = gathered_data
print("I collect ", len(gathered_data), "samples")

import json

if local_rank == 0:
    with open(output_dir, 'w', encoding='utf8') as f:
        json.dump(output_eval_dataset, f, ensure_ascii=False)
####