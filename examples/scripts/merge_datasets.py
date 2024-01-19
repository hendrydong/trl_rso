import json
from typing import Optional
from dataclasses import dataclass, field

from transformers import (
    HfArgumentParser,
)
from tqdm import tqdm
from datasets import load_dataset

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    path1: Optional[str] = field(
        default="/home/share/data/hanze/gen_data/online_self_iter1_s2.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    path2: Optional[str] = field(
        default="/home/share/data/hanze/gen_data/online_self_iter2_s2.json",
        metadata={"help": "the location of the output file"},
    )
    output_dir: Optional[str] = field(
        default="/home/share/data/hanze/gen_data/online_self_iter1+2_s2.json",
        metadata={"help": "the location of the output file"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


ds1 = load_dataset("json", data_files=script_args.path1, split="train", field="instances")
ds2 = load_dataset("json", data_files=script_args.path2, split="train", field="instances")


gathered_data = list(ds1) + list(ds2)

output_eval_dataset = {}
output_eval_dataset['type'] = 'text_only'
output_eval_dataset['instances'] = gathered_data
print("I collect ", len(gathered_data), "samples")

import json


with open(script_args.output_dir, 'w', encoding='utf8') as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)
