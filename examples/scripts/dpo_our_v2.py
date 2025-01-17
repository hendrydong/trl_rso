# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, AutoModelForCausalLM
from trl import DPOTrainer

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="/home/share/data/hanze/open_llama_3b_v2_if_1epoch_1e5_bz128_block2048_plus_hh_1epoch_1e5_bz6_blocksize2048",
        metadata={"help": "the location of the SFT model name or path"},
    )
    train_dir: Optional[str] = field(
        default="/home/hdongaj/Projects/LMFlow/data/helpful/rm/rm_train/rm1003.json",
        metadata={"help": "the location of the SFT model name or path"},
    )
    eval_dir: Optional[str] = field(
        default="/home/hdongaj/Projects/LMFlow/data/helpful/rm/rm_validate/m_val1003.json",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=1e-6, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="adamw_hf", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=5, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    #master_port: Optional[int] = field(
    #    default=29485, metadata={"help": "whether to use gradient checkpointing"}
    #)
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=1400, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=2800, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=100000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=2, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=940, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=940, metadata={"help": "the evaluation frequency"})
    run_name: Optional[str] = field(default="dpo_test", metadata={"help": "the run name"})
    output_dir: Optional[str] = field(default="/home/xiongwei/over_opt/LMFlow_RAFT_Dev/output_models/forgetting_proj/dpo2", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


def get_stack_exchange_paired(
    data_dir: str = "/home/xiongwei/rm_study/LMFlow/data/helpful/rm/rm_train/rm1003.json",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    ds = load_dataset("json", data_files=data_dir, split="train")['instances'][0]
    pos = [sample['positive'] for sample in ds]
    neg = [sample['negative'] for sample in ds]
    dataset = Dataset.from_dict({
        "positive": pos,
        "negative": neg
    })
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))
    
    def tokenize(sample):
        tmp = sample['positive'].split("###Assistant:")
        tmp_neg = sample['negative'].split("###Assistant:")
        prompt = "###Assistant:".join(tmp[:-1]) + "###Assistant:"
        sample["input"] = prompt
        sample["positive2"] = tmp[-1]#tokenizer.decode(sample["input_ids"])
        sample['negative2'] = tmp_neg[-1]
        return sample

    dataset = dataset.map(tokenize, batched=False, remove_columns=original_columns)

    original_columns = dataset.column_names


    def return_prompt_and_responses(samples) -> Dict[str, str]:

        return    {
            "prompt": [prompt for prompt in samples["input"]],
            "chosen": samples["positive2"],
            "rejected": samples["negative2"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        #load_in_4bit=True,
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        #load_in_4bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token


    def tokenize(sample):
        #tokenized_pos = rm_tokenizer(sample['positive'], truncation=True)
        #tokenized_neg = rm_tokenizer(sample['negative'], truncation=True)
        tokenized_pos = tokenizer(sample['prompt'] + sample['chosen'])
        tokenized_neg = tokenizer(sample['prompt'] + sample['rejected'])
        prompt_id = tokenizer(sample['prompt'])
        sample['tprompdt_ids'] = prompt_id['input_ids']
        sample["tchosen_input_ids"] = tokenized_pos["input_ids"]
        sample["trejected_input_ids"] = tokenized_neg["input_ids"]
        return sample        
    # 2. Load the Stack-exchange paired dataset
    train_dataset = get_stack_exchange_paired(data_dir=script_args.train_dir,sanity_check=script_args.sanity_check)
    '''
    train_dataset = train_dataset.filter(
        lambda x: len(x["tchosen_input_ids"]) <= script_args.max_length
        and len(x["trejected_input_ids"]) <= script_args.max_length
    )
    '''
    train_dataset = train_dataset.map(tokenize)
    train_dataset = train_dataset.filter(lambda x: len(x["tchosen_input_ids"]) <= 800 and len(x["trejected_input_ids"]) <= 800 and len(x['tprompdt_ids']) <= 300)

    # 3. Load evaluation dataset
    eval_dataset = get_stack_exchange_paired(data_dir=script_args.eval_dir, sanity_check=True)
    eval_dataset = eval_dataset.map(tokenize)
    '''
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    )
    '''
    eval_dataset = eval_dataset.filter(lambda x: len(x["tchosen_input_ids"]) <= 800 and len(x["trejected_input_ids"]) <= 800 and len(x['tprompdt_ids']) <= 300)

    
    print(train_dataset[0])

    print(train_dataset[1])
    print(len(train_dataset), len(eval_dataset))

    print(train_dataset[14124])
    # 4. initialize training arguments:
    print("1111")
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        #report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        #optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=script_args.run_name,
    )
    '''
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    '''
    print("22222")
    # 5. initialize the DPO trainer
#    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=script_args.learning_rate)

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        #optimizers=optimizer,
        #peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )
    print("begin to train")
    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
