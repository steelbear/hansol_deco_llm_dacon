import os
import os.path as osp
from string import Template
import sys
import fire
import json
from typing import List, Union

import numpy as np
import pandas as pd

import torch
from torch.nn import functional as F

import transformers
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict
)
from peft import PeftModel


device = 'auto'
base_LLM_model = 'beomi/OPEN-SOLAR-KO-10.7B'

batch_size = 16
num_epochs = 1
micro_batch = 2
gradient_accumulation_steps = batch_size // micro_batch

cutoff_len = 4096
lr_scheduler = 'cosine'
warmup_ratio = 0.06
learning_rate = 1e-4
optimizer = 'adamw_torch'
weight_decay = 0.01
max_grad_norm = 1.0

lora_r = 16
lora_alpha = 16
lora_dropout = 0.05
lora_target_modules = ["gate_proj", "down_proj", "up_proj"]

train_on_inputs = False
add_eos_token = False

resume_from_checkpoint = False
output_dir = './custom_LLM'
final_save_folder = './custom_LLM_final'
output_hugggingface_repo = 'SteelBear/open-solar-hansol-0308'

prompt_template = Template(
    """### System:
${system}

### User:
${user}

### Assistant:
""")

config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)


def train_generator():
    for _, row in train_df.iterrows():
        questions = row[['질문_1', '질문_2']].values
        answers = row[['답변_1', '답변_2', '답변_3', '답변_4', '답변_5']].values
        for question in questions:
            for answer in answers:
                yield dict(question=question, answer=answer)
    
    indice = np.random.randin([0, 0, 0], [len(train_df), 2, 5], size=(256, 3))
    for i in range(128):
        i_1, q_1, a_1 = indice[2 * i - 1]
        i_2, q_2, a_2 = indice[2 * i]

        question_1 = train_df.iloc[i_1, q_1 + 1]
        answer_1 = train_df.iloc[i_1, a_1 + 4]
        question_2 = train_df.iloc[i_2, q_2 + 1]
        answer_2 = train_df.iloc[i_2, a_2 + 4]

        question = question_1 + [" 그리고 ", " 또한 ", " 또, "][i // 3] + question_2
        answer = answer_1 + " " + answer_2
        yield dict(question=question, answer=answer)


def preprocess(example):
  prompt = prompt_template.substitute(
      system="당신은 건물 도배 전문가로써 User의 질문에 알맞은 답변을 제시하세요.",
      user=example['question']
  )
  prompt_with_answer = prompt + example['answer']

  full_tokenized = tokenizer(prompt_with_answer)

  # add eos token
  full_tokenized['input_ids'].append(eos)
  full_tokenized['attention_mask'].append(1)

  nums_prompt_tokens = len(tokenizer.encode(prompt))

  full_tokenized['labels'] = [-100] * nums_prompt_tokens + full_tokenized['input_ids'][nums_prompt_tokens:]

  return full_tokenized


if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained(
        base_LLM_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device)

    tokenizer = AutoTokenizer.from_pretrained(base_LLM_model)

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    print("BOS token:", bos)
    print("EOS token:", eos)
    print("PAD toekn:", pad)

    if pad == None or pad == eos:
        tokenizer.pad_token_id = 0
    print("length of tokenizer:", len(tokenizer))

    train_df = pd.read_csv('data/train.csv')
    train_dataset = Dataset.from_generator(train_generator)
    val_dataset = None

    train_dataset = train_dataset.shuffle()
    train_dataset = train_dataset.map(preprocess)

    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, config)

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )

        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )
            resume_from_checkpoint = (
                True
            )
        
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")


    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="no",
            save_strategy="steps",
            max_grad_norm=max_grad_norm,
            save_steps=30,
            lr_scheduler_type=lr_scheduler,
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            group_by_length=False,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True,
        ),
    )

    model.config.use_cache = False
    model.print_trainable_parameters()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save({}, model_path)
    tokenizer.save_pretrained(output_dir)

    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_LLM_model,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device
    )

    model = PeftModel.from_pretrained(base_model, output_dir, device)
    model = model.merge_and_unload()

    model.save_pretrained(final_save_folder)
    tokenizer.save_pretrained(final_save_folder)

    model = AutoModelForCausalLM.from_pretrained(final_save_folder)
    model.push_to_hub(output_hugggingface_repo, token=True)
    tokenizer = AutoTokenizer.from_pretrained(final_save_folder)
    tokenizer.push_to_hub(output_hugggingface_repo, token=True)