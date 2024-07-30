import os
import gc
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig,
)
from trl import SFTTrainer
from huggingface_hub import HfApi, HfFolder, Repository, login
import json
import torch
from transformers import TrainingArguments, Trainer,AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import  AutoModelForCausalLM
from huggingface_hub import login
## inputs
login()
dataset_name = input("Enter HF dataset ID")
split = input("Enter split(test/train)")
system_prompt = input("Enter System Prompt")
user = input("Enter User Column")
assistant = input("Enter Assistant Column")
model_name = input("Enter model you want to finetune")
saved_name = input("Enter model name to save")
your_username = input("Enter your HF username")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.enable_input_require_grads()
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
##preprocessing
def format_instruction(sample):
    SYSTEM_PROMPT = system_prompt
    question = sample[user]
    correct_answer = sample[assistant]
    return {
        "prompt": SYSTEM_PROMPT + question,
        "completion": correct_answer
    }

dataset = load_dataset(dataset_name, split=split)
formatted_data = dataset.map(format_instruction, remove_columns=dataset.features)
formatted_data.to_json("train.jsonl")
def preprocess_function(examples):
    inputs = [f"{prompt}\n" for prompt in examples["prompt"]]
    targets = [f"{completion}\n" for completion in examples["completion"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
dataset = load_dataset("json", data_files="train.jsonl")
tokenized_dataset = dataset["train"].train_test_split(test_size=0.1)
tokenized_dataset = tokenized_dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    learning_rate=1e-4,
    fp16=True,
    gradient_checkpointing=True,
    gradient_accumulation_steps=4,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model("adapter")
model = model_name
tokenizer = AutoTokenizer.from_pretrained(model)
fp16_model = AutoModelForCausalLM.from_pretrained(
        model,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
model = PeftModel.from_pretrained(fp16_model, "adapter")
model = model.merge_and_unload()
repo = Repository(local_dir=saved_name, clone_from=f"{your_username}/{saved_name}")
repo.push_to_hub()
with open("model_id.txt", "w") as f:
  f.write(f"{your_username}/{saved_name}")
