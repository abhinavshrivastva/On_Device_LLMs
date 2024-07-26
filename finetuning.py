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

def main():
    # Training Configuration
    training_config = {
        "bf16": True,
        "do_eval": False,
        "learning_rate": 5.0e-06,
        "log_level": "info",
        "logging_steps": 20,
        "logging_strategy": "steps",
        "lr_scheduler_type": "cosine",
        "num_train_epochs": 3,
        "max_steps": -1,
        "output_dir": "./models",
        "overwrite_output_dir": True,
        "per_device_eval_batch_size": 1,
        "per_device_train_batch_size": 1,
        "remove_unused_columns": True,
        "save_steps": 100,
        "save_total_limit": 1,
        "seed": 0,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs":{"use_reentrant": False},
        "gradient_accumulation_steps": 1,
        "warmup_ratio": 0.2,
    }

    peft_config = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": "all-linear",
        "modules_to_save": None,
    }

    train_conf = TrainingArguments(**training_config)
    peft_conf = LoraConfig(**peft_config)

    # Model and Tokenizer
    checkpoint_path = input("Base HF model: ")
    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.model_max_length = 2048
    tokenizer.pad_token = tokenizer.unk_token 
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'

    # Dataset
    dataset_name = input("Enter HF dataset: ")
    split = input("Enter split (test/train): ")
    system_prompt = input("Enter system prompt: ")
    user = input("Enter user prompt column: ")
    assistant = input("Enter assistant prompt column: ")

    def format_instruction(sample):
        return {"text": PROMPT_TEMPLATE.format(system_prompt= system_prompt,prompt=sample[user], response=sample[assistant])}

    PROMPT_TEMPLATE = "[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nInput:\n{prompt} [/INST]\n\nOutput: {response}"
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.map(format_instruction, num_proc=4)

    # Training
    trainer = SFTTrainer(
        model=model,
        args=train_conf,
        peft_config=peft_conf,
        train_dataset=dataset,
        max_seq_length=2048,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True
    )

    gc.collect()
    torch.cuda.empty_cache()

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model("adapter")

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    # Save and Push Model to Hugging Face Hub
    model = checkpoint_path
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

    saved_name = input("Save your model name: ")
    model.save_pretrained(saved_name)
    tokenizer.save_pretrained(saved_name)

    your_username = input("Enter username: ")
    hf_token = input("Enter HF token: ")
    login(token=hf_token)

    repo = Repository(local_dir=saved_name, clone_from=f"{your_username}/{saved_name}")
    repo.push_to_hub()

    # Save the model ID to a file
    with open("model_id.txt", "w") as f:
        f.write(f"{your_username}/{saved_name}")


if __name__ == "__main__":
    main()
