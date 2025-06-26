# train_model.py

from peft import PeftModel, LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
)
from trl import SFTTrainer
from datasets import load_dataset
from huggingface_hub import login
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.hub")

# Set GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# Authenticate Hugging Face
login(token="***REMOVED***")

def preprocess_dataset(dataset_name, split="train"):
    dataset = load_dataset(dataset_name, split=split, cache_dir=os.path.expanduser("~/.cache/huggingface_datasets"))
    sample = dataset[0]

    if "messages" in sample or "text" in sample:
        return dataset
    elif {"instruction", "input", "output"}.issubset(sample.keys()):
        def merge_fields(example):
            return {
                "text": f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
            }
        return dataset.map(merge_fields)
    else:
        raise ValueError("Unsupported dataset structure.")

def chat_formatting_fn(example):
    context = ""
    for msg in example["messages"]:
        if msg["role"] == "system":
            context = msg["content"].strip()

    user_msg = next((m["content"].strip() for m in example["messages"] if m["role"] == "user"), "")
    assistant_msg = next((m["content"].strip() for m in example["messages"] if m["role"] == "assistant"), "")
    current_date = datetime.now().strftime("%d %b %Y")

    text = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "Fecha de corte del conocimiento: Diciembre 2023\n"
        f"Fecha actual: {current_date}\n\n"
        "Eres un asistente útil, preciso y conciso. Prioriza el contexto proporcionado al responder preguntas del usuario.\n"
        "Sé claro y exacto. Utiliza tu conocimiento cuando sea relevante.\n\n"
        f"Contexto:\n{context}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant_msg}"
        "<|eot_id|>"
    )

    return {"text": text}

def run_train_on_model(
    hu_fa_data="mlabonne/guanaco-llama2-1k",
    output="./results_lora",
    base_model_name="meta-llama/Llama-3.2-3B-Instruct",
    new_model_name="Llama-3.2-3B-lora",
    epochs=5,
    batch_size=4,
    lr=4e-5,
    resume_from_checkpoint=True
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_fast=True, cache_dir="hello_moto")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", trust_remote_code=True, cache_dir="hello_moto")
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    base_model.gradient_checkpointing_enable()

    raw_data = preprocess_dataset(hu_fa_data) if isinstance(hu_fa_data, str) else hu_fa_data
    formatted_data = raw_data.map(chat_formatting_fn)
    if "messages" in formatted_data.column_names:
        formatted_data = formatted_data.remove_columns("messages")

    train_params = TrainingArguments(
        output_dir=output,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=100,
        learning_rate=lr,
        weight_decay=0.0,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard"
    )

    peft_parameters = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(base_model, peft_parameters)

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_data,
        peft_config=peft_parameters,
        args=train_params
    )

    print(f"🚀 Starting training – saving to: {new_model_name}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.model.save_pretrained(new_model_name)
    print("✅ Successfully saved the model!")

if __name__ == "__main__":
    run_train_on_model()
