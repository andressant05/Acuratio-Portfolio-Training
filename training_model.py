from peft import PeftModel, LoraConfig, get_peft_model 

from transformers import ( 

    AutoModelForCausalLM, 

    AutoTokenizer, 

    TrainingArguments, 

    pipeline, 

    TrainerCallback 

) 

from trl import SFTTrainer 

from datasets import load_dataset, Dataset 

from huggingface_hub import login 

import warnings 

import os 

import torch 

import evaluate 

import random 

import json 

from datetime import datetime 

  

  

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.hub") 

  

# Set GPU devices 

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' 

  

# Login 

login(token="***REMOVED***") 

  

def preprocess_dataset(dataset_name, split="train"): 

    dataset = load_dataset(dataset_name, split=split, cache_dir=os.path.expanduser("~/.cache/huggingface_datasets")) 

    sample = dataset[0] 

  

    if "messages" in sample: 

        return dataset  # leave messages as-is 

  

    elif "text" in sample: 

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

    user, assistant = example["messages"] 

    q = user["content"].strip() 

    a = assistant["content"].strip() 

    text = ( 

        "<|begin_of_text|>" 

        "<|start_header_id|>system<|end_header_id|>\n\n" 

        "Cutting Knowledge Date: December 2023\n" 

        "Today Date: 23 Jun 2025\n\n" 

        "Eres un asistente técnico experto en responder preguntas sobre normativas ISO. " 

        "Tu respuesta debe ser clara, precisa y directa, si es necesario, indica que se basa en tu conocimiento previo." 

        "<|eot_id|>" 

        "<|start_header_id|>user<|end_header_id|>\n\n" 

        f"{q}" 

        "<|eot_id|>" 

        "<|start_header_id|>assistant<|end_header_id|>\n\n" 

        f"{a}" 

        "<|eot_id|>" 

    ) 

    return {"text": text} 

  

  

def clean_chatml_dataset(dataset): 

    cleaned = [] 

    removed = 0 

  

    for ex in dataset: 

        messages = ex.get("messages", []) 

        if not messages or not isinstance(messages, list): 

            removed += 1 

            continue 

  

        # Only keep valid roles 

        valid = [m for m in messages if m.get("role") in {"user", "assistant"} and m.get("content")] 

        if len(valid) < 2 or valid[-1]["role"] != "assistant": 

            removed += 1 

            continue 

  

        # Get last user → assistant pair 

        for i in range(len(valid) - 1, 0, -1): 

            if valid[i]["role"] == "assistant" and valid[i-1]["role"] == "user": 

                cleaned.append({"messages": [valid[i-1], valid[i]]}) 

                break 

        else: 

            removed += 1 

  

    print(f"✅ Cleaned {len(cleaned)} samples | 🗑️ Removed {removed}") 

    return cleaned 

  

def run_train_on_model( 

    hu_fa_data="mlabonne/guanaco-llama2-1k", 

    output="./results_lora", 

    base_model_name="meta-llama/Llama-3.2-3B-Instruct",  

    new_model_name="Llama-3.2-3B-lora", 

    epochs=5, 

    batch_size=4, 

    lr=4e-5, 

    resume_from_checkpoint = True 

): 

    tokenizer = AutoTokenizer.from_pretrained( 

        base_model_name, 

        trust_remote_code=True, 

        use_fast=True, 

        cache_dir="hello_moto" 

    ) 

    tokenizer.pad_token = tokenizer.eos_token 

    tokenizer.padding_side = "right" 

  

    base_model = AutoModelForCausalLM.from_pretrained( 

        base_model_name, 

        device_map="auto", 

        trust_remote_code=True, 

        cache_dir="hello_moto" 

    ) 

    base_model.config.use_cache = False 

    base_model.config.pretraining_tp = 1 

     

    # After loading base_model 

    base_model.gradient_checkpointing_enable() 

  

    raw_data = preprocess_dataset(hu_fa_data) if isinstance(hu_fa_data, str) else hu_fa_data 

     

    formatted_data = raw_data.map(chat_formatting_fn) 

    if "messages" in formatted_data.column_names: 

        formatted_data = formatted_data.remove_columns("messages") 

    training_data = formatted_data 

  

  

  

    train_params = TrainingArguments( 

        output_dir=output, 

        num_train_epochs=epochs, 

        per_device_train_batch_size=batch_size, 

        gradient_accumulation_steps=2, 

        optim="paged_adamw_32bit", 

        save_strategy="epoch",  # or 2000 if you’re feeling bold 

        save_total_limit=3,  # Keep only last 3 checkpoints 

        logging_steps=100,  # still get a good sense of training dynamics 

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

        train_dataset=training_data, 

        peft_config=peft_parameters, 

        args=train_params 

    ) 

  

  

    trainer.train(resume_from_checkpoint=resume_from_checkpoint) 

    trainer.model.save_pretrained(new_model_name) 

    print("Successfully saved the model!") 

  

def generate_response_from_lora_model( 

    base_model_name: str, 

    adapter_model_path: str, 

    question: str, 

    max_new_tokens: int = 200, 

    temperature: float = 0.0 

) -> str: 

    base = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True, device_map="auto") 

    model = PeftModel.from_pretrained(base, adapter_model_path) 

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True) 

    tokenizer.pad_token = tokenizer.eos_token 

  

    prompt = ( 

        "<|begin_of_text|>" 

        "<|start_header_id|>system<|end_header_id|>\n\n" 

        "Cutting Knowledge Date: December 2023\n" 

        f"Today Date: {datetime.now().strftime('%d %b %Y')}\n\n" 

        "Eres un asistente técnico experto en responder preguntas sobre normativas ISO. " 

        "Tu respuesta debe ser clara, precisa y directa, si es necesario, indica que se basa en tu conocimiento previo." 

        "<|eot_id|>" 

        "<|start_header_id|>user<|end_header_id|>\n\n" 

        f"{question}" 

        "<|eot_id|>" 

        "<|start_header_id|>assistant<|end_header_id|>\n\n" 

    ) 

  

    generator = pipeline( 

        "text-generation", model=model, tokenizer=tokenizer, 

        do_sample=False, num_beams=4, 

        max_new_tokens=max_new_tokens, 

        temperature=temperature, 

        pad_token_id=tokenizer.pad_token_id 

    ) 

    out = generator(prompt)[0]["generated_text"] 

    return out[len(prompt):].split("<|eot_id|>")[0].strip() 

   

  

if __name__ == "__main__": 

    run_train_on_model() 

 
