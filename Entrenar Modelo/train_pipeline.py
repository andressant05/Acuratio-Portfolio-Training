import os
import json
import warnings
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import PeftModel, LoraConfig, get_peft_model
from huggingface_hub import login
from trl import SFTTrainer

# 🔧 CONFIGURACIÓN GLOBAL
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # GPUs disponibles
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.hub")
login(token=hugging_face_token)  # Autenticación Hugging Face

# 🔁 FUNCION: convierte el dataset JSONL original en formato ChatML
def convertir_dataset_a_chatml(ruta_entrada, ruta_salida):
    with open(ruta_entrada, "r", encoding="utf-8") as fin, open(ruta_salida, "w", encoding="utf-8") as fout:
        for linea in fin:
            try:
                item = json.loads(linea)
                contexto = item["context"]
                prompt = item["prompt"]
                respuesta = item["response"]
                formato = {
                    "messages": [
                        {"role": "system", "content": contexto},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": respuesta}
                    ]
                }
                fout.write(json.dumps(formato, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"❌ Línea ignorada debido a error: {e}")
    print("✅ Dataset convertido a ChatML en:", ruta_salida)

# 🧼 FUNCION: preprocesado del dataset si viene en otro formato
def preprocesar_dataset(nombre_dataset, division="train"):
    dataset = load_dataset(nombre_dataset, split=division,
                           cache_dir=os.path.expanduser("~/.cache/huggingface_datasets"))
    muestra = dataset[0]

    if "messages" in muestra or "text" in muestra:
        return dataset
    elif {"instruction", "input", "output"}.issubset(muestra.keys()):
        def unir_campos(ej):
            return {
                "text": (
                    f"### Instruction:\n{ej['instruction']}\n\n"
                    f"### Input:\n{ej['input']}\n\n"
                    f"### Response:\n{ej['output']}"
                )
            }
        return dataset.map(unir_campos)
    else:
        raise ValueError("Estructura de dataset no soportada.")

# 🧠 FUNCION: formatea cada ejemplo al estilo ChatML con encabezados y fecha actual
def formato_chat_para_llama(ejemplo):
    contexto = ""
    for msg in ejemplo["messages"]:
        if msg["role"] == "system":
            contexto = msg["content"].strip()

    msg_usuario = next((m["content"].strip() for m in ejemplo["messages"] if m["role"] == "user"), "")
    msg_asistente = next((m["content"].strip() for m in ejemplo["messages"] if m["role"] == "assistant"), "")
    fecha_actual = datetime.now().strftime("%d %b %Y")

    texto = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "Fecha de corte del conocimiento: Diciembre 2023\n"
        f"Fecha actual: {fecha_actual}\n\n"
        "Eres un asistente útil, preciso y conciso. Prioriza el contexto proporcionado al responder.\n\n"
        f"Contexto:\n{contexto}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{msg_usuario}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{msg_asistente}"
        "<|eot_id|>"
    )

    return {"text": texto}

# 🔥 FUNCION: realiza el entrenamiento con LoRA sobre el modelo base o de checkpoint
def entrenar_modelo(
    datos_hf="full_contexted_manual_trained_dataset.jsonl",
    salida="./results_lora",
    modelo_base="meta-llama/Llama-3.2-3B-Instruct",
    nombre_modelo_nuevo="Llama-3.2-3B-lora",
    epocas=1,
    tam_batch=4,
    lr=4e-5,
    reanudar=False
):
    # Tokenizador
    tokenizer = AutoTokenizer.from_pretrained(modelo_base, trust_remote_code=True, use_fast=True, cache_dir="hello_moto")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Modelo base con checkpoints y gradient checkpointing
    base = AutoModelForCausalLM.from_pretrained(modelo_base, device_map="auto", trust_remote_code=True, cache_dir="hello_moto")
    base.config.use_cache = False
    base.config.pretraining_tp = 1
    base.gradient_checkpointing_enable()

    # Carga o usa el dataset
    datos_raw = preprocesar_dataset("json", split="train") if isinstance(datos_hf, str) else datos_hf
    datos_formateados = datos_raw.map(formato_chat_para_llama)

    if "messages" in datos_formateados.column_names:
        datos_formateados = datos_formateados.remove_columns("messages")

    # Parámetros de entrenamiento
    args = TrainingArguments(
        output_dir=salida,
        num_train_epochs=epocas,
        per_device_train_batch_size=tam_batch,
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

    # Configuración LoRA
    parametros_lora = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )

    modelo_peft = get_peft_model(base, parametros_lora)

    trainer = SFTTrainer(
        model=modelo_peft,
        train_dataset=datos_formateados,
        peft_config=parametros_lora,
        args=args
    )

    print(f"🚀 Iniciando entrenamiento — guardando como: {nombre_modelo_nuevo}")
    trainer.train(resume_from_checkpoint=reanudar)
    trainer.model.save_pretrained(nombre_modelo_nuevo)
    print("✅ Modelo guardado exitosamente:", nombre_modelo_nuevo)

# ✅ MAIN: orquesta flujo completo del pipeline de entrenamiento
def main():
    convertir_dataset_a_chatml(
        "context_full_dataset.jsonl",
        "full_contexted_manual_trained_dataset.jsonl"
    )
    dataset_cargado = load_dataset("json", data_files="full_contexted_manual_trained_dataset.jsonl", split="train")
    entrenar_modelo(hu_fa_data=dataset_cargado)

# 🛠️ Ejecuta main solo si se llama este archivo directamente
if __name__ == "__main__":
    main()

"""
📌 NOTA:
Este script incluye el campo 'contexto' como mensaje de sistema, lo que mejora la calidad
del modelo pero aumenta ~3× el tiempo de entrenamiento.

👉 Para un entrenamiento más rápido (sin contexto):
   - Modificar `convertir_dataset_a_chatml` para que omita el campo 'contexto'
   - Entrenar con ese dataset simplificado
   - Luego, opcionalmente, hacer fine-tuning con dataset completo
"""
