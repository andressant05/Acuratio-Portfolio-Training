# Acuratio Model Training

Este repositorio contiene todo lo necesario para generar un dataset estructurado en español y entrenar un modelo de lenguaje basado en LLaMA-3 usando LoRA. Se incluyen instrucciones detalladas, scripts modulares y configuraciones de máquinas virtuales para facilitar el trabajo colaborativo.

## Estructura del Proyecto

```
Acuratio-Model-Training/
│
├── dataset/                      # Lógica de generación del dataset
│   ├── fewshot_templates.py
│   ├── chunker.py
│   ├── generator.py
│   ├── postprocessor.py
│   └── run_generate_dataset.py
│
├── training/                     # Código para entrenamiento del modelo
│   ├── train_model.py
│   ├── training_model.py
│   ├── prepare_dataset_and_train.py
│   └── training_model_notebook.ipynb
│
├── virtual_machines/            # Instrucciones de configuración de VMs
│   ├── README_andres_vllm.md
│   └── README_jonander_a100.md
│
├── context_full_dataset.jsonl
├── full_contexted_manual_trained_dataset.jsonl
├── requirements.txt
└── README.md
```

## Objetivo

El proyecto tiene como objetivo construir un sistema de entrenamiento de modelos LLaMA-3 basado en documentación técnica. El sistema genera preguntas y respuestas técnicas en español a partir de documentos, entrena modelos con LoRA y permite su despliegue en VMs con vllm.

## Instalación

1. Clona el repositorio:
   git clone https://github.com/tu_usuario/Acuratio-Model-Training.git
   cd Acuratio-Model-Training

2. Instala los requerimientos:
   pip install -r requirements.txt

## Generación de Dataset

cd dataset
python run_generate_dataset.py

El resultado se guarda como context_full_dataset.jsonl.

## Conversión de Dataset para Entrenamiento

python training/prepare_dataset_and_train.py

Este script convierte el dataset anterior al formato esperado por ChatML.

## Entrenamiento del Modelo

python training/train_model.py

Esto entrena el modelo usando LoRA y guarda los pesos entrenados.

## Configuración de Máquinas Virtuales

Consulta los archivos dentro de la carpeta /virtual_machines para desplegar el modelo con vllm en Google Cloud Platform.

## Contacto

Cualquier duda o mejora, contactar con andressant05.
