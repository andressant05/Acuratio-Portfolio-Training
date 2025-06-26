
# Acuratio Model Training

Este repositorio contiene todo lo necesario para generar un dataset técnico en español y entrenar un modelo de lenguaje basado en LLaMA-3 utilizando LoRA. Incluye pipelines automatizados y configuración para su despliegue en máquinas virtuales con `vllm`.

---

## Estructura del Proyecto

```
Acuratio-Model-Training/
│
├── Dataset/
│   └── generate_dataset_pipeline.py         # Generación completa del dataset
│
├── Entrenar Modelo/
│   └── train_pipeline.py                    # Pipeline de entrenamiento con LoRA
│
├── Maquina Virtual/
│   └── README.md                            # Configuración de entorno en VMs (GCP)
│
├── requirements.txt                         # Requisitos del entorno
└── README.md                                # Este archivo
```

---

## Objetivo

Construir un sistema modular y reproducible para:

- Generar datasets QA a partir de documentación técnica en español.
- Convertir los datos al formato `ChatML`.
- Entrenar modelos LLaMA-3 con `LoRA`.
- Desplegar el modelo en entornos con GPUs (A100) mediante `vllm`.

---

## Instalación

1. Clona el repositorio:
```
git clone https://github.com/tu_usuario/Acuratio-Model-Training.git
cd Acuratio-Model-Training
```

2. Instala las dependencias:
```
pip install -r requirements.txt
```

---

## Generación del Dataset

Ejecuta el siguiente pipeline para generar el dataset inicial estructurado:
```
python Dataset/generate_dataset_pipeline.py
```
- El archivo de salida será: `context_full_dataset.jsonl`

---

## Entrenamiento del Modelo

Lanza el entrenamiento del modelo sobre el dataset procesado:
```
python Entrenar\\ Modelo/train_pipeline.py
```
- Entrena con LoRA y guarda los pesos finos.
- Incluye la conversión automática a `ChatML`.

---

## Configuración de la Máquina Virtual

Consulta el siguiente archivo para instrucciones completas de despliegue:
```
Maquina Virtual/README.md
```
- Incluye configuración de `nginx`, `Jupyter Lab`, `SSH`, y ejecución de `vllm` con Docker.

---

## Contacto

Para dudas, mejoras o colaboración: **andressant05**


