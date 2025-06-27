# Acuratio Model Training

Este repositorio automatiza por completo el flujo:  
**.docx → chunks .json →  generar dataset QA →  entrenar modelo LLaMA‑3 con LoRA**.

---

##  Estructura del Proyecto

```
Acuratio-Model-Training/
├── Dataset/                #  Pipeline para generar el dataset
│   └── generate_dataset_pipeline.py
├── Entrenar Modelo/        #  Entrenamiento del modelo LLaMA con LoRA
│   └── train_pipeline.py
├── Maquina Virtual/        #  Configuración para máquinas virtuales (GCP)
│   └── README.md
├── Nuevos Documentos/      #  DOCX nuevos y script para convertirlos a JSON
│   ├── parse_docx_to_chunks.py
│   ├── processed_chunks.zip
│   ├── nuevos_docs.zip
│   └── README.md
├── requirements.txt        #  Requisitos Python
└── README.md               #  Este archivo
```

---

##  Flujo en 3 pasos

### 1. Añadir nuevos `.docx`
Desde la carpeta principal:

```bash
cd Nuevos\ Documentos/
unzip processed_chunks.zip
unzip nuevos_docs.zip
python parse_docx_to_chunks.py
```

→ Esto generará archivos `.json` automáticamente dentro de `processed_chunks/`, listos para el siguiente paso.

---

### 2. Generar Dataset

```bash
cd ../Dataset/
python generate_dataset_pipeline.py
```

→ Crea el archivo `context_full_dataset.jsonl` (formato instructivo de QA).

---

### 3. Entrenar el Modelo

```bash
cd ../Entrenar\ Modelo/
python train_pipeline.py
```

→ Usa LoRA y guarda el modelo ajustado en `results_lora/`.

---

##  Instalación del entorno

```bash
git clone https://github.com/andressant05/Acuratio-Model-Training.git
cd Acuratio-Model-Training
pip install -r requirements.txt
```

---

##  Qué hace cada script

### `parse_docx_to_chunks.py`
- Limpia errores comunes de OCR
- Divide por secciones lógicas
- Corta en fragmentos ≤ 1000 palabras
- Elimina encabezados y duplicados
- Guarda `.json` directamente en `processed_chunks/`

### `generate_dataset_pipeline.py`
- Lee todos los `.json` de `processed_chunks/`
- Genera pares pregunta-respuesta vía LLM (LLaMA 70B)
- Evita duplicados con fuzzy logic
- Guarda el dataset final en `context_full_dataset.jsonl`

### `train_pipeline.py`
- Convierte el dataset al formato ChatML
- Aplica LoRA sobre un LLaMA-3 base
- Entrena y guarda el modelo afinado

---

## Despliegue en la nube (opcional)

Consulta `Maquina Virtual/README.md` para ejecutar los modelos con `vllm` y GPUs A100 en GCP.

---

## Autor y contacto

**andressant05**  
Gracias por continuar con el proyecto. ¡A entrenar modelos con estilo!
