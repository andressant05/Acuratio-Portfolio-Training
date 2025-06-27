import os
import json
import time
import difflib
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from fuzzywuzzy import fuzz

# 💬 Few-shot templates
FEW_SHOTS = {
    "definition": [
        {
            "prompt": "¿Qué es un equipo de tratamiento térmico según la norma?",
            "response": "Un equipo de tratamiento térmico es un sistema utilizado para calentar materiales mediante combustibles o electricidad, tal como se especifica en el capítulo 3 de la norma."
        },
        {
            "prompt": "¿Qué se entiende por peligro mecánico en el contexto de esta norma?",
            "response": "Un peligro mecánico incluye el movimiento de maquinaria, la expulsión de materiales, o el fallo estructural que puede causar daño físico."
        },
    ],
    "justification": [
        {
            "prompt": "¿Por qué es importante evitar el retroceso de llamaradas en los equipos?",
            "response": "Porque el retroceso puede causar explosiones o daños al sistema, lo cual pone en riesgo a los operadores y al entorno."
        },
        {
            "prompt": "¿Por qué deben marcarse adecuadamente las tuberías de conducción?",
            "response": "Porque un marcaje adecuado permite identificar peligros como temperaturas extremas o presión alta, ayudando a prevenir accidentes."
        },
    ],
    "hypothetical": [
        {
            "prompt": "Si los equipos carecen de accesos seguros para mantenimiento, ¿qué podría ocurrir?",
            "response": "Los operarios podrían exponerse a riesgos físicos al intentar operar o reparar el equipo sin medios de acceso adecuados."
        },
        {
            "prompt": "Si se omiten medidas contra la acumulación de gases tóxicos en el interior del equipo, ¿cuál podría ser el resultado?",
            "response": "Podría producirse una intoxicación o incluso una explosión, especialmente si no hay salidas de emergencia o ventilación adecuada."
        },
    ],
    "comparison": [
        {
            "prompt": "¿Cuál es la diferencia entre los techos accesibles y no accesibles según la norma?",
            "response": "Los techos accesibles deben contar con rampas seguras y barandillas, mientras que los no accesibles deben estar marcados o protegidos para evitar el acceso."
        },
    ],
    "scenario": [
        {
            "prompt": "Un operador necesita acceder a una zona elevada del equipo durante una inspección. ¿Qué requisitos debe cumplir esa zona?",
            "response": "Debe ser accesible mediante rampas seguras, estar equipada con barandillas, y cumplir los requisitos de seguridad indicados en el apartado 5.2.11."
        },
    ],
    "role": [
        {
            "prompt": "¿Cuál es la responsabilidad del usuario en cuanto a la instalación del equipo según la norma?",
            "response": "El usuario debe asegurarse de que el lugar de instalación cumple con los requisitos de seguridad y que se dispone de ventilación adecuada."
        },
        {
            "prompt": "¿Qué debe hacer el fabricante respecto al manual de instrucciones?",
            "response": "Debe proporcionar un manual técnico que incluya los métodos de operación, mantenimiento, protección personal requerida y riesgos previstos."
        },
    ]
}

# 🔧 LLM setup
client = OpenAI(api_key="EMPTY", base_url="http://localhost:9000/v1")

SYSTEM = (
    "You are a professional dataset generator for machine learning. Your goal is to produce instruction-response pairs "
    "that are strictly based on the input document chunk. Never fabricate information or introduce concepts not explicitly supported by the text. "
    "Responses must be formal, technical, and reference the language or logic of the document exactly. Avoid summaries, advice, or motivation."
)

def chunk_text(full_text, max_chars=2000):
    sentences = full_text.split(". ")
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < max_chars:
            current += (s + ". ")
        else:
            chunks.append(current.strip())
            current = s + ". "
    if current:
        chunks.append(current.strip())
    return chunks

def build_prompt(fewshots, chunk_text, type_label, n=2):
    fs_txt = "\n\n".join(json.dumps(x, ensure_ascii=False) for x in fewshots)
    return f"""{SYSTEM}
--- FEW-SHOT {type_label.upper()} ---
{fs_txt}

--- DOCUMENT CONTENT ---
{chunk_text}

Now generate {n} instruction-response pairs in JSONL format. Base every detail strictly on the document content. Do not invent facts.
"""

def call_llm(prompt):
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1024
    )
    return response.choices[0].message.content

def parse_response(response_text, qtype):
    parsed = []
    for line in response_text.strip().splitlines():
        try:
            item = json.loads(line)
            if "prompt" in item and "response" in item:
                item["type"] = qtype
                parsed.append(item)
        except json.JSONDecodeError:
            continue
    return parsed

def generate_for_chunk(text, questions_per_type=1, required_types=None):
    if required_types is None:
        required_types = ["definition", "justification", "scenario", "role"]
    all_qas = []
    for qtype in required_types:
        prompt = build_prompt(FEW_SHOTS[qtype], text, qtype, n=questions_per_type)
        response = call_llm(prompt)
        all_qas.extend(parse_response(response, qtype))
    return all_qas

def dedupe(entries, threshold=90):
    uniq = []
    for e in entries:
        if any(fuzz.ratio(e["prompt"], u["prompt"]) > threshold for u in uniq):
            continue
        uniq.append(e)
    return uniq

def is_duplicate(new_prompt, existing_prompts, threshold=0.9):
    for ep in existing_prompts:
        if difflib.SequenceMatcher(None, new_prompt.lower(), ep.lower()).ratio() > threshold:
            return True
    return False

def process_file(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    text = "\n\n".join(data["chunks"])
    chunks = chunk_text(text, max_chars=3000)
    selected = [c for c in chunks if len(c.strip()) > 300]
    out = []
    seen = set()

    for idx, ch in enumerate(selected):
        qas = generate_for_chunk(ch, questions_per_type=1)
        for qa in qas:
            prompt = qa["prompt"]
            if prompt in seen or is_duplicate(prompt, seen):
                continue
            seen.add(prompt)
            out.append({
                "context": ch.strip(),
                "prompt": prompt,
                "response": qa["response"],
                "metadata": {
                    "source_file": json_path.name,
                    "chunk_index": idx
                }
            })

    return dedupe(out)

def main():
    start = time.time()
    folder = Path("manuela_shower")
    json_files = sorted(folder.glob("*.json"))
    total_files = len(json_files)
    output_dir = Path("temp_outputs")
    output_dir.mkdir(exist_ok=True)
    results = []

    print(f"📁 Found {total_files} files to process.")

    completed = 0
    start_batch = time.time()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process_file, jf): jf for jf in json_files}
        for fut in as_completed(futures):
            file = futures[fut]
            try:
                entries = fut.result()
                print(f"✅ {file.name}: {len(entries)} QAs")
                results.extend(entries)

                out_path = output_dir / f"{file.stem}_qas.jsonl"
                with open(out_path, "w", encoding="utf-8") as f:
                    for qa in entries:
                        f.write(json.dumps(qa, ensure_ascii=False) + "\n")

                print(f"💾 Autosaved {len(entries)} QAs from {file.name}")
                completed += 1
                elapsed = time.time() - start_batch
                print(f"📊 Progress: {completed}/{total_files} | ⏱️ {elapsed:.1f}s elapsed ({(completed/total_files)*100:.1f}%)\n")
            except Exception as e:
                print(f"❌ Failed {file.name}: {e}")

    final = dedupe(results)
    with open("context_full_dataset.jsonl", "w", encoding="utf-8") as f:
        for qa in final:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    elapsed = time.time() - start
    print(f"✅ All done – dataset saved to context_full_dataset.jsonl")
    print(f"⏱️ Total time: {elapsed/60:.2f} minutes")

def main():
    start = time.time()
    folder = Path("manuela_shower")
    json_files = sorted(folder.glob("*.json"))
    output_dir = Path("temp_outputs")
    output_dir.mkdir(exist_ok=True)

    processed_path = Path(".processed_files.json")
    processed_files = set()
    if processed_path.exists():
        with open(processed_path, "r", encoding="utf-8") as pf:
            processed_files = set(json.load(pf))

    files_to_process = [f for f in json_files if f.name not in processed_files]
    total_files = len(files_to_process)
    results = []

    print(f"📁 Total new files to process: {total_files}")

    completed = 0
    start_batch = time.time()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process_file, jf): jf for jf in files_to_process}
        for fut in as_completed(futures):
            file = futures[fut]
            try:
                entries = fut.result()
                print(f"✅ {file.name}: {len(entries)} QAs")
                results.extend(entries)

                out_path = output_dir / f"{file.stem}_qas.jsonl"
                with open(out_path, "w", encoding="utf-8") as f:
                    for qa in entries:
                        f.write(json.dumps(qa, ensure_ascii=False) + "\n")

                # Mark as processed
                processed_files.add(file.name)
                with open(processed_path, "w", encoding="utf-8") as pf:
                    json.dump(sorted(processed_files), pf, ensure_ascii=False, indent=2)

                print(f"💾 Autosaved {len(entries)} QAs from {file.name}")
                completed += 1
                elapsed = time.time() - start_batch
                print(f"📊 Progress: {completed}/{total_files} | ⏱️ {elapsed:.1f}s elapsed ({(completed/total_files)*100:.1f}%)\n")
            except Exception as e:
                print(f"❌ Failed {file.name}: {e}")

    # Merge all files in temp_outputs/
    final = dedupe(results)
    with open("context_full_dataset.jsonl", "w", encoding="utf-8") as f:
        for qa in final:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    elapsed = time.time() - start
    print(f"✅ All done – dataset saved to context_full_dataset.jsonl")
    print(f"⏱️ Total time: {elapsed/60:.2f} minutes")
    
    
if __name__ == "__main__":
    main()
    
    
```
# ✅ Acuratio Model Training — Handoff Notes

Hola equipo 👋  
Aquí les dejo una guía clara para continuar el trabajo fácilmente.

---

## 🔁 Generación de Dataset (`generate_dataset_pipeline.py`)

### Cómo usarlo:
1. Coloca nuevos archivos `.json` dentro de la carpeta `manuela_shower/`.
2. Ejecuta:
   ```bash
   python generate_dataset_pipeline.py
   ```
3. El sistema procesará **sólo los archivos nuevos**.
4. El dataset final estará en: `context_full_dataset.jsonl`.

> 🧠 Se guarda un archivo `.processed_files.json` para llevar control de qué archivos ya fueron procesados.

---

## 🎓 Entrenamiento del Modelo (`train_pipeline.py`)

### Qué hace:
- Convierte `context_full_dataset.jsonl` al formato `ChatML`.
- Lanza entrenamiento sobre el modelo `meta-llama/Llama-3.2-3B-Instruct` con LoRA.

### Cómo usarlo:
```bash
python train_pipeline.py
```

### Tips:
- El entrenamiento actual incluye `context` como mensaje de sistema → más lento pero más preciso.
- Si desean entrenar más rápido (sin contexto):
  1. Modifiquen `convert_dataset_to_chatml()` para omitir el campo `"context"`.
  2. Entrenen con el nuevo dataset.

---

## 🛠️ Reentrenamiento vs Fine-tuning

- Para **entrenar desde cero**: simplemente reejecuten el pipeline.
- Para **continuar entrenamiento** sobre el modelo ya ajustado:
  - Cambien en `run_train_on_model()`:
    ```python
    resume_from_checkpoint=True
    base_model_name="Llama-3.2-3B-lora"
    ```

---

## 🤝 Dudas o mejoras

Para cualquier ajuste adicional, les recomiendo revisar los docstrings y comentarios en los `.py`. El código está modularizado y comentado.

¡Éxito en el proyecto! 🚀  
—Andres
```
