import os
import json
import time
import difflib
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from fuzzywuzzy import fuzz

# 💬 Ejemplos que sirven de guía (few-shots) para el modelo LLM
EJEMPLOS_TIPICOS = {
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

# 🔧 Cliente para comunicarse con el modelo LLM (en este caso, llama3 local)
cliente_llm = OpenAI(api_key="EMPTY", base_url="http://localhost:9000/v1")

# 🎯 Instrucción general que guía al modelo
INSTRUCCION_SISTEMA = (
    "You are a professional dataset generator for machine learning. Your goal is to produce instruction-response pairs "
    "that are strictly based on the input document chunk. Never fabricate information or introduce concepts not explicitly supported by the text. "
    "Responses must be formal, technical, and reference the language or logic of the document exactly. Avoid summaries, advice, or motivation."
)

# 🔪 Divide un texto largo en fragmentos manejables de cierto tamaño
def dividir_texto(texto_completo, max_caracteres=2000):
    oraciones = texto_completo.split(". ")
    fragmentos, actual = [], ""
    for oracion in oraciones:
        if len(actual) + len(oracion) < max_caracteres:
            actual += (oracion + ". ")
        else:
            fragmentos.append(actual.strip())
            actual = oracion + ". "
    if actual:
        fragmentos.append(actual.strip())
    return fragmentos

# 🧱 Construye el prompt con los ejemplos + texto a analizar
def construir_prompt(ejemplos, fragmento, tipo, n=2):
    ejemplos_str = "\n\n".join(json.dumps(x, ensure_ascii=False) for x in ejemplos)
    return f"""{INSTRUCCION_SISTEMA}
--- FEW-SHOT {tipo.upper()} ---
{ejemplos_str}

--- DOCUMENTO ---
{fragmento}

Now generate {n} instruction-response pairs in JSONL format. Base every detail strictly on the document content. Do not invent facts.
"""

# 🔮 Envía el prompt al modelo y obtiene la respuesta
def llamar_llm(prompt):
    respuesta = cliente_llm.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": INSTRUCCION_SISTEMA},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1024
    )
    return respuesta.choices[0].message.content

# 📤 Interpreta la respuesta del modelo y extrae los pares válidos
def interpretar_respuesta(texto_respuesta, tipo):
    resultado = []
    for linea in texto_respuesta.strip().splitlines():
        try:
            item = json.loads(linea)
            if "prompt" in item and "response" in item:
                item["type"] = tipo
                resultado.append(item)
        except json.JSONDecodeError:
            continue
    return resultado

# 🤖 Genera pares QA para cada fragmento de texto
def generar_qa_para_fragmento(texto, preguntas_por_tipo=1, tipos_requeridos=None):
    if tipos_requeridos is None:
        tipos_requeridos = ["definition", "justification", "scenario", "role"]
    resultado = []
    for tipo in tipos_requeridos:
        prompt = construir_prompt(EJEMPLOS_TIPICOS[tipo], texto, tipo, n=preguntas_por_tipo)
        respuesta = llamar_llm(prompt)
        resultado.extend(interpretar_respuesta(respuesta, tipo))
    return resultado

# 🧹 Elimina duplicados similares entre prompts
def eliminar_duplicados(lista, umbral=90):
    unicos = []
    for item in lista:
        if any(fuzz.ratio(item["prompt"], otro["prompt"]) > umbral for otro in unicos):
            continue
        unicos.append(item)
    return unicos

# 🔁 Compara prompts para evitar repetir
def es_duplicado(prompt_nuevo, prompts_existentes, umbral=0.9):
    for existente in prompts_existentes:
        if difflib.SequenceMatcher(None, prompt_nuevo.lower(), existente.lower()).ratio() > umbral:
            return True
    return False

# 🔍 Procesa un archivo JSON, genera pares y adjunta metadatos
def procesar_archivo(ruta_json):
    datos = json.loads(ruta_json.read_text(encoding="utf-8"))
    texto = "\n\n".join(datos["chunks"])
    fragmentos = dividir_texto(texto, max_caracteres=3000)
    fragmentos_filtrados = [f for f in fragmentos if len(f.strip()) > 300]
    resultado = []
    prompts_vistos = set()

    for idx, frag in enumerate(fragmentos_filtrados):
        pares = generar_qa_para_fragmento(frag, preguntas_por_tipo=1)
        for qa in pares:
            prompt = qa["prompt"]
            if prompt in prompts_vistos or es_duplicado(prompt, prompts_vistos):
                continue
            prompts_vistos.add(prompt)
            resultado.append({
                "context": frag.strip(),
                "prompt": prompt,
                "response": qa["response"],
                "metadata": {
                    "source_file": ruta_json.name,
                    "chunk_index": idx
                }
            })

    return eliminar_duplicados(resultado)

# 🧠 Ejecución principal del pipeline: detecta archivos nuevos, procesa y graba
def main():
    inicio = time.time()
    carpeta = Path("processed_chunks")
    archivos = sorted(carpeta.glob("*.json"))
    carpeta_salida = Path("temp_outputs")
    carpeta_salida.mkdir(exist_ok=True)

    ruta_procesados = Path(".processed_files.json")
    archivos_procesados = set()
    if ruta_procesados.exists():
        with open(ruta_procesados, "r", encoding="utf-8") as f:
            archivos_procesados = set(json.load(f))

    nuevos_archivos = [f for f in archivos if f.name not in archivos_procesados]
    print(f"📁 Archivos nuevos a procesar: {len(nuevos_archivos)}")

    resultados = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        tareas = {pool.submit(procesar_archivo, a): a for a in nuevos_archivos}
        for t in as_completed(tareas):
            archivo = tareas[t]
            try:
                pares = t.result()
                print(f"✅ {archivo.name}: {len(pares)} QAs generadas")
                resultados.extend(pares)

                salida = carpeta_salida / f"{archivo.stem}_qas.jsonl"
                with open(salida, "w", encoding="utf-8") as f:
                    for qa in pares:
                        f.write(json.dumps(qa, ensure_ascii=False) + "\n")

                archivos_procesados.add(archivo.name)
                with open(ruta_procesados, "w", encoding="utf-8") as f:
                    json.dump(sorted(archivos_procesados), f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"❌ Error al procesar {archivo.name}: {e}")

    resultado_final = eliminar_duplicados(resultados)
    with open("context_full_dataset.jsonl", "w", encoding="utf-8") as f:
        for qa in resultado_final:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    duracion = time.time() - inicio
    print(f"✅ ¡Listo! Dataset generado en context_full_dataset.jsonl")
    print(f"🕐 Tiempo total: {duracion/60:.2f} minutos")

if __name__ == "__main__":
    main()

    
    
"""
# ✅ Acuratio Model Training — Handoff Notes

Hola equipo 👋  
Aquí les dejo una guía clara para continuar el trabajo fácilmente.

---

## 🔁 Generación de Dataset (`generate_dataset_pipeline.py`)

### Cómo usarlo:
1. Coloca nuevos archivos `.json` dentro de la carpeta `processed_chunks/`.
2. Ejecuta:
   python generate_dataset_pipeline.py
3. El sistema procesará **sólo los archivos nuevos**.
4. El dataset final estará en: context_full_dataset.jsonl

> 🧠 Se guarda un archivo `.processed_files.json` para llevar control de qué archivos ya fueron procesados.

"""
