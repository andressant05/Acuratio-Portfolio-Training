import json
import random
from openai import OpenAI
from fuzzywuzzy import fuzz

#💬 Few-shot templates
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

# 🔧 LLM settings
client = OpenAI(api_key="EMPTY", base_url="http://localhost:9000/v1")

SYSTEM = (
    "You are a professional dataset generator for machine learning. Your goal is to produce instruction-response pairs "
    "that are strictly based on the input document chunk. Never fabricate information or introduce concepts not explicitly supported by the text. "
    "Responses must be formal, technical, and reference the language or logic of the document exactly. Avoid summaries, advice, or motivation."
)

def build_prompt(fewshots, chunk_text, type_label, n=2):
    fs_txt = "\n\n".join(json.dumps(x, ensure_ascii=False) for x in fewshots)
    return f"""{SYSTEM}
--- FEW-SHOT {type_label.upper()} ---
{fs_txt}

--- DOCUMENT CONTENT ---
{chunk_text}

Now generate {n} instruction-response pairs in JSONL format. Base every detail strictly on the document content. Do not invent facts.
"""

def generate_for_chunk(text, questions_per_type=1, required_types=None):
    if required_types is None:
        required_types = ["definition", "justification", "scenario", "role"]
    all_qas = []
    for qtype in required_types:
        prompt = build_prompt(FEW_SHOTS[qtype], text, qtype, n=questions_per_type)
        response = call_llm(prompt)
        parsed = parse_response(response, qtype)
        all_qas.extend(parsed)
    return all_qas

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

def dedupe(entries, threshold=90):
    uniq = []
    for e in entries:
        if any(fuzz.ratio(e["prompt"], u["prompt"]) > threshold for u in uniq):
            continue
        uniq.append(e)
    return uniq
