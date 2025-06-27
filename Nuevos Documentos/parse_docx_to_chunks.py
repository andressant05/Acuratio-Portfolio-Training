import os
import re
import json
from docx import Document
from pathlib import Path
from typing import List

# 📥 Extrae texto desde un archivo .docx
def extraer_texto_docx(ruta: str) -> str:
    doc = Document(ruta)
    return "\n".join([p.text for p in doc.paragraphs])

# 🧼 Limpia el texto extraído de errores comunes de OCR
def limpiar_texto_ocr(texto: str) -> str:
    texto = re.sub(r'-\n', '', texto)
    texto = re.sub(r'\n(?=\S)', ' ', texto)
    texto = re.sub(r'\n\s*\n', '\n\n', texto)
    return texto.strip()

# 📑 Divide el texto en secciones
def dividir_en_secciones(texto: str) -> List[str]:
    secciones = re.split(r'\n{2,}', texto)
    return [s.strip() for s in secciones if s.strip()]

# 📏 Divide secciones largas
def cortar_secciones_largas(secciones: List[str], max_palabras: int = 1000) -> List[str]:
    resultado = []
    for seccion in secciones:
        palabras = seccion.split()
        if len(palabras) <= max_palabras:
            resultado.append(seccion)
        else:
            for i in range(0, len(palabras), max_palabras):
                resultado.append(" ".join(palabras[i:i + max_palabras]))
    return resultado

# 🚫 Elimina boilerplate
def eliminar_boilerplate(chunks: List[str]) -> List[str]:
    frases = [
        "norma española", "Depósito legal:", "Editada e impresa",
        "LAS OBSERVACIONES A ESTE DOCUMENTO", "ÍNDICE", "Página",
        "Fecha de la inspección", "Firma del inspector",
        "UNE 23580", "Dirección", "Teléfono", "Fax"
    ]
    return [c for c in chunks if not any(f.lower() in c.lower() for f in frases)]

# 🌀 Elimina encabezados repetidos
def eliminar_encabezados_repetidos(chunks: List[str]) -> List[str]:
    vistos = set()
    resultado = []
    for chunk in chunks:
        primera_linea = chunk.split('\n')[0].strip().lower()
        if primera_linea.startswith("actas para la revisión de las instalaciones"):
            if primera_linea in vistos:
                continue
            vistos.add(primera_linea)
        resultado.append(chunk)
    return resultado

# 🔁 Elimina duplicados exactos
def deduplicar_chunks(chunks: List[str]) -> List[str]:
    vistos, resultado = set(), []
    for c in chunks:
        limpio = c.strip()
        if limpio not in vistos:
            vistos.add(limpio)
            resultado.append(limpio)
    return resultado

# 🧠 Procesa todos los .docx de una carpeta y guarda jsons en processed_chunks/
def procesar_nuevos_documentos():
    carpeta_nuevos = Path("nuevos_docs")
    carpeta_destino = Path("processed_chunks")
    carpeta_destino.mkdir(parents=True, exist_ok=True)

    for archivo in carpeta_nuevos.glob("*.docx"):
        try:
            texto = limpiar_texto_ocr(extraer_texto_docx(archivo))
            secciones = dividir_en_secciones(texto)
            secciones = eliminar_boilerplate(secciones)
            secciones = eliminar_encabezados_repetidos(secciones)
            chunks = deduplicar_chunks(cortar_secciones_largas(secciones))

            salida_json = {"filename": archivo.name, "chunks": chunks}
            destino = carpeta_destino / f"{archivo.stem}.json"

            with open(destino, 'w', encoding='utf-8') as f:
                json.dump(salida_json, f, indent=2, ensure_ascii=False)

            print(f"✅ Procesado: {archivo.name} → {destino.name}")
        except Exception as e:
            print(f"❌ Error procesando {archivo.name}: {e}")

if __name__ == "__main__":
    procesar_nuevos_documentos()

