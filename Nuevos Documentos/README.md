# Processed Chunks — Añadir Nuevos Documentos DOCX

Este script convierte nuevos archivos `.docx` en archivos `.json` estructurados y los añade automáticamente a la carpeta `processed_chunks/`, la cual contiene los documentos que ya han sido procesados por el sistema.

---

## Estructura esperada

```
.
├── processed_chunks/         # Contiene todos los archivos JSON ya procesados
├── nuevos_docs/            # Aquí debes colocar los nuevos documentos .docx
├── parse_docx_to_chunks.py # Script que transforma los .docx en .json
```

---

## ¿Cómo usarlo?

1. Asegúrate de tener descargada la carpeta `processed_chunks/` desde este repositorio.
2. Coloca tus nuevos documentos `.docx` en la carpeta `nuevos_docs/`.
3. Ejecuta el siguiente comando:

```
python parse_docx_to_chunks.py
```

Esto generará automáticamente un archivo `.json` por cada `.docx`, en la carpeta `processed_chunks/`.

---

## Notas importantes

- No necesitas modificar manualmente nada en `processed_chunks/`.
- Si un archivo `.json` con el mismo nombre ya existe, será **reemplazado**.
- Este script **no requiere conexión a internet** ni ningún entorno especial.
- Este paso es previo a ejecutar los siguientes scripts del pipeline:

```
python generate_dataset_pipeline.py
python train_pipeline.py
```

---

## ¿Qué hace internamente?

- Limpia el texto de errores comunes de OCR.
- Elimina texto boilerplate legal/técnico.
- Divide el contenido en fragmentos de hasta 1000 palabras.
- Guarda los fragmentos en archivos `.json` compatibles con el pipeline de generación de dataset y entrenamiento.

---

## Resultado

Convertir nuevos `.docx` y añadirlos al sistema es ahora un proceso de 1 comando. Rápido, sencillo y sin riesgo de interferir con el trabajo ya hecho.
