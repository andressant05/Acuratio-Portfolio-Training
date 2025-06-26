# Configuración de Máquinas Virtuales para LLaMA

Instrucciones detalladas para la configuración y uso de dos máquinas virtuales que ejecutan modelos LLaMA para tareas de generación y entrenamiento de datasets.

---

## Máquina Virtual `jonander_a100` — LLaMA 70B (Generación del Dataset)

1. Iniciar la Máquina Virtual
- Enciende la instancia `jonander_a100`.
- Abre una terminal SSH conectada a la VM.

2. Configurar nginx
'''
sudo nano /etc/nginx/sites-available/vllm  # CAMBIA LA IP
sudo systemctl restart nginx
docker restart my_vllm_container
docker logs my_vllm_container -f
'''

3. Iniciar Jupyter Lab
'''
sudo env "PATH=$PATH" jupyter lab --allow-root --port=9999
'''

4. Crear Túnel SSH desde tu Máquina Local
'''
gcloud compute ssh jonander-a100 --zone=us-central1-a -- -L 9999:localhost:9999
'''
- Accede a Jupyter en tu navegador: http://localhost:9999/lab

5. Descargar y Ejecutar LLaMA 70B con Docker
'''
docker run -d --runtime nvidia --gpus all \
  --name my_vllm_container \
  -v /home/jonanderjimenezz/.cache/huggingface/:/root/.cache/huggingface \
  -p 9000:8000 --ipc=host \
  --env "HUGGING_FACE_HUB_TOKEN=***REMOVED***" \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 4
'''

---

## Máquina Virtual `andres_vllm` — LLaMA 3B (Entrenamiento del Modelo)

1. Iniciar la Máquina Virtual
- Enciende la instancia `andres_vllm`.
- Abre una terminal SSH conectada a la VM.

2. Configurar nginx
'''
sudo nano /etc/nginx/sites-available/vllm  # CAMBIA LA IP
sudo systemctl restart nginx
docker restart my_vllm_container
'''

3. Iniciar Jupyter Lab
'''
sudo env "PATH=$PATH" jupyter lab --allow-root --port=8888
'''

4. Crear Túnel SSH desde tu Máquina Local
'''
gcloud compute ssh andres-vllm-2 --zone=europe-west1-b -- -L 8888:localhost:8888
'''
- Accede a Jupyter en tu navegador: http://localhost:8888

5. Descargar y Ejecutar LLaMA 3B con Docker
'''
docker run -d --runtime nvidia --gpus all \
  --name my_vllm_container \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=***REMOVED***" \
  -p 8000:8000 --ipc=host \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --max-model-len 100000
'''

---

## Notas Finales

- Asegúrate de reemplazar las direcciones IP correctamente en la configuración de nginx.
- Mantén seguro tu token de Hugging Face.
- Ambas VMs deben contar con soporte GPU (idealmente A100) y tener Docker instalado.
