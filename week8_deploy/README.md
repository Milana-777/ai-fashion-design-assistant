# AI Fashion Design Assistant

An end-to-end generative AI system for fashion design, combining
Stable Diffusion, ControlNet, LoRA fine-tuning, CLIP embeddings,
ChromaDB retrieval, Gradio UI, and FastAPI backend.

## Architecture
## Tech stack

| Component | Technology |
|---|---|
| Image generation | Stable Diffusion v1.5 |
| Sketch control | ControlNet (canny edges) |
| Brand style | LoRA fine-tuned adapters |
| Style search | CLIP + ChromaDB RAG |
| UI | Gradio 4-tab creative studio |
| API | FastAPI + Pydantic |
| Async tasks | Celery + Redis |
| Deployment | Docker + Kubernetes |

## Setup

```bash
git clone <your-repo>
cd fashion-ai-assistant
python -m venv fashion-ai-env
source fashion-ai-env/bin/activate
pip install -r week8_deploy/requirements.txt
```

## Run locally

```bash
# Start API
cd week7_api && uvicorn main:app --port 8000

# Start Gradio studio
python week6_gradio/sketch2design_fix.py

# Start Celery worker (separate terminal)
celery -A week7_api.main.celery_app worker --loglevel=info
```

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Health check |
| POST | /generate | Submit generation job |
| GET | /status/{job_id} | Poll job status |
| GET | /result/{job_id} | Download result image |
| GET | /jobs | List all jobs |
| GET | /styles | Available style options |

## Deploy with Kubernetes

```bash
docker build -t fashion-ai:latest .
kubectl apply -f week8_deploy/kubernetes.yaml
kubectl get services
```

## Week-by-week build log

| Week | What was built |
|---|---|
| 1 | Dataset pipeline (Fashionpedia, 1000 images, PyTorch DataLoader) |
| 2 | SD v1.5 pipeline, prompt engineering, CLIP evaluation |
| 3 | ControlNet sketch-to-design, pose control |
| 4 | LoRA brand adapters (minimalist, streetwear, luxury) |
| 5 | CLIP embeddings + ChromaDB RAG recommendation engine |
| 6 | 4-tab Gradio creative studio |
| 7 | FastAPI + Celery async generation backend |
| 8 | Docker, Kubernetes, lookbook, documentation |

## Results

- 20+ AI-generated fashion designs
- 3 LoRA brand style adapters trained
- CLIP score improvement: uncontrolled 23.1 → controlled 26.1
- Full REST API with rate limiting, NSFW filtering, watermarking
- Live Gradio demo at http://localhost:8090

## Author

Internship project — Aarivya Labs Generative AI Program
