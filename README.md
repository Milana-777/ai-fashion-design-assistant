# AI Fashion Design Assistant

8-week Generative AI internship project at Aarivya Labs.
Built an end-to-end AI fashion design system using Stable Diffusion, ControlNet, LoRA, CLIP, ChromaDB, Gradio and FastAPI.

## Tech Stack
- PyTorch + HuggingFace Diffusers
- Stable Diffusion v1.5
- ControlNet (Canny + OpenPose)
- LoRA fine-tuning (PEFT)
- CLIP (ViT-B-32) + ChromaDB
- Gradio + FastAPI + Celery + Redis
- Docker + Kubernetes

## Setup
```bash
python -m venv fashion-ai-env
source fashion-ai-env/bin/activate
pip install torch diffusers transformers open-clip-torch datasets pillow pandas matplotlib gradio fastapi uvicorn chromadb peft
```

## Week-by-week progress

### Week 1 — Fashion Domain Research and Dataset Curation
- Studied the 6-stage fashion design workflow: inspiration → sketch → technical drawing → fabric selection → prototype → final garment
- Learned key fashion vocabulary: silhouette, croquis, tech pack, toile, drape, grading, colorway, neckline
- Downloaded and curated 1000 Fashionpedia images using HuggingFace streaming mode
- Built 10-attribute metadata schema: category, silhouette, neckline, sleeve, pattern, color tone, fabric weight, occasion, gender, season
- Auto-labelled all images using pixel analysis and saved as fashion_labels.csv
- Built PyTorch DataLoader pipeline with 80/10/10 train/val/test split
- Applied augmentation: random horizontal flip, color jitter, resize to 256×256
- Final output: batches of shape [8, 3, 256, 256] — PIPELINE TEST PASSED

### Week 2 — Text-to-Image Foundation with Stable Diffusion
- Set up Stable Diffusion v1.5 pipeline (CPU-optimised mode with attention slicing)
- Generated first 3 AI fashion images: elegant dress, streetwear jacket, floral summer top
- Ran 5 prompt engineering experiments comparing CFG scale (7.5 vs 12) and steps (20 vs 30)
- Built CLIP evaluation system using ViT-B-32 to score prompt-image alignment
- CLIP score results: bare prompt 18.2 → full structured prompt 24.9 → high CFG 25.8
- Built reusable prompt template library with 8 named templates and unlimited random generation
- Prompt anatomy: [SUBJECT] + [FABRIC] + [COLOR] + [SILHOUETTE] + [PHOTO STYLE] + [QUALITY TAGS]
- Generated library outputs: elegant_evening, casual_summer, streetwear_set designs

### Week 3 — Style Control with ControlNet
- Integrated Canny edge ControlNet with Stable Diffusion v1.5
- Built sketch-to-design pipeline (sketch → edge map → generated garment)
- Added OpenPose-based pose control for fashion model poses (standing, walking, hands on hips)
- Compared controlled vs uncontrolled generation using CLIP scores
- Result: average CLIP score improved from 23.1 to 26.1 (+3.0) with ControlNet

### Week 4 — LoRA Fine-Tuning for Brand Style
- Curated 3 brand-specific datasets: minimalist, streetwear, luxury
- Trained lightweight LoRA adapters (rank=2) optimized for CPU training
- Implemented prompt-token style switching (e.g. minimalist-style token)
- Evaluated brand consistency using CLIP similarity scoring
- Minimalist adapter achieved final training loss of 0.0835

### Week 5 — Intelligent Style Recommendation (RAG)
- Built fashion knowledge base with 8 trends, 6 style personas, 6 color palettes
- Embedded 200 fashion images using CLIP ViT-B-32 into 512-dimensional vectors
- Stored and queried embeddings using ChromaDB vector database
- Built RAG pipeline: text query → CLIP embedding → ChromaDB search → trend matching
- Processed 5 style queries with relevant image and trend recommendations

### Week 6 — Gradio Creative Studio
- Built 4-tab interactive Gradio creative studio
- Sketch2Design: upload hand-drawn sketch → AI generates fashion photo
- StyleMixer: select brand style + garment + color → styled design output
- WardrobeGen: choose occasion + season + persona → complete outfit generation
- Trend explorer: type style query → RAG retrieves trends + similar images
- Live demo runs locally at http://127.0.0.1:7860

### Week 7 — FastAPI Backend and Async Generation
- Built production REST API using FastAPI + Uvicorn
- Async task queue using Celery + Redis for heavy AI generation jobs
- NSFW prompt filtering, image watermarking, rate limiting (5 requests/minute)
- Job status polling with progress tracking (queued → processing → completed)
- API endpoints: /generate, /status/{job_id}, /result/{job_id}, /jobs, /styles
- Interactive Swagger UI at http://127.0.0.1:8000/docs

### Week 8 — Kubernetes Deployment and Portfolio
- Dockerized all services (API, Gradio, Celery worker, Redis)
- Kubernetes deployment YAML with resource limits (4GB RAM, 2 CPU per pod)
- Generated 20-image AI fashion lookbook with CLIP score analysis
- Complete project documentation: README, architecture diagram, API docs
- Final presentation outline: 10-slide deck covering all 8 weeks

## Results
- 20+ AI-generated fashion lookbook designs
- 3 LoRA brand style adapters trained (minimalist, streetwear, luxury)
- CLIP score improvement: uncontrolled 23.1 → ControlNet controlled 26.1
- Full REST API with rate limiting, NSFW filtering and watermarking
- Live Gradio creative studio with 4 functional tabs

## Author
Sreenivasulu Gari Milana — Generative AI Internship — Aarivya Labs 2026
