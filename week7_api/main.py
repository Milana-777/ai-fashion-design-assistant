from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from celery import Celery
from PIL import Image, ImageDraw, ImageFont
import torch, uuid, os, time, json
from collections import defaultdict
from datetime import datetime, timedelta
from diffusers import StableDiffusionPipeline

app = FastAPI(
    title="AI Fashion Design API",
    description="Production backend for AI fashion generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REDIS_URL  = "redis://localhost:6379/0"
OUTPUT_DIR = "week7_api/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

celery_app = Celery("fashion_tasks", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
)

# ── In-memory stores ──────────────────────────────────────────
job_store   = {}
rate_limits = defaultdict(list)

# ── NSFW filter ───────────────────────────────────────────────
NSFW_KEYWORDS = [
    "nude","naked","explicit","nsfw","adult","sexual",
    "pornographic","xxx","erotic","undressed",
]

def is_nsfw(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in NSFW_KEYWORDS)

# ── Rate limiter ──────────────────────────────────────────────
def check_rate_limit(client_ip: str, max_requests=5, window=60) -> bool:
    now = datetime.now()
    cutoff = now - timedelta(seconds=window)
    rate_limits[client_ip] = [
        t for t in rate_limits[client_ip] if t > cutoff
    ]
    if len(rate_limits[client_ip]) >= max_requests:
        return False
    rate_limits[client_ip].append(now)
    return True

# ── Watermarking ──────────────────────────────────────────────
def add_watermark(image: Image.Image, text="AI Fashion Studio") -> Image.Image:
    draw = ImageDraw.Draw(image)
    w, h = image.size
    draw.text(
        (w - 160, h - 24),
        text,
        fill=(200, 200, 200, 128),
    )
    return image

# ── Pydantic models ───────────────────────────────────────────
class GenerationRequest(BaseModel):
    prompt:          str
    negative_prompt: str  = "blurry, low quality, ugly"
    steps:           int  = 20
    cfg_scale:       float = 7.5
    width:           int  = 512
    height:          int  = 512
    style:           str  = "fashion photography"

class GenerationResponse(BaseModel):
    job_id:  str
    status:  str
    message: str

class StatusResponse(BaseModel):
    job_id:   str
    status:   str
    progress: int
    result:   str | None = None
    error:    str | None = None

# ── Celery task ───────────────────────────────────────────────
@celery_app.task(bind=True)
def generate_fashion_image(self, job_id: str, request_data: dict):
    try:
        job_store[job_id] = {"status": "processing", "progress": 10}

        prompt = (
            f"{request_data['prompt']}, "
            f"{request_data['style']}, "
            f"white background, sharp focus"
        )

        job_store[job_id]["progress"] = 30

        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            safety_checker=None,
        )
        pipe.enable_attention_slicing()

        job_store[job_id]["progress"] = 50

        image = pipe(
            prompt              = prompt,
            negative_prompt     = request_data["negative_prompt"],
            num_inference_steps = request_data["steps"],
            guidance_scale      = request_data["cfg_scale"],
            height              = request_data["height"],
            width               = request_data["width"],
        ).images[0]

        job_store[job_id]["progress"] = 80

        image = add_watermark(image)

        out_path = os.path.join(OUTPUT_DIR, f"{job_id}.png")
        image.save(out_path)

        job_store[job_id] = {
            "status":   "completed",
            "progress": 100,
            "result":   out_path,
        }
        return {"status": "completed", "path": out_path}

    except Exception as e:
        job_store[job_id] = {
            "status":   "failed",
            "progress": 0,
            "error":    str(e),
        }
        raise

# ── API endpoints ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name":    "AI Fashion Design API",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest, req: Request):
    client_ip = req.client.host

    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 5 requests per minute.",
        )

    if is_nsfw(request.prompt):
        raise HTTPException(
            status_code=400,
            detail="Prompt contains inappropriate content.",
        )

    if request.steps > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum steps allowed is 50.",
        )

    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "queued", "progress": 0}

    generate_fashion_image.apply_async(
        args=[job_id, request.model_dump()],
        task_id=job_id,
    )

    return GenerationResponse(
        job_id=job_id,
        status="queued",
        message=f"Generation started. Poll /status/{job_id} for updates.",
    )

@app.get("/status/{job_id}", response_model=StatusResponse)
def get_status(job_id: str):
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = job_store[job_id]
    return StatusResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        progress=job.get("progress", 0),
        result=job.get("result"),
        error=job.get("error"),
    )

@app.get("/result/{job_id}")
def get_result(job_id: str):
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = job_store[job_id]
    if job.get("status") != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete. Status: {job.get('status')}",
        )
    path = job.get("result")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Result file not found.")
    return FileResponse(path, media_type="image/png")

@app.get("/jobs")
def list_jobs():
    return {
        "total": len(job_store),
        "jobs": [
            {"job_id": k, "status": v.get("status"), "progress": v.get("progress")}
            for k, v in job_store.items()
        ],
    }

@app.get("/styles")
def list_styles():
    return {
        "styles": [
            "fashion photography",
            "vogue editorial",
            "studio lighting white background",
            "street style photography",
            "lookbook style",
            "runway fashion",
        ]
    }
