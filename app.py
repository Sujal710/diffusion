"""
app.py
------
FastAPI server that generates images with Stable Diffusion 1.5 + the dog LoRA
trained by finetune_sd_lora.py (checkpoint-200 by default).

If S3_BUCKET is set, every generated image (+ a JSON file with its settings) and
the LoRA checkpoint used are stored in that S3 bucket:
    s3://<bucket>/models/dog-lora/<checkpoint>/pytorch_lora_weights.safetensors
    s3://<bucket>/generated/<YYYY-MM-DD>/<id>.png
    s3://<bucket>/generated/<YYYY-MM-DD>/<id>.json
AWS credentials come from the usual places (aws configure / env vars).

Setup (inside the venv):
    pip install fastapi uvicorn boto3

Run:
    set S3_BUCKET=my-bucket-name
    uvicorn app:app --host 127.0.0.1 --port 8000
    # use a different checkpoint:   set LORA_CHECKPOINT=checkpoint-400   (then run uvicorn)

Try it:
    open http://127.0.0.1:8000/docs  ->  POST /generate  ->  "Try it out"
    or: curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json"
              -d "{\"prompt\": \"a photo of sks dog on a beach\"}" --output out.png
"""

import io
import json
import os
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import boto3
import torch
from botocore.exceptions import BotoCoreError, ClientError
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

# ---------------------------------------------------------------- settings
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
ROOT = Path(__file__).resolve().parent
CHECKPOINT = os.environ.get("LORA_CHECKPOINT", "checkpoint-200")
LORA_FILE = ROOT / "output" / "dog-lora" / CHECKPOINT / "pytorch_lora_weights.safetensors"
S3_BUCKET = os.environ.get("S3_BUCKET")              # unset = don't store anything in S3
MODEL_KEY = f"models/dog-lora/{CHECKPOINT}/{LORA_FILE.name}"

pipe = None
s3 = boto3.client("s3") if S3_BUCKET else None
gpu_lock = threading.Lock()     # one image at a time - a 6 GB GPU can't run two


# ---------------------------------------------------------------- S3
def upload_model():
    """Upload the LoRA checkpoint once (skipped if it's already in the bucket)."""
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=MODEL_KEY)
        print(f"[s3] model already in s3://{S3_BUCKET}/{MODEL_KEY}")
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchKey"):
            raise
        s3.upload_file(str(LORA_FILE), S3_BUCKET, MODEL_KEY)
        print(f"[s3] uploaded model to s3://{S3_BUCKET}/{MODEL_KEY}")


def upload_image(png: bytes, info: dict) -> str:
    """Store the image and a JSON file with its settings; return the image key."""
    now = datetime.now(timezone.utc)
    base = f"generated/{now:%Y-%m-%d}/{now:%H%M%S}_{uuid.uuid4().hex[:8]}"
    info = {**info, "created_at": now.isoformat(), "model_s3_key": MODEL_KEY}
    s3.put_object(Bucket=S3_BUCKET, Key=f"{base}.png", Body=png, ContentType="image/png")
    s3.put_object(Bucket=S3_BUCKET, Key=f"{base}.json", Body=json.dumps(info, indent=2).encode(),
                  ContentType="application/json")
    return f"{base}.png"


# ---------------------------------------------------------------- model
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found.")
    if not LORA_FILE.exists():
        raise RuntimeError(f"LoRA not found: {LORA_FILE}")
    if s3:
        upload_model()          # fails fast here if the bucket name or credentials are wrong
    else:
        print("[s3] S3_BUCKET not set - images will not be stored")

    print(f"[load] {BASE_MODEL} + {LORA_FILE}")
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(str(LORA_FILE.parent), weight_name=LORA_FILE.name)
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    print("[load] ready")
    yield
    pipe = None
    torch.cuda.empty_cache()


app = FastAPI(title="sks dog generator", lifespan=lifespan)


# ---------------------------------------------------------------- API
class GenerateRequest(BaseModel):
    prompt: str = Field("a photo of sks dog", description='use "sks dog" to get the trained dog')
    negative_prompt: str = "blurry, low quality, deformed"
    steps: int = Field(25, ge=1, le=100)
    guidance_scale: float = Field(7.5, ge=0, le=20)
    lora_scale: float = Field(1.0, ge=0, le=2, description="0 = base model, 1 = full LoRA")
    seed: int | None = Field(None, description="same seed + same settings = same image")
    width: int = Field(512, ge=256, le=768, multiple_of=8)
    height: int = Field(512, ge=256, le=768, multiple_of=8)


@app.get("/health")
def health():
    return {"status": "ok" if pipe is not None else "loading", "checkpoint": CHECKPOINT,
            "s3_bucket": S3_BUCKET}


@app.post("/generate", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
def generate(req: GenerateRequest):
    # plain `def` (not async): FastAPI runs it in a worker thread, so the server stays responsive
    if pipe is None:
        raise HTTPException(503, "model is still loading")

    seed = req.seed if req.seed is not None else torch.seed() % 2**32
    with gpu_lock:
        try:
            g = torch.Generator("cuda").manual_seed(seed)
            image = pipe(
                req.prompt,
                negative_prompt=req.negative_prompt,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                width=req.width,
                height=req.height,
                generator=g,
                cross_attention_kwargs={"scale": req.lora_scale},
            ).images[0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(507, "GPU out of memory - try 512x512")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    png = buf.getvalue()
    headers = {"X-Seed": str(seed)}

    if s3:
        # an S3 problem shouldn't throw away an image that took a while to make
        try:
            key = upload_image(png, {**req.model_dump(), "seed": seed, "checkpoint": CHECKPOINT})
            headers["X-S3-Key"] = key
            headers["X-S3-URL"] = s3.generate_presigned_url(
                "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=3600)
        except (BotoCoreError, ClientError) as e:
            print(f"[s3] upload failed: {e}")
            headers["X-S3-Error"] = type(e).__name__

    return Response(png, media_type="image/png", headers=headers)
