import os
import io
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl

import torch
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

DTYPE = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32
pipe = DiffusionPipeline.from_pretrained(
	MODEL_ID,
	token=HF_TOKEN,
	trust_remote_code=True,
	torch_dtype=DTYPE,
)
pipe = pipe.to(DEVICE)
try:
    pipe.enable_attention_slicing()
except Exception:
    pass
try:
    if DEVICE == "cuda":
        pipe.enable_model_cpu_offload()  # accelerate
except Exception:
    pass

# FastAPI
app = FastAPI(title="FLUX Kontext Image API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class GenerateReq(BaseModel):
    prompt: str
    image_url: Optional[HttpUrl] = None
    num_inference_steps: int = 28
    guidance_scale: Optional[float] = None

DEFAULT_IMG = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "dtype": str(DTYPE)}

@app.post("/generate")
def generate(req: GenerateReq):
    try:
        src = req.image_url or DEFAULT_IMG
        init_img = load_image(str(src)).convert("RGB")

        kwargs = dict(image=init_img, prompt=req.prompt, num_inference_steps=req.num_inference_steps)
        if req.guidance_scale is not None:
            kwargs["guidance_scale"] = req.guidance_scale

        with torch.no_grad():
            out = pipe(**kwargs).images[0]

        buf = io.BytesIO()
        out.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

