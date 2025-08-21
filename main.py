import os
import io
from typing import Optional
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

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

@app.get("/validate_token")
def validate_token():
    try:
        if not HF_TOKEN:
            return {"valid": False, "message": "HF_TOKEN이 설정되지 않았습니다"}
        
        # 토큰 정보 출력 (보안상 일부만)
        token_preview = HF_TOKEN[:10] + "..." if len(HF_TOKEN) > 10 else HF_TOKEN
        print(f"Token preview: {token_preview}")
        print(f"Token length: {len(HF_TOKEN)}")
        print(f"Token starts with 'hf_': {HF_TOKEN.startswith('hf_')}")
        
        # Hugging Face API로 토큰 유효성 검증
        import requests
        
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        print(f"Request headers: {headers}")
        
        response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text[:200]}...")
        
        if response.status_code == 200:
            user_info = response.json()
            return {
                "valid": True, 
                "message": "토큰이 유효합니다",
                "user": user_info.get("name", "unknown"),
                "email": user_info.get("email", "unknown")
            }
        else:
            return {"valid": False, "message": f"토큰이 유효하지 않습니다. 상태 코드: {response.status_code}", "response": response.text}
            
    except Exception as e:
        return {"valid": False, "message": f"토큰 검증 중 오류 발생: {str(e)}"}

@app.post("/generate")
def generate(req: GenerateReq):
    try:
        src = req.image_url or DEFAULT_IMG
        print(f"Loading image from: {src}")
        init_img = load_image(str(src)).convert("RGB")
        print(f"Input image size: {init_img.size}, mode: {init_img.mode}")

        kwargs = dict(image=init_img, prompt=req.prompt, num_inference_steps=req.num_inference_steps)
        if req.guidance_scale is not None:
            kwargs["guidance_scale"] = req.guidance_scale
        print('kwargs:', kwargs)
        
        with torch.no_grad():
            print('Starting inference...')
            result = pipe(**kwargs)
            print(f"Pipeline result type: {type(result)}")
            print(f"Pipeline result attributes: {dir(result)}")
            
            if hasattr(result, 'images'):
                print(f"Images length: {len(result.images)}")
                out = result.images[0]
                print(f"Output image size: {out.size}, mode: {out.mode}")
                
                # FLUX 모델 출력을 안전하게 처리
                import numpy as np
                
                # 원본 이미지 데이터 보존
                if hasattr(out, 'numpy'):
                    img_array = out.numpy()
                else:
                    img_array = np.array(out)
                
                print(f"Original image array - shape: {img_array.shape}, dtype: {img_array.dtype}")
                print(f"Original image array - min: {img_array.min()}, max: {img_array.max()}")
                
                # 데이터 타입에 따른 처리
                if img_array.dtype in [np.float32, np.float64]:
                    print("Processing float image data...")
                    
                    # NaN, 무한대 값 처리
                    if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
                        print("NaN or infinite values detected, cleaning...")
                        img_array = np.nan_to_num(img_array, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    # 값 범위 확인 및 정규화
                    if img_array.max() > 1.0:
                        print("Values > 1.0 detected, normalizing to 0-1 range")
                        img_array = img_array / img_array.max()
                    elif img_array.min() < 0.0:
                        print("Negative values detected, shifting to 0-1 range")
                        img_array = img_array - img_array.min()
                        if img_array.max() > 0:
                            img_array = img_array / img_array.max()
                    
                    # 0-255 범위로 변환
                    img_array = (img_array * 255).astype(np.uint8)
                    
                elif img_array.dtype == np.uint8:
                    print("Processing uint8 image data...")
                    # 이미 올바른 범위
                    pass
                else:
                    print(f"Unexpected dtype: {img_array.dtype}, converting to uint8")
                    img_array = img_array.astype(np.uint8)
                
                print(f"Final image array - min: {img_array.min()}, max: {img_array.max()}, dtype: {img_array.dtype}")
                
                # numpy 배열을 PIL Image로 변환
                out = Image.fromarray(img_array, mode='RGB')
                print(f"Processed image size: {out.size}, mode: {out.mode}")
            else:
                print("No images attribute found in result")
                raise Exception("Pipeline did not return images")
        
        print('Saving image to buffer...')
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        buf.seek(0)
        print(f"Buffer size: {len(buf.getvalue())} bytes")
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

