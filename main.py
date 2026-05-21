from dotenv import load_dotenv
load_dotenv()

import uuid
from typing import Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from services.prompt_builder import build_image_prompt
from services.image_generator import generate_image

app = FastAPI(title="IA Calypso - Image Generator", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

previews: dict[str, dict] = {}


class ImageRequest(BaseModel):
    context: dict[str, Any]


class ImageResponse(BaseModel):
    image_base64: str
    prompt_used: str
    preview_url: str
    format: str = "png"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-image", response_model=ImageResponse)
async def generate_image_endpoint(request: ImageRequest, req: Request):
    prompt = await build_image_prompt(request.context)
    image_b64 = await generate_image(prompt)

    preview_id = str(uuid.uuid4())
    previews[preview_id] = {"image_b64": image_b64, "prompt": prompt}

    base_url = str(req.base_url).rstrip("/")
    preview_url = f"{base_url}/preview/{preview_id}"

    return ImageResponse(
        image_base64=image_b64,
        prompt_used=prompt,
        preview_url=preview_url,
    )


@app.get("/preview/{preview_id}", response_class=HTMLResponse)
def get_preview(preview_id: str):
    data = previews.get(preview_id)
    if not data:
        raise HTTPException(status_code=404, detail="Preview not found or expired")
    return HTMLResponse(_build_html(data["image_b64"], data["prompt"]))


def _build_html(image_b64: str, prompt: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>IA Calypso — Resultado</title>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ background:#0f0f0f; color:#fff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
            min-height:100vh; display:flex; flex-direction:column; align-items:center; padding:48px 20px; }}
    .badge {{ font-size:11px; letter-spacing:3px; text-transform:uppercase; color:#ff6b35; margin-bottom:16px; }}
    h1 {{ font-size:28px; font-weight:700; margin-bottom:6px; }}
    .sub {{ color:#555; font-size:13px; margin-bottom:40px; }}
    .card {{ background:#1a1a1a; border-radius:20px; overflow:hidden; max-width:580px; width:100%;
             box-shadow:0 40px 80px rgba(0,0,0,0.6); }}
    .card img {{ width:100%; display:block; }}
    .card-body {{ padding:28px; }}
    .label {{ font-size:10px; letter-spacing:2px; text-transform:uppercase; color:#555; margin-bottom:8px; }}
    .prompt {{ color:#aaa; font-size:13px; line-height:1.7; background:#111; padding:16px; border-radius:10px;
               border-left:3px solid #ff6b35; word-break:break-word; }}
    .actions {{ margin-top:24px; display:flex; gap:12px; }}
    a.btn {{ display:inline-block; padding:10px 20px; border-radius:8px; font-size:13px; font-weight:600; text-decoration:none; }}
    a.btn-primary {{ background:#ff6b35; color:#fff; }}
    a.btn-secondary {{ background:#222; color:#aaa; border:1px solid #333; }}
    a.btn:hover {{ opacity:0.85; }}
    .meta {{ display:flex; gap:24px; margin-top:20px; padding-top:20px; border-top:1px solid #222; }}
    .meta-item label {{ font-size:10px; letter-spacing:1px; text-transform:uppercase; color:#444; display:block; margin-bottom:4px; }}
    .meta-item span {{ font-size:13px; font-weight:600; color:#ff6b35; }}
  </style>
</head>
<body>
  <div class="badge">IA Calypso · Image Generator</div>
  <h1>Imagen generada</h1>
  <p class="sub">gpt-image-1 · medium quality · 1024×1024</p>
  <div class="card">
    <img src="data:image/png;base64,{image_b64}" alt="Imagen generada por IA">
    <div class="card-body">
      <div class="label">Prompt utilizado</div>
      <div class="prompt">{prompt}</div>
      <div class="actions">
        <a class="btn btn-primary" href="data:image/png;base64,{image_b64}" download="calypso.png">Descargar imagen</a>
        <a class="btn btn-secondary" href="/docs" target="_blank">API Docs</a>
      </div>
      <div class="meta">
        <div class="meta-item"><label>Modelo</label><span>gpt-image-1</span></div>
        <div class="meta-item"><label>Calidad</label><span>Medium</span></div>
        <div class="meta-item"><label>Resolución</label><span>1024×1024</span></div>
      </div>
    </div>
  </div>
</body>
</html>"""
