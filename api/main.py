from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional, Dict, Any
import sys

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.pharm_scan_agent import PharmScanAgent
from agents.rag_store import PharmRAGStore, RAGConfig


app = FastAPI(
    title="PharmScan AI API",
    version="1.0.0",
    description="YOLO (drug_name) + TrOCR + Risk Scoring + RAG (EN/UZ) API. Screening tool, not medical advice."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons (load once)

agent: Optional[PharmScanAgent] = None
rag: Optional[PharmRAGStore] = None


def get_agent() -> PharmScanAgent:
    global agent
    if agent is None:
        agent = PharmScanAgent()
        agent.ensure_kb_indexed()
    return agent


def get_rag() -> PharmRAGStore:
    """
    This points to YOUR persisted Chroma DB.
    Make sure persist_dir matches where you indexed: artifacts/vectordb
    and collection_name matches your index collection name.
    """
    global rag
    if rag is None:
        cfg = RAGConfig(
            persist_dir="data/vectordb",
            collection_name="pharm_kb",
            embed_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        rag = PharmRAGStore(cfg)
    return rag


# Helpers

UPLOADS_DIR = Path("artifacts/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# Routes

@app.get("/")
def root():
    return {"name": "PharmScan AI API", "status": "ok", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "PharmScan AI API"}


#  RAG endpoint (query vector DB directly)
class RagAskRequest(BaseModel):
    query: str
    k: int = 5

@app.post("/rag/ask")
def rag_ask(req: RagAskRequest):
    store = get_rag()
    hits = store.query_open(req.query, top_k=req.k)
    return {"query": req.query, "k": req.k, "hits": hits}


@app.post("/scan-image")
async def scan_image(
    file: UploadFile = File(...),
    lang: str = Form("en"),
    conf_thres: float = Form(0.25),
    return_images: bool = Form(True),
) -> JSONResponse:
    """
    Upload a medicine photo -> detect drug-name region -> TrOCR -> risk decision -> RAG answer.
    Returns structured JSON + optional base64 images
    """
    if lang not in {"en", "uz"}:
        raise HTTPException(status_code=400, detail="Language must be 'en' or 'uz'")

    if not (0.01 <= conf_thres <= 0.9):
        raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.01 and 0.9")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:  
        raise HTTPException(status_code=400, detail="Unsupported format: use .jpg/.jpeg/.png/.webp")

    content = await file.read()
    save_path = UPLOADS_DIR / file.filename
    save_path.write_bytes(content)

    a = get_agent()
    out = a.run_on_image(str(save_path), lang=lang, conf_thres=conf_thres)

    if out.get("status") != "OK":
        return JSONResponse(
            status_code=200,
            content={
                "status": "FAIL",
                "message": out.get("message", "Failed to process image."),
                "input_path": str(save_path),
            },
        )

    resp = {
        "status": "OK",
        "input_path": str(save_path),
        "drug": out.get("drug"),  
        "answer": out.get("answer"),
        "report_path": out.get("report_path"),
        "report": out.get("report"),
        "signals": out.get("signals", {}),
        "sources": out.get("sources", []),
    }

    if return_images:
        viz = out.get("viz_image")
        crop = out.get("crop_image")

        viz_img = Image.open(viz).convert("RGB") if isinstance(viz, (str, Path)) else viz
        crop_img = Image.open(crop).convert("RGB") if isinstance(crop, (str, Path)) else crop

        resp["images"] = {
            "bbox_png_base64": pil_to_base64_png(viz_img),
            "crop_png_base64": pil_to_base64_png(crop_img),
        }

    return JSONResponse(content=resp)


@app.get("/artifact")
def get_artifact(path: str) -> FileResponse:
    """
    Download any artifact report JSON, images) by a relative path.
    Example:
      /artifact?path=artifacts/reports/mediguard_report_...json
    """
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(str(p))
