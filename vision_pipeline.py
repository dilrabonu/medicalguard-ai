from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import re
from datetime import datetime

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from agents.config import Paths, RiskThresholds


# Model loading (reusable)
PATHS = Paths()
RISK = RiskThresholds()

assert PATHS.yolo_weights.exists(), f"YOLO weights not found: {PATHS.yolo_weights}"
assert PATHS.trocr_local_dir.exists(), f"TrOCR local dir not found: {PATHS.trocr_local_dir}"

detector = YOLO(str(PATHS.yolo_weights))

processor = TrOCRProcessor.from_pretrained(str(PATHS.trocr_local_dir))
trocr = VisionEncoderDecoderModel.from_pretrained(str(PATHS.trocr_local_dir))

device = "cuda" if torch.cuda.is_available() else "cpu"
trocr = trocr.to(device).eval()



# OCR Enhancement
def enhance_for_ocr(img: Image.Image) -> Image.Image:
    """
    Lightweight enhancement before TrOCR to improve readability:
    - contrast up
    - sharpness up
    - mild denoise
    """
    img = img.convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = ImageEnhance.Sharpness(img).enhance(1.8)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    return img



# Text normalization + quality
def normalize_text(s: str) -> str:
    s = str(s).upper().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^A-Z0-9 \-]", "", s)
    return s.strip()

def text_quality_features(s: str) -> Dict[str, float]:
    s_norm = normalize_text(s)
    n = len(s_norm)
    if n == 0:
        return {"quality_score": 0.0, "len_score": 0.0, "alpha_ratio": 0.0, "weird_char_penalty": 1.0}

    letters = sum(ch.isalpha() for ch in s_norm)
    digits = sum(ch.isdigit() for ch in s_norm)
    spaces = sum(ch == " " for ch in s_norm)
    hyphens = sum(ch == "-" for ch in s_norm)
    allowed = letters + digits + spaces + hyphens

    if n < 4:
        len_score = 0.2
    elif n < 6:
        len_score = 0.6
    elif n > 30:
        len_score = 0.3
    else:
        len_score = 1.0

    alpha_ratio = letters / max(1, n)
    weird_ratio = 1.0 - (allowed / max(1, n))
    weird_char_penalty = min(1.0, max(0.0, weird_ratio * 2.0))

    quality_score = 0.45 * len_score + 0.45 * alpha_ratio + 0.10 * (1.0 - weird_char_penalty)

    return {
        "quality_score": float(max(0.0, min(1.0, quality_score))),
        "len_score": float(len_score),
        "alpha_ratio": float(alpha_ratio),
        "weird_char_penalty": float(weird_char_penalty),
    }

def regex_drug_plausibility(text: str) -> float:
    """
    Simple plausibility check. Returns [0,1].
    """
    t = normalize_text(text)
    if not t:
        return 0.0
    if len(t) < 4 or len(t) > 30:
        return 0.3
    # looks like uppercase word(s) with optional hyphen
    if re.fullmatch(r"[A-Z][A-Z0-9 \-]{2,40}", t) is None:
        return 0.4
    alpha_ratio = sum(c.isalpha() for c in t) / max(1, len(t))
    return 1.0 if alpha_ratio >= 0.60 else 0.5


# Risk Decision
@dataclass
class RiskDecision:
    risk_score: float
    risk_level: str
    decision: str
    reasons: List[str]


def compute_risk_score(
    yolo_conf: float,
    trocr_conf: float,
    ocr_text: str,
    semantic_sim: float = 0.0,
    regex_score: float = 0.0,
) -> RiskDecision:
    """
    Improved transparent risk scoring using:
    - YOLO conf
    - TrOCR conf proxy
    - text quality
    - regex plausibility
    - semantic similarity (OCR -> closest known drug)
    """
    reasons: List[str] = []
    feats = text_quality_features(ocr_text)
    q = feats["quality_score"]

    # Convert to risks (higher = worse)
    det_risk = 1.0 - float(np.clip(yolo_conf, 0, 1))
    ocr_risk = 1.0 - float(np.clip(trocr_conf, 0, 1))
    text_risk = 1.0 - float(np.clip(q, 0, 1))
    sem_risk = 1.0 - float(np.clip(semantic_sim, 0, 1))
    reg_risk = 1.0 - float(np.clip(regex_score, 0, 1))

    # Weighted risk (still interpretable)
    risk_score = (
        0.35 * det_risk +
        0.25 * ocr_risk +
        0.20 * text_risk +
        0.10 * sem_risk +
        0.10 * reg_risk
    )
    risk_score = float(np.clip(risk_score, 0, 1))

    # Reasons (transparent)
    if yolo_conf < RISK.yolo_floor:
        reasons.append(f"Low detection confidence (YOLO={yolo_conf:.3f} < {RISK.yolo_floor}).")
    if trocr_conf < RISK.trocr_floor:
        reasons.append(f"Low OCR confidence proxy (TrOCR={trocr_conf:.3f} < {RISK.trocr_floor}).")
    if q < 0.60:
        reasons.append(f"Weak text quality (quality_score={q:.2f}).")
    if semantic_sim > 0 and semantic_sim < RISK.semantic_floor:
        reasons.append(f"Low semantic match to known drugs (sim={semantic_sim:.2f}).")
    if regex_score < RISK.regex_floor:
        reasons.append(f"Text format looks unusual for a drug name (regex_score={regex_score:.2f}).")
    if normalize_text(ocr_text) == "":
        reasons.append("Extracted text becomes empty after normalization.")

    # Map to decisions
    if risk_score < RISK.low_max:
        level, decision = "LOW", "OK"
    elif risk_score < RISK.med_max:
        level, decision = "MEDIUM", "REVIEW"
    else:
        level, decision = "HIGH", "REJECT"

    if not reasons:
        reasons.append("Signals look strong: detection + TrOCR + text plausibility are reliable.")

    return RiskDecision(risk_score=risk_score, risk_level=level, decision=decision, reasons=reasons)



# Vision + TrOCR core

def yolo_detect_and_crop(image_path: str, conf_thres: float = 0.25, pad: int = 6):
    orig = Image.open(image_path).convert("RGB")
    w, h = orig.size

    preds = detector.predict(source=image_path, conf=conf_thres, verbose=False)
    r = preds[0]

    if r.boxes is None or len(r.boxes) == 0:
        return None, None, 0.0, orig

    confs = r.boxes.conf.detach().cpu().numpy()
    best_i = int(np.argmax(confs))
    x1, y1, x2, y2 = r.boxes.xyxy[best_i].detach().cpu().numpy().astype(int).tolist()

    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

    crop = orig.crop((x1, y1, x2, y2))
    return crop, (x1, y1, x2, y2), float(confs[best_i]), orig


@torch.no_grad()
def trocr_read_with_confidence(crop_pil: Image.Image, max_new_tokens: int = 32):
    crop_pil = enhance_for_ocr(crop_pil)

    pixel_values = processor(images=crop_pil, return_tensors="pt").pixel_values.to(device)
    out = trocr.generate(
        pixel_values,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True
    )

    text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()

    if out.scores is None or len(out.scores) == 0:
        return text, 0.0

    step_max = []
    for s in out.scores:
        probs = torch.softmax(s[0], dim=-1)
        step_max.append(float(probs.max().cpu()))
    return text, float(np.mean(step_max))


def draw_bbox(orig: Image.Image, bbox):
    img = orig.copy()
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
    return img



# Report builder + save

def build_screening_report(result: Dict[str, Any], semantic_best: str = "", semantic_sim: float = 0.0) -> Dict[str, Any]:
    yolo_conf = float(result.get("yolo_conf", 0.0) or 0.0)
    trocr_conf = float(result.get("ocr_conf_proxy", 0.0) or 0.0)
    text_raw = str(result.get("ocr_text", "") or "")
    text_norm = normalize_text(text_raw)

    regex_score = regex_drug_plausibility(text_norm)
    decision = compute_risk_score(
        yolo_conf=yolo_conf,
        trocr_conf=trocr_conf,
        ocr_text=text_norm,
        semantic_sim=semantic_sim,
        regex_score=regex_score
    )

    report = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "image_path": result.get("image_path"),
        "bbox_xyxy": result.get("bbox_xyxy"),
        "signals": {
            "yolo_conf": yolo_conf,
            "trocr_conf_proxy": trocr_conf,
            "extracted_text_raw": text_raw,
            "extracted_text_normalized": text_norm,
            "regex_plausibility": float(regex_score),
            "semantic_best_match": semantic_best,
            "semantic_similarity": float(semantic_sim),
            "text_quality_features": text_quality_features(text_norm),
        },
        "risk_assessment": {
            "risk_score": decision.risk_score,
            "risk_level": decision.risk_level,
            "decision": decision.decision,
            "reasons": decision.reasons,
        },
        "safety_note": (
            "This tool is for screening only. It does not verify authenticity and is not medical advice. "
            "For exact use and dosing, consult a pharmacist/clinician."
        ),
        "recommended_next_steps": (
            ["Proceed with a manual visual check."] if decision.decision == "OK" else
            ["Re-take the photo with better lighting.", "Try another angle.", "Manual review by pharmacist."] if decision.decision == "REVIEW" else
            ["Do not rely on this result.", "Manual verification required.", "Consult a pharmacist/official source."]
        )
    }
    return report


def save_report_json(report: Dict[str, Any], filename: Optional[str] = None) -> Path:
    PATHS.reports_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"pharmscan_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

    out_path = PATHS.reports_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return out_path
