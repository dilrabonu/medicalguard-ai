from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import json
from pathlib import Path

from agents.config import Paths, AgentDefaults
from agents.safety_rules import safety_guard
from agents.prompts import format_answer
from agents.rag_store import PharmRAGStore, RAGConfig
from agents.vision_pipeline import (
    yolo_detect_and_crop,
    trocr_read_with_confidence,
    draw_bbox,
    build_screening_report,
    save_report_json,
    normalize_text,
)

PATHS = Paths()
DEFAULTS = AgentDefaults()


@dataclass
class AgentConfig:
    lang: str = DEFAULTS.lang
    top_k: int = DEFAULTS.top_k
    yolo_conf_thres: float = DEFAULTS.yolo_conf_thres


def load_kb_drug_list(kb_jsonl: str) -> List[str]:
    p = Path(kb_jsonl)
    assert p.exists(), f"KB not found: {kb_jsonl}"

    drugs = set()
    with open(p, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                drugs.add(item["drug"].upper().strip())
            except Exception:
                continue
    if not drugs:
        raise ValueError(f"No valid drug entries found in KB: {kb_jsonl}")
    return sorted(drugs)


class PharmScanAgent:
    def __init__(self, cfg: AgentConfig = AgentConfig()):
        self.cfg = cfg
        self.rag = PharmRAGStore(RAGConfig(persist_dir=str(PATHS.vectordb_dir)))
        self.drug_candidates = load_kb_drug_list(str(PATHS.kb_jsonl))

    def ensure_kb_indexed(self) -> None:
        """
        Index KB into vector store (run once).
        """
        self.rag.index_jsonl(str(PATHS.kb_jsonl))

    def answer_text_query(self, drug_name: str, lang: str = "en", user_query: str = "") -> Dict[str, Any]:
        guard = safety_guard(user_query or drug_name)
        if not guard.allowed:
            return {"status": "SAFE_BLOCK", "message": guard.message}

        drug = normalize_text(drug_name)
        # semantic match to closest known drug (for typos)
        best, sim = self.rag.best_drug_match(drug, self.drug_candidates)
        final_drug = best if sim >= 0.55 else drug

        chunks = self.rag.query(drug=final_drug, lang=lang, top_k=self.cfg.top_k)
        answer = format_answer(drug=final_drug, lang=lang, context_chunks=chunks)
        return {
            "status": "OK",
            "drug": final_drug,
            "lang": lang,
            "semantic_best": best,
            "semantic_sim": sim,
            "answer": answer,
            "sources": chunks
        }

    def run_on_image(self, image_path: str, lang: str = "en", conf_thres: float | None = None) -> Dict[str, Any]:
        if conf_thres is None:
            conf_thres = self.cfg.yolo_conf_thres

        crop, bbox, yolo_conf, orig = yolo_detect_and_crop(image_path, conf_thres=conf_thres)
        if crop is None:
            return {"status": "NO_DETECTION", "message": "No drug-name region detected. Try lower threshold or better image."}

        ocr_text, trocr_conf = trocr_read_with_confidence(crop)
        ocr_norm = normalize_text(ocr_text)

        # semantic match OCR output to known drugs
        best, sim = self.rag.best_drug_match(ocr_norm, self.drug_candidates)
        final_drug = best if sim >= 0.55 else ocr_norm

        result = {
            "image_path": image_path,
            "bbox_xyxy": bbox,
            "yolo_conf": yolo_conf,
            "ocr_text": ocr_text,
            "ocr_conf_proxy": trocr_conf
        }

        report = build_screening_report(result, semantic_best=best, semantic_sim=sim)
        report_path = save_report_json(report)

        decision_block = (
            f"### Decision\n"
            f"- Risk: **{report['risk_assessment']['risk_level']}** ({report['risk_assessment']['risk_score']:.2f})\n"
            f"- Decision: **{report['risk_assessment']['decision']}**"
        )

        chunks = self.rag.query(drug=final_drug, lang=lang, top_k=self.cfg.top_k)
        answer = format_answer(drug=final_drug, lang=lang, context_chunks=chunks, decision_block=decision_block)

        return {
            "status": "OK",
            "drug": final_drug,
            "semantic_best": best,
            "semantic_sim": sim,
            "report": report,
            "report_path": str(report_path),
            "answer": answer,
            "viz_image": draw_bbox(orig, bbox),
            "crop_image": crop,
            "sources": chunks,
            "signals": {
                "yolo_conf": yolo_conf,
                "trocr_conf_proxy": trocr_conf,
                "ocr_text": ocr_text
            }
        }
