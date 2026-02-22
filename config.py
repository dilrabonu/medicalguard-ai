from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    # Models
    yolo_weights: Path = Path("runs/drugname_yolo4/weights/best.pt")
    trocr_local_dir: Path = Path("artifacts/trocr_local")

    # Knowledge Base
    kb_jsonl: Path = Path("data/kb/drugs_en_uz.jsonl")

    # Artifacts
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("artifacts/reports")
    examples_dir: Path = Path("artifacts/examples")
    vectordb_dir: Path = Path("artifacts/vectordb")


@dataclass(frozen=True)
class AgentDefaults:
    lang: str = "en"           # "en" or "uz"
    top_k: int = 3
    yolo_conf_thres: float = 0.25


@dataclass(frozen=True)
class RiskThresholds:
    # Decision thresholds on risk_score
    low_max: float = 0.35
    med_max: float = 0.65

    # Floors used for reason messages
    yolo_floor: float = 0.25
    trocr_floor: float = 0.60
    semantic_floor: float = 0.55
    regex_floor: float = 0.60
