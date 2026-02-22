from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import json

import chromadb
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


@dataclass
class RAGConfig:
    persist_dir: str = "data/vectordb"
    collection_name: str = "pharm_kb"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class PharmRAGStore:
    """
    Chroma-backed RAG store.

    Supports two styles of KB:

    A) "Old style" small KB JSONL:
        {"drug": "...", "lang": "en|uz", "text": "..."}
       -> query(drug, lang) uses where filter.

    B) "OpenFDA style" large KB JSONL:
        {"id": "...", "text": "...", "brand": "...", "generic": "...", ...}
       -> query_open(query) uses semantic search with no where filter.
    """

    def __init__(self, cfg: Optional[RAGConfig] = None):
        self.cfg = cfg or RAGConfig()
        Path(self.cfg.persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.cfg.persist_dir)
        self.col = self.client.get_or_create_collection(name=self.cfg.collection_name)
        self.embedder = SentenceTransformer(self.cfg.embed_model)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts, normalize_embeddings=True).tolist()

   
    # Indexing
    def index_old_jsonl(self, jsonl_path: str) -> None:
        """
        Index KB documents into Chroma (your original format).
        Each JSONL row must contain: drug, lang, text
        """
        p = Path(jsonl_path)
        if not p.exists():
            raise FileNotFoundError(f"KB not found: {jsonl_path}")

        docs, ids, metas = [], [], []
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                drug = str(item["drug"]).upper().strip()
                lang = str(item["lang"]).lower().strip()
                text = str(item["text"]).strip()

                doc_id = f"{drug}_{lang}_{i}"
                ids.append(doc_id)
                docs.append(text)
                metas.append({"drug": drug, "lang": lang, "source": "custom_kb"})

        embs = self._embed(docs)
        self.col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

    def index_openfda_chunks_jsonl(self, jsonl_path: str, batch_size: int = 256) -> None:
        """
        Index OpenFDA chunked JSONL into Chroma (recommended for your huge file).

        Expected fields per line:
          {
            "id": "...",
            "text": "...",
            "brand": "...",
            "generic": "...",
            "product_ndc": "...",
            "spl_id": "...",
            "source": "openfda_druglabel"
          }
        """
        p = Path(jsonl_path)
        if not p.exists():
            raise FileNotFoundError(f"KB not found: {jsonl_path}")

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, Any]] = []

        def flush():
            if not ids:
                return
            embs = self._embed(docs)
            self.col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
            ids.clear()
            docs.clear()
            metas.clear()

        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                item = json.loads(line)
                text = str(item.get("text", "")).strip()
                if not text:
                    continue

                doc_id = str(item.get("id", f"openfda:{i}"))
                ids.append(doc_id)
                docs.append(text)
                metas.append({
                    "source": item.get("source", "openfda_druglabel"),
                    "brand": item.get("brand", ""),
                    "generic": item.get("generic", ""),
                    "product_ndc": item.get("product_ndc", ""),
                    "spl_id": item.get("spl_id", ""),
                })

                if len(ids) >= batch_size:
                    flush()
                    if i % (batch_size * 10) == 0:
                        print(f"Indexed ~{i:,} lines...")

        flush()
        print("Indexing done ")

    # Querying
    
    def query(self, drug: str, lang: str, top_k: int = 3) -> list[dict]:
        """
        Your original query with metadata filter: where drug/lang match.
        """
        drug = drug.upper().strip()
        lang = lang.lower().strip()

        q = f"{drug} medication information uses cautions warnings dosage interactions side effects"
        q_emb = self._embed([q])[0]

        res = self.col.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where={"$and": [{"drug": drug}, {"lang": lang}]},
        )

        out = []
        for doc, meta, doc_id in zip(res["documents"][0], res["metadatas"][0], res["ids"][0]):
            out.append({"id": doc_id, "text": doc, "meta": meta})
        return out

    def query_open(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Open semantic query WITHOUT filters (best for OpenFDA).
        """
        q = (query or "").strip()
        if not q:
            return []

        q_emb = self._embed([q])[0]
        res = self.col.query(query_embeddings=[q_emb], n_results=top_k)

        out = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]  

        for doc, meta, doc_id, dist in zip(docs, metas, ids, dists):
            score = float(1.0 / (1.0 + dist)) if dist is not None else 0.0
            out.append({"id": doc_id, "text": doc, "meta": meta, "score": score})
        return out

    # Utility
    def best_drug_match(self, ocr_text: str, candidates: list[str]) -> tuple[str, float]:
        """
        Semantic match OCR text to the closest known drug name in candidates.
        Returns (best_name, similarity in [0,1]).
        """
        q = (ocr_text or "").upper().strip()
        if not q:
            return "", 0.0

        cand = [c.upper().strip() for c in candidates if c and c.strip()]
        if not cand:
            return "", 0.0

        q_emb = self.embedder.encode([q], normalize_embeddings=True)
        c_emb = self.embedder.encode(cand, normalize_embeddings=True)

        sims = cos_sim(q_emb, c_emb).cpu().numpy()[0]
        best_i = int(sims.argmax())
        return cand[best_i], float(sims[best_i])
