import os
import io
import re
from typing import List, Optional, Dict, Any

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ====== ENV ======
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
BUCKET = os.environ.get("STORAGE_BUCKET", "documents")
MODEL_NAME = os.environ.get("EMBED_MODEL", "intfloat/e5-small-v2")  # 384-dim

# ====== Clients & Model ======
sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
model = SentenceTransformer(MODEL_NAME)

# ====== Chunker (versi clean) ======
from chunker import ImprovedSemanticChunker, ChunkerConfig

app = FastAPI()


# ===================== Schemas =====================
class ProcessReq(BaseModel):
    document_id: str
    target_chunk_size: Optional[int] = 500
    min_chunk_size: Optional[int] = 200
    max_chunk_size: Optional[int] = 800
    overlap_ratio: Optional[float] = 0.1

class EmbedReq(BaseModel):
    text: str
    mode: str | None = "query"  # "query" / "passage"

class SearchReq(BaseModel):
    question: str
    top_k: int = 6
    filter_document_id: Optional[str] = None


# ===================== Utils =====================
@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}

def pdf_to_text(data: bytes) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(data))
    pages = len(reader.pages)
    texts: List[str] = []
    for i in range(pages):
        texts.append(reader.pages[i].extract_text() or "")
    return ("\n\n".join(texts).strip(), pages)

def embed_passages(texts: List[str]) -> List[List[float]]:
    texts = [f"passage: {t or ''}" for t in texts]  # e5 needs prefix
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return vecs.astype(np.float32).tolist()

def _build_chunker(req: ProcessReq) -> ImprovedSemanticChunker:
    cfg = ChunkerConfig(
        target_chars=req.target_chunk_size or 500,
        min_chars=req.min_chunk_size or 200,
        max_chars=req.max_chunk_size or 800,
        overlap_ratio=req.overlap_ratio or 0.10,
        use_spacy=True,
    )
    return ImprovedSemanticChunker(cfg)

def _keyword_boost(cands: List[Dict[str, Any]], question: str, boost: float = 0.15):
    qwords = set(re.findall(r"\b\w+\b", question.lower()))
    if not qwords:
        return cands
    for c in cands:
        words = set(re.findall(r"\b\w+\b", (c.get("content") or "").lower()))
        k = len(qwords & words)
        ratio = k / max(1, len(qwords))
        c["keyword_score"] = ratio * boost
        c["score"] = c.get("similarity", 0.0) + c["keyword_score"]
    cands.sort(key=lambda x: x.get("score", x.get("similarity", 0.0)), reverse=True)
    return cands

def _dedup_jaccard(cands: List[Dict[str, Any]], thr: float = 0.75):
    out: List[Dict[str, Any]] = []
    for cand in cands:
        cw = set((cand.get("content") or "").lower().split())
        if not cw:
            continue
        unique = True
        for ex in out:
            ew = set((ex.get("content") or "").lower().split())
            inter = len(cw & ew); uni = len(cw | ew)
            if uni and inter / uni > thr:
                unique = False; break
        if unique:
            out.append(cand)
    return out


# ===================== M3: Process Document =====================
@app.post("/process/document")
def process_document(req: ProcessReq):
    # 1) get document
    doc = sb.table("documents").select("*").eq("id", req.document_id).single().execute().data
    if not doc:
        return {"ok": False, "error": "document not found", "id": req.document_id}

    path = doc.get("storage_path") or doc.get("file_path")
    if not path:
        return {"ok": False, "error": "missing storage_path"}

    # 2) download
    try:
        file_bytes = sb.storage.from_(BUCKET).download(path)
        if not isinstance(file_bytes, (bytes, bytearray)):
            file_bytes = bytes(file_bytes)
    except Exception as e:
        return {"ok": False, "error": f"download failed: {e}"}

    # 3) extract
    try:
        text, pages = pdf_to_text(file_bytes)
    except Exception as e:
        sb.table("documents").update({"status": "error"}).eq("id", req.document_id).execute()
        return {"ok": False, "error": f"pdf parse failed: {e}"}

    if not text:
        sb.table("documents").update({"status": "error"}).eq("id", req.document_id).execute()
        return {"ok": False, "error": "empty text after extraction"}

    # 4) chunk
    chunker = _build_chunker(req)
    chunks = chunker.chunk_text(text)
    n_chunks = len(chunks)

    # 5) save chunks
    sb.table("chunks").delete().eq("document_id", req.document_id).execute()
    rows = [{"document_id": req.document_id, "chunk_index": i, "content": c} for i, c in enumerate(chunks)]
    if rows:
        sb.table("chunks").upsert(rows, on_conflict="document_id,chunk_index").execute()

    # 6) embed
    if rows:
        embeddings = embed_passages([r["content"] for r in rows])
        upserts = [
            {"document_id": r["document_id"], "chunk_index": r["chunk_index"], "embedding": e}
            for r, e in zip(rows, embeddings)
        ]
        sb.table("chunks").upsert(upserts, on_conflict="document_id,chunk_index").execute()

    # 7) update doc
    sb.table("documents").update({"status": "embedded", "pages": pages}).eq("id", req.document_id).execute()

    return {"ok": True, "document_id": req.document_id, "pages": pages, "chunks": n_chunks}


# ===================== M4: Retrieval =====================
@app.post("/search")
def search(req: SearchReq):
    """
    Retrieve konteks untuk RAG.
    1) embed query (e5, prefix 'query:')
    2) RPC match_chunks (pgvector)
    3) keyword boost + dedup
    """
    q_vec = model.encode([f"query: {req.question}"], normalize_embeddings=True, show_progress_bar=False)[0]
    q_vec = q_vec.astype(np.float32).tolist()

    # ambil kandidat lebih banyak untuk rerank
    match_count = max(20, req.top_k * 3)
    rpc = sb.rpc(
        "match_chunks",
        {"query_embedding": q_vec, "match_count": match_count, "filter_document": req.filter_document_id},
    ).execute()

    items: List[Dict[str, Any]] = rpc.data or []  # [{document_id, chunk_index, content, similarity}, ...]

    # keyword boost + sort
    items = _keyword_boost(items, req.question, boost=0.15)
    # dedup
    items = _dedup_jaccard(items, thr=0.75)
    # top_k
    items = items[: max(1, req.top_k)]

    # rapikan output
    out = [
        {
            "document_id": it["document_id"],
            "chunk_index": it["chunk_index"],
            "content": it.get("content", ""),
            "similarity": float(it.get("similarity", 0.0)),
            "score": float(it.get("score", it.get("similarity", 0.0))),
        }
        for it in items
    ]
    return {"ok": True, "items": out, "count": len(out)}


# ===================== Utilities =====================
@app.post("/embed/query")
def embed_query(req: EmbedReq):
    prefix = "query:" if (req.mode or "query").lower() == "query" else "passage:"
    vec = model.encode([f"{prefix} {req.text or ''}"], normalize_embeddings=True, show_progress_bar=False)[0]
    return {"embedding": vec.astype(np.float32).tolist(), "dim": len(vec)}

# Alias lama
@app.post("/embed/document")
def embed_document(req: ProcessReq):
    return process_document(req)
