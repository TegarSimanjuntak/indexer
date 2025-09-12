import os
import io
from typing import List, Optional

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


class ProcessReq(BaseModel):
    document_id: str
    # parameter opsional utk tuning chunker
    target_chunk_size: Optional[int] = 500
    min_chunk_size: Optional[int] = 200
    max_chunk_size: Optional[int] = 800
    overlap_ratio: Optional[float] = 0.1


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}


def pdf_to_text(data: bytes) -> tuple[str, int]:
    """Extract text & page count from raw PDF bytes."""
    reader = PdfReader(io.BytesIO(data))
    pages = len(reader.pages)
    texts: List[str] = []
    for i in range(pages):
        t = reader.pages[i].extract_text() or ""
        texts.append(t)
    return ("\n\n".join(texts).strip(), pages)


def embed_passages(texts: List[str]) -> List[List[float]]:
    """Encode passages with e5; returns list[vector]."""
    # e5 family expects prefixes
    texts = [f"passage: {t or ''}" for t in texts]
    vecs = model.encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
    )
    return vecs.astype(np.float32).tolist()


def _build_chunker(req: ProcessReq) -> ImprovedSemanticChunker:
    cfg = ChunkerConfig(
        target_chars=req.target_chunk_size or 500,
        min_chars=req.min_chunk_size or 200,
        max_chars=req.max_chunk_size or 800,
        overlap_ratio=req.overlap_ratio or 0.10,
        use_spacy=True,  # set False jika ingin tanpa spaCy
    )
    return ImprovedSemanticChunker(cfg)


@app.post("/process/document")
def process_document(req: ProcessReq):
    # 1) ambil dokumen
    doc = (
        sb.table("documents")
        .select("*")
        .eq("id", req.document_id)
        .single()
        .execute()
        .data
    )
    if not doc:
        return {"ok": False, "error": "document not found", "id": req.document_id}

    path = doc.get("storage_path") or doc.get("file_path")
    if not path:
        return {"ok": False, "error": "missing storage_path"}

    # 2) download PDF dari Storage
    try:
        file_bytes = sb.storage.from_(BUCKET).download(path)
        # supabase-py v2 sudah mengembalikan bytes
        if not isinstance(file_bytes, (bytes, bytearray)):
            # just in case: ada wrapper object
            file_bytes = bytes(file_bytes)
    except Exception as e:
        return {"ok": False, "error": f"download failed: {e}"}

    # 3) extract text + pages
    try:
        text, pages = pdf_to_text(file_bytes)
    except Exception as e:
        sb.table("documents").update({"status": "error"}).eq("id", req.document_id).execute()
        return {"ok": False, "error": f"pdf parse failed: {e}"}

    if not text:
        sb.table("documents").update({"status": "error"}).eq("id", req.document_id).execute()
        return {"ok": False, "error": "empty text after extraction"}

    # 4) chunking
    chunker = _build_chunker(req)
    chunks = chunker.chunk_text(text)  # -> List[str]
    n_chunks = len(chunks)

    # 5) simpan chunks (hapus lama dulu)
    sb.table("chunks").delete().eq("document_id", req.document_id).execute()
    rows = [
        {"document_id": req.document_id, "chunk_index": i, "content": c}
        for i, c in enumerate(chunks)
    ]
    if rows:
        sb.table("chunks").upsert(rows, on_conflict="document_id,chunk_index").execute()

    # 6) embedding
    if rows:
        embeddings = embed_passages([r["content"] for r in rows])
        upserts = [
            {
                "document_id": r["document_id"],
                "chunk_index": r["chunk_index"],
                "embedding": e,
            }
            for r, e in zip(rows, embeddings)
        ]
        sb.table("chunks").upsert(upserts, on_conflict="document_id,chunk_index").execute()

    # 7) update status dokumen
    sb.table("documents").update({"status": "embedded", "pages": pages}).eq(
        "id", req.document_id
    ).execute()

    return {
        "ok": True,
        "document_id": req.document_id,
        "pages": pages,
        "chunks": n_chunks,
    }


# Alias lama untuk kompatibilitas
@app.post("/embed/document")
def embed_document(req: ProcessReq):
    return process_document(req)
