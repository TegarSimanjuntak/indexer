import os
import io
import re
import traceback
from typing import List, Optional, Dict, Any
from uuid import UUID

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ====== ENV ======
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment")
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
    # prefix sesuai kebutuhan model
    texts = [f"passage: {t or ''}" for t in texts]
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    # pastikan mengembalikan Python nested list of float
    out = []
    for v in vecs:
        if hasattr(v, "tolist"):
            out.append([float(x) for x in v.tolist()])
        else:
            out.append([float(x) for x in list(v)])
    return out


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


def is_uuid(val: str) -> bool:
    try:
        UUID(str(val))
        return True
    except Exception:
        return False


def resolve_document_record(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Try to resolve a document record from Supabase by:
     - id (uuid)
     - storage_path
     - filename
    Returns the row dict or None.
    """
    try:
        # try id (uuid)
        if is_uuid(document_id):
            q = sb.table("documents").select("*").eq("id", document_id).limit(1).execute()
            if getattr(q, "error", None):
                # continue to other resolution attempts
                pass
            else:
                data = getattr(q, "data", None)
                if data:
                    # Postgrest may return list
                    return data[0] if isinstance(data, list) else data

        # try storage_path
        q = sb.table("documents").select("*").eq("storage_path", document_id).limit(1).execute()
        if not getattr(q, "error", None) and getattr(q, "data", None):
            data = getattr(q, "data")
            return data[0] if isinstance(data, list) else data

        # try filename
        q = sb.table("documents").select("*").eq("filename", document_id).limit(1).execute()
        if not getattr(q, "error", None) and getattr(q, "data", None):
            data = getattr(q, "data")
            return data[0] if isinstance(data, list) else data
    except Exception as e:
        print("resolve_document_record exception:", e)
    return None


# ===================== M3: Process Document =====================
@app.post("/process/document")
def process_document(req: ProcessReq, request: Request):
    """
    Steps:
      1) resolve document record (support uuid OR storage_path/filename)
      2) download PDF from storage bucket
      3) extract text
      4) chunk text -> rows (use column 'text' to match DB)
      5) upsert chunks (text)
      6) embed and upsert embedding column
      7) update document status
    Returns detailed error information in response (dev).
    """
    try:
        print("process_document called with:", req.dict())

        # 1) resolve document
        doc = resolve_document_record(req.document_id)
        if not doc:
            msg = f"document not found: {req.document_id}"
            print(msg)
            return JSONResponse(status_code=404, content={"ok": False, "error": msg})

        doc_id = doc.get("id")
        storage_path = doc.get("storage_path") or doc.get("file_path")
        print("resolved doc id:", doc_id, "storage_path:", storage_path)

        if not storage_path:
            return JSONResponse(status_code=400, content={"ok": False, "error": "missing storage_path on document"})

        # mark processing (best-effort) so UI / backend can see it quickly
        try:
            if doc_id and is_uuid(str(doc_id)):
                sb.table("documents").update({"status": "processing"}).eq("id", doc_id).execute()
        except Exception as e:
            print("warning: could not set status=processing:", e)

        # 2) download
        try:
            data_resp = sb.storage.from_(BUCKET).download(storage_path)
            file_bytes = data_resp if isinstance(data_resp, (bytes, bytearray)) else bytes(data_resp)
        except Exception as e:
            tb = traceback.format_exc()
            print("download error:", e, tb)
            try:
                if doc_id and is_uuid(str(doc_id)):
                    sb.table("documents").update({"status": "error"}).eq("id", doc_id).execute()
            except Exception:
                pass
            return JSONResponse(status_code=500, content={"ok": False, "error": "download failed", "detail": str(e), "trace": tb})

        # 3) extract
        try:
            text, pages = pdf_to_text(file_bytes)
        except Exception as e:
            tb = traceback.format_exc()
            print("pdf parse error:", e, tb)
            try:
                if doc_id and is_uuid(str(doc_id)):
                    sb.table("documents").update({"status": "error"}).eq("id", doc_id).execute()
            except Exception:
                pass
            return JSONResponse(status_code=500, content={"ok": False, "error": "pdf parse failed", "detail": str(e), "trace": tb})

        if not text:
            try:
                if doc_id and is_uuid(str(doc_id)):
                    sb.table("documents").update({"status": "error"}).eq("id", doc_id).execute()
            except Exception:
                pass
            return JSONResponse(status_code=400, content={"ok": False, "error": "empty text after extraction"})

        # 4) chunk
        try:
            chunker = _build_chunker(req)
            chunks = chunker.chunk_text(text)
            n_chunks = len(chunks)
            print("n_chunks:", n_chunks)
        except Exception as e:
            tb = traceback.format_exc()
            print("chunking error:", e, tb)
            return JSONResponse(status_code=500, content={"ok": False, "error": "chunking failed", "detail": str(e), "trace": tb})

        # 5) save chunks (use 'text' column to match schema)
        try:
            # Delete previous chunks (best-effort) — try by uuid and by storage_path
            try:
                if doc_id:
                    sb.table("chunks").delete().eq("document_id", doc_id).execute()
            except Exception:
                try:
                    sb.table("chunks").delete().eq("document_id", storage_path).execute()
                except Exception:
                    pass

            # Create rows using the canonical doc_id (prefer uuid), fallback to storage_path
            canonical_doc_id = doc_id if doc_id else storage_path
            rows = [{"document_id": canonical_doc_id, "chunk_index": i, "text": c} for i, c in enumerate(chunks)]

            if rows:
                up = sb.table("chunks").upsert(rows, on_conflict="document_id,chunk_index").execute()
                print("chunks upsert response:", getattr(up, "status_code", None), getattr(up, "data", None), getattr(up, "error", None))
        except Exception as e:
            tb = traceback.format_exc()
            print("save chunks error:", e, tb)
            try:
                if doc_id and is_uuid(str(doc_id)):
                    sb.table("documents").update({"status": "error"}).eq("id", doc_id).execute()
            except Exception:
                pass
            return JSONResponse(status_code=500, content={"ok": False, "error": "save chunks failed", "detail": str(e), "trace": tb})

        # 6) embed
        try:
            if rows:
                embeddings = embed_passages([r["text"] for r in rows])
                upserts = []
                for r, emb in zip(rows, embeddings):
                    # ensure embedding is plain python list of floats
                    vec = emb if isinstance(emb, list) else (emb.tolist() if hasattr(emb, "tolist") else list(emb))
                    vec = [float(x) for x in vec]
                    upserts.append({
                        "document_id": r["document_id"],
                        "chunk_index": r["chunk_index"],
                        "embedding": vec
                    })
                emb_r = sb.table("chunks").upsert(upserts, on_conflict="document_id,chunk_index").execute()
                print("embedding upsert response:", getattr(emb_r, "status_code", None), getattr(emb_r, "data", None), getattr(emb_r, "error", None))
        except Exception as e:
            tb = traceback.format_exc()
            print("embedding error:", e, tb)
            try:
                if doc_id and is_uuid(str(doc_id)):
                    sb.table("documents").update({"status": "error"}).eq("id", doc_id).execute()
            except Exception:
                pass
            return JSONResponse(status_code=500, content={"ok": False, "error": "embedding failed", "detail": str(e), "trace": tb})

        # 7) update doc
        try:
            if doc_id and is_uuid(str(doc_id)):
                sb.table("documents").update({"status": "embedded", "pages": pages}).eq("id", doc_id).execute()
            else:
                # if no uuid available, try update by storage_path
                try:
                    sb.table("documents").update({"status": "embedded", "pages": pages}).eq("storage_path", storage_path).execute()
                except Exception:
                    pass
        except Exception as e:
            print("warning: could not update document status:", e)

        return JSONResponse(status_code=200, content={"ok": True, "document_id": canonical_doc_id, "pages": pages, "chunks": n_chunks})
    except Exception as e:
        tb = traceback.format_exc()
        print("Unhandled exception in process_document:", e, tb)
        return JSONResponse(status_code=500, content={"ok": False, "error": "internal", "detail": str(e), "trace": tb})


# ===================== M4: Retrieval =====================
@app.post("/search")
def search(req: SearchReq):
    try:
        q_vec = model.encode([f"query: {req.question}"], normalize_embeddings=True, show_progress_bar=False)[0]
        q_vec = q_vec.astype(np.float32).tolist()

        match_count = max(20, req.top_k * 3)
        rpc = sb.rpc(
            "match_chunks",
            {"query_embedding": q_vec, "match_count": match_count, "filter_document": req.filter_document_id},
        ).execute()

        items: List[Dict[str, Any]] = getattr(rpc, "data", None) or []

        # Some RPCs may return 'text' field; normalize to 'content'
        normalized = []
        for it in items:
            content = it.get("content") or it.get("text") or ""
            normalized.append({
                "document_id": it.get("document_id"),
                "chunk_index": it.get("chunk_index"),
                "content": content,
                "similarity": float(it.get("similarity", 0.0)),
            })

        # keyword boost + sort
        normalized = _keyword_boost(normalized, req.question, boost=0.15)
        # dedup
        normalized = _dedup_jaccard(normalized, thr=0.75)
        # top_k
        out_items = normalized[: max(1, req.top_k)]

        out = [
            {
                "document_id": it["document_id"],
                "chunk_index": it["chunk_index"],
                "content": it.get("content", ""),
                "similarity": float(it.get("similarity", 0.0)),
                "score": float(it.get("score", it.get("similarity", 0.0))),
            }
            for it in out_items
        ]
        return {"ok": True, "items": out, "count": len(out)}
    except Exception as e:
        tb = traceback.format_exc()
        print("search error:", e, tb)
        return JSONResponse(status_code=500, content={"ok": False, "error": "search failed", "detail": str(e), "trace": tb})


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
