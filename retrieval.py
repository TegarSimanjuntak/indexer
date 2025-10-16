# retrieval.py
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import hashlib
import os
import logging

logger = logging.getLogger("retrieval")
ENABLE_RERANK = os.environ.get("ENABLE_RERANK", "false").lower() in ("1", "true", "yes")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "intfloat/e5-base-v2")

# load models (caller may also reuse embed_model from indexer)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
reranker = None
if ENABLE_RERANK:
    try:
        reranker = CrossEncoder(RERANK_MODEL)
        logger.info("Loaded reranker %s", RERANK_MODEL)
    except Exception as e:
        logger.exception("Failed loading reranker: %s", e)
        reranker = None

def _normalize_text(t: str) -> str:
    return " ".join(t.strip().lower().split())

def _dedup_candidates(candidates: List[Dict], max_per_doc: int = 5) -> List[Dict]:
    """
    Simple dedup:
    - remove exact duplicates by prefix hash
    - limit how many chunks per document (optional)
    """
    seen = set()
    out = []
    per_doc_count = {}
    for c in candidates:
        txt = _normalize_text(c.get("text", "")[:200])  # prefix
        h = hashlib.md5(txt.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        doc = str(c.get("document_id"))
        per_doc_count[doc] = per_doc_count.get(doc, 0) + 1
        if per_doc_count[doc] > max_per_doc:
            continue
        seen.add(h)
        out.append(c)
    return out

def rpc_search(supabase_client, q_emb: List[float], k: int = 50, filter_document: Optional[str] = None) -> List[Dict]:
    """
    Call Supabase RPC match_chunks(query_embedding, k, filter_document)
    Returns list of dicts with keys: document_id, chunk_index, text, similarity
    """
    try:
        body = {"query_embedding": q_emb, "k": k, "filter_document": filter_document}
        res = supabase_client.rpc("match_chunks", body).execute()
        # error handling
        if hasattr(res, "error") and res.error:
            logger.warning("RPC match_chunks returned error: %s", res.error)
            return []
        data = getattr(res, "data", None) or (res if isinstance(res, dict) else None)
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        logger.exception("rpc_search failed: %s", e)
        return []

def retrieve_and_rerank(supabase_client,
                        query: str,
                        q_emb: List[float],
                        search_k: int = 50,
                        rerank_top_n: int = 10,
                        final_k: int = 3,
                        filter_document: Optional[str] = None) -> List[Dict]:
    """
    Full pipeline:
    - RPC vector search k
    - dedup
    - rerank top N (optional)
    - combine scores and return top final_k
    """
    # 1. raw vector candidates
    candidates = rpc_search(supabase_client, q_emb, k=search_k, filter_document=filter_document)
    if not candidates:
        return []

    # ensure float similarity
    for c in candidates:
        c['similarity'] = float(c.get('similarity') or 0.0)

    # 2. dedup / limit per doc
    candidates = _dedup_candidates(candidates, max_per_doc=8)

    # 3. take top search_k (already limited), sort by similarity desc
    candidates = sorted(candidates, key=lambda x: x.get('similarity', 0), reverse=True)

    # 4. rerank top-N if reranker available
    rerank_n = min(rerank_top_n, len(candidates))
    reranked = []
    if ENABLE_RERANK and reranker is not None and rerank_n > 0:
        topN = candidates[:rerank_n]
        pairs = [[query, c['text']] for c in topN]
        try:
            scores = reranker.predict(pairs)  # higher is better
            for i, c in enumerate(topN):
                c['rerank_score'] = float(scores[i])
            reranked = topN
        except Exception as e:
            logger.exception("Reranker failed: %s", e)
            reranked = topN
    else:
        # fallback: use similarity as rerank_score proxy
        for c in candidates[:rerank_n]:
            c['rerank_score'] = c.get('similarity', 0.0)
        reranked = candidates[:rerank_n]

    # 5. combine scores (weighted)
    for c in reranked:
        sim = c.get('similarity', 0.0)
        rr = c.get('rerank_score', 0.0)
        # weights tunable
        c['combined_score'] = 0.45 * sim + 0.55 * rr

    # 6. sort by combined, pick final_k
    final = sorted(reranked, key=lambda x: x['combined_score'], reverse=True)[:final_k]
    return final
