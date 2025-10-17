#!/usr/bin/env python3
"""
Indexer RAG (FastAPI)
- stores embeddings to column `embedding_768` (vector(768))
- uses EMBED_MODEL_NAME (default intfloat/e5-base-v2)
- RPC call expects match_chunks(query_embedding, k, filter_document uuid)
"""
import os
import io
import logging
from typing import Optional, List, Any, Dict
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request, Body
from pydantic import BaseModel
from pypdf import PdfReader
# we no longer import SentenceTransformer directly here (embedder in separate module)
from supabase import create_client
import requests
import numpy as np

# import chunker + embed helper (must be in same folder)
try:
    from chunker_embedder import chunk_text, embed_batches
except Exception as e:
    # will raise later when used, but log now
    chunk_text = None
    embed_batches = None
    logging.getLogger('indexer').exception('Failed importing chunker_embedder: %s', e)

# --- Config ---
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger('indexer')

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
SUPABASE_BUCKET = os.environ.get('SUPABASE_BUCKET', 'documents')

# chunk/embedding defaults (can be overridden by env)
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', '200'))
EMBED_MODEL_NAME = os.environ.get('EMBED_MODEL_NAME', 'intfloat/e5-base-v2')
EMBED_DIM_EXPECTED = int(os.environ.get('EMBED_DIM', '768'))

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    logger.warning('Supabase not fully configured. This indexer requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY')

supabase = None
try:
    if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        logger.info('Supabase client initialized')
except Exception as e:
    logger.exception('Failed to init supabase client: %s', e)

# --- Attempt to infer embedding dim by a quick embed call (lazy)
EMBED_DIM_ACTUAL = EMBED_DIM_EXPECTED
try:
    if embed_batches is not None:
        test_embs = embed_batches(["hello world"], model_name=EMBED_MODEL_NAME, batch_size=1)
        if test_embs and isinstance(test_embs, list) and len(test_embs[0]) > 0:
            EMBED_DIM_ACTUAL = int(len(test_embs[0]))
            logger.info('Detected embedding dim from model %s: %d', EMBED_MODEL_NAME, EMBED_DIM_ACTUAL)
            if EMBED_DIM_ACTUAL != EMBED_DIM_EXPECTED:
                logger.warning('EMBED_DIM_EXPECTED (%d) != actual model dim (%d). Set EMBED_DIM env if needed.',
                               EMBED_DIM_EXPECTED, EMBED_DIM_ACTUAL)
except Exception as e:
    logger.exception('Failed infer embedding dim via embed_batches; using EMBED_DIM_EXPECTED=%d', EMBED_DIM_EXPECTED)
    EMBED_DIM_ACTUAL = EMBED_DIM_EXPECTED

app = FastAPI(title='RAG Indexer')

class IndexPayload(BaseModel):
    document_id: str
    public_url: Optional[str] = None
    storage_path: Optional[str] = None
    filename: Optional[str] = None

# --- Utilities ---
def download_file(url: str) -> bytes:
    logger.info('Downloading file from %s', url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content

def _extract_bytes_from_supabase_storage_response(res: Any) -> Optional[bytes]:
    try:
        if res is None:
            return None
        if hasattr(res, 'read') and callable(res.read):
            try:
                return res.read()
            except Exception:
                pass
        if hasattr(res, 'content'):
            c = getattr(res, 'content')
            if isinstance(c, (bytes, bytearray)):
                return bytes(c)
        if isinstance(res, (bytes, bytearray)):
            return bytes(res)
        if isinstance(res, dict) and res.get('data'):
            d = res.get('data')
            if isinstance(d, (bytes, bytearray)):
                return bytes(d)
        if isinstance(res, tuple) and len(res) >= 1 and isinstance(res[0], (bytes, bytearray)):
            return bytes(res[0])
    except Exception as e:
        logger.debug('Exception while normalizing supabase storage response: %s', e)
    return None

def download_from_supabase(storage_path: str) -> Optional[bytes]:
    if not supabase:
        logger.warning('Supabase client not configured, cannot download from storage')
        return None
    try:
        logger.info('Fetching object from supabase storage: %s/%s', SUPABASE_BUCKET, storage_path)
        res = None
        try:
            res = supabase.storage.from_(SUPABASE_BUCKET).download(storage_path)
        except Exception as e:
            logger.debug('storage.download raised: %s', e)
            try:
                public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{storage_path}"
                logger.info('Falling back to public URL: %s', public_url)
                return download_file(public_url)
            except Exception as ee:
                logger.exception('Fallback download failed: %s', ee)
                return None

        data_bytes = _extract_bytes_from_supabase_storage_response(res)
        if data_bytes is None:
            err = None
            try:
                err = getattr(res, 'error', None)
            except Exception:
                err = None
            if not err and isinstance(res, dict):
                err = res.get('error') or res.get('message')
            if err:
                logger.warning('Supabase storage download reported error: %s', err)
            else:
                logger.warning('Supabase storage download returned unknown shape; falling back to public URL attempt')
                try:
                    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{storage_path}"
                    return download_file(public_url)
                except Exception as e:
                    logger.exception('Fallback public URL download failed: %s', e)
                    return None
        return data_bytes
    except Exception as e:
        logger.exception('Exception downloading from supabase: %s', e)
        return None

def extract_text_from_pdf_bytes(b: bytes) -> str:
    logger.info('Extracting text from PDF bytes (%d bytes)', len(b) if b else 0)
    try:
        reader = PdfReader(io.BytesIO(b))
        parts = []
        for p in reader.pages:
            try:
                t = p.extract_text() or ''
            except Exception:
                t = ''
            parts.append(t)
        text = '\n\n'.join(parts).strip()
        logger.info('Extracted approx %d chars', len(text))
        return text
    except Exception as e:
        logger.exception('Failed to extract pdf text: %s', e)
        return ''

def approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)

# --- DB helpers ---
def _get_supabase_response_error(res: Any) -> Optional[str]:
    try:
        if hasattr(res, 'error'):
            return getattr(res, 'error')
    except Exception:
        pass
    try:
        if isinstance(res, dict):
            if res.get('error'):
                return res.get('error')
            if res.get('message'):
                return res.get('message')
    except Exception:
        pass
    try:
        if hasattr(res, 'status_code'):
            code = getattr(res, 'status_code', 0)
            if int(code) >= 400:
                if hasattr(res, 'text'):
                    return f'status {code}: {getattr(res, "text", "")}'
                if hasattr(res, 'data'):
                    return f'status {code}: {getattr(res, "data", "")}'
                return f'status {code}'
    except Exception:
        pass
    return None

def upsert_chunks_to_supabase(document_id: str, chunks: List[str], embeddings: List[List[float]]):
    """
    Upsert chunk rows to Supabase; writes embedding into `embedding_768`.
    Requires unique constraint on (document_id, chunk_index) for upsert to replace duplicates.
    """
    if not supabase:
        logger.warning('Supabase client not configured; skipping DB insert')
        return {'inserted': 0}
    payload = []
    ts = datetime.utcnow().isoformat()
    target_col = 'embedding_768'  # prefer this column (vector(768))
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        # ensure float32 list
        arr = np.array(emb, dtype=np.float32).tolist()
        row = {
            'document_id': document_id,
            'chunk_index': i,
            'text': chunk,
            'tokens': approx_token_count(chunk),
            'created_at': ts,
            target_col: arr
        }
        payload.append(row)

    try:
        logger.info('Upserting %d chunks to supabase (target_col=%s)', len(payload), target_col)
        # try upsert (works if unique constraint exists)
        res = supabase.table('chunks').upsert(payload).execute()
        err = _get_supabase_response_error(res)
        if err:
            logger.warning('Upsert returned error or not supported: %s - trying insert', err)
            res2 = supabase.table('chunks').insert(payload).execute()
            err2 = _get_supabase_response_error(res2)
            if err2:
                logger.error('Insert fallback failed: %s', err2)
                return {'error': err2, 'inserted': 0}
            return {'inserted': len(payload), 'data': getattr(res2, 'data', None)}
        return {'inserted': len(payload), 'data': getattr(res, 'data', None)}
    except Exception as e:
        logger.exception('Exception upserting chunks: %s', e)
        return {'error': str(e), 'inserted': 0}

def update_document_status(document_id: str, status: str):
    if not supabase:
        logger.warning('Supabase not configured; cannot update document status')
        return
    try:
        logger.info('Updating documents.%s -> %s', document_id, status)
        res = supabase.table('documents').update({'status': status}).eq('id', document_id).execute()
        err = _get_supabase_response_error(res)
        if err:
            logger.info('Update by id failed (%s), trying storage_path/filename fallback', err)
            res2 = supabase.table('documents').update({'status': status}).eq('storage_path', document_id).execute()
            err2 = _get_supabase_response_error(res2)
            if err2:
                logger.warning('Failed to update documents status by id and storage_path: %s / %s', err, err2)
            else:
                logger.info('Updated document.status via storage_path')
        else:
            logger.info('Updated document.status via id')
    except Exception as e:
        logger.exception('Exception updating document status: %s', e)

# --- Search endpoint for RAG worker (used by backend) ---
@app.post('/search')
async def search_endpoint(payload: Dict[str, Any] = Body(...)):
    """
    Expected body:
    {
      "query": "text",
      "k": 6,
      "filter_document": "<optional uuid or null>"
    }
    Response:
    { items: [ { document_id, chunk_index, text, similarity } ] }
    """
    query = (payload.get('query') or '').strip()
    k = int(payload.get('k') or 6)
    filter_document = payload.get('filter_document')  # should be uuid string or None

    if not query:
        raise HTTPException(status_code=400, detail='query required')

    logger.info('Search request: k=%d filter=%s q="%s"', k, filter_document, query[:120])

    # 1) compute embedding for query (use embed_batches to be consistent)
    try:
        if embed_batches is None:
            raise RuntimeError('embed_batches not available; ensure chunker_embedder.py is present')
        q_embs = embed_batches([query], model_name=EMBED_MODEL_NAME, batch_size=1)
        if not q_embs or not isinstance(q_embs, list):
            raise RuntimeError('embed_batches returned unexpected result')
        q_emb = q_embs[0]
    except Exception as e:
        logger.exception('Query embedding failed: %s', e)
        raise HTTPException(status_code=500, detail='query embed failed')

    # 2) try pgvector RPC 'match_chunks' (common pattern). If it exists, use it.
    try:
        if supabase:
            logger.info('Attempting supabase RPC match_chunks')
            # call RPC with parameter names: query_embedding, k, filter_document
            rpc_body = {'query_embedding': q_emb, 'k': k, 'filter_document': filter_document}
            rpc_res = supabase.rpc('match_chunks', rpc_body).execute()
            err = _get_supabase_response_error(rpc_res)
            if not err:
                data = getattr(rpc_res, 'data', None) or (rpc_res if isinstance(rpc_res, dict) else None)
                items = []
                if isinstance(data, list):
                    for it in data[:k]:
                        items.append({
                            'document_id': it.get('document_id'),
                            'chunk_index': it.get('chunk_index'),
                            'text': it.get('text'),
                            'similarity': float(it.get('similarity') or 0)
                        })
                logger.info('RPC match_chunks returned %d items', len(items))
                return {'items': items}
            else:
                logger.debug('RPC match_chunks returned error or not available: %s', err)
    except Exception as e:
        logger.debug('RPC match_chunks attempt failed: %s', e)

    # 3) Fallback: simple vector search using match on chunks table if pgvector operator available via filter
    try:
        if supabase:
            logger.info('Fallback: performing naive text search on chunks (ilike)')
            safe_q = query.replace('%', '').replace('_', ' ')
            qlike = f'%{safe_q}%'
            sel = supabase.table('chunks').select('document_id,chunk_index,text,created_at').ilike('text', qlike).limit(k)
            if filter_document:
                # supabase-js/py will handle uuid string; this equality expects uuid type in DB
                sel = sel.eq('document_id', filter_document)
            res = sel.execute()
            err = _get_supabase_response_error(res)
            if err:
                logger.warning('Fallback text search failed: %s', err)
            else:
                data = getattr(res, 'data', None) or (res if isinstance(res, dict) else None)
                items = []
                if isinstance(data, list):
                    for it in data:
                        items.append({
                            'document_id': it.get('document_id'),
                            'chunk_index': it.get('chunk_index'),
                            'text': (it.get('text') or '')[:2000],
                            'similarity': 0.0
                        })
                logger.info('Fallback text search returned %d items', len(items))
                return {'items': items}
    except Exception as e:
        logger.exception('Fallback text search failed: %s', e)

    # 4) final fallback: empty
    logger.info('Search returning empty items')
    return {'items': []}

# --- Main endpoint ---
@app.post('/')
async def index(payload: IndexPayload, request: Request):
    logger.info('Received indexing request: %s', payload.dict())
    document_id = str(payload.document_id)

    # 1) fetch file bytes
    file_bytes = None
    try:
        if payload.public_url:
            file_bytes = download_file(payload.public_url)
        elif payload.storage_path:
            file_bytes = download_from_supabase(payload.storage_path)
        else:
            raise HTTPException(status_code=400, detail='No public_url or storage_path provided')

        if not file_bytes:
            logger.warning('No bytes fetched for document %s', document_id)
            update_document_status(document_id, 'error')
            raise HTTPException(status_code=400, detail='Failed to download file bytes')
    except HTTPException:
        raise
    except Exception as e:
        logger.exception('Download/prepare failed: %s', e)
        update_document_status(document_id, 'error')
        raise HTTPException(status_code=500, detail=f'download failed: {e}')

    # 2) extract text
    try:
        text = extract_text_from_pdf_bytes(file_bytes)
        if not text:
            logger.warning('No text extracted from PDF for document %s', document_id)
    except Exception as e:
        logger.exception('Text extraction failed: %s', e)
        update_document_status(document_id, 'error')
        raise HTTPException(status_code=500, detail=f'extract failed: {e}')

    # 3) chunk (use sentence-aware chunker from chunker_embedder)
    if chunk_text is None:
        logger.error('chunk_text function not available. Ensure chunker_embedder.py is present.')
        update_document_status(document_id, 'error')
        raise HTTPException(status_code=500, detail='chunker not available')

    chunks = chunk_text(
        text,
        chunk_size=int(os.environ.get('CHUNK_MAX_CHARS', CHUNK_SIZE)),
        overlap=CHUNK_OVERLAP,
        by_sentences=True,
        sentences_per_chunk=int(os.environ.get('CHUNK_SENTENCES', '3')),
        sentence_overlap=int(os.environ.get('CHUNK_SENTENCE_OVERLAP', '1')),
        max_chars=int(os.environ.get('CHUNK_MAX_CHARS', CHUNK_SIZE))
    )
    logger.info('Created %d chunks (sentence-based, max_chars=%s)', len(chunks), os.environ.get('CHUNK_MAX_CHARS', CHUNK_SIZE))

    if not chunks:
        update_document_status(document_id, 'error')
        raise HTTPException(status_code=400, detail='No chunks produced')

    # 4) embed in batches using embed_batches helper
    if embed_batches is None:
        logger.error('embed_batches not available. Ensure chunker_embedder.py is present.')
        update_document_status(document_id, 'error')
        raise HTTPException(status_code=500, detail='embedder not available')

    try:
        embeddings = embed_batches(chunks, model_name=EMBED_MODEL_NAME, batch_size=int(os.environ.get('EMBED_BATCH_SIZE', '64')))
    except Exception as e:
        logger.exception('Embedding failed via embed_batches: %s', e)
        update_document_status(document_id, 'error')
        raise HTTPException(status_code=500, detail=f'embed failed: {e}')

    # 5) validate embedding dims quickly
    if len(embeddings) > 0 and len(embeddings[0]) != EMBED_DIM_ACTUAL:
        logger.warning('Embedding dim mismatch: actual %d vs expected %d', len(embeddings[0]), EMBED_DIM_ACTUAL)

    # 6) upsert into Supabase
    try:
        ins = upsert_chunks_to_supabase(document_id, chunks, embeddings)
        if ins.get('inserted', 0) == 0:
            logger.warning('No chunks inserted for document %s', document_id)
            update_document_status(document_id, 'error')
            raise HTTPException(status_code=500, detail='db insert failed')
    except HTTPException:
        raise
    except Exception as e:
        logger.exception('Insert failed: %s', e)
        update_document_status(document_id, 'error')
        raise HTTPException(status_code=500, detail=f'insert failed: {e}')

    # 7) update document status
    try:
        update_document_status(document_id, 'embedded')
    except Exception:
        logger.exception('Failed to update document status to embedded')

    return { 'status': 'ok', 'document_id': document_id, 'chunks': len(chunks) }
