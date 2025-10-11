# chunker.py
# Clean semantic chunker: optional spaCy sentence split + TF-IDF (sklearn) similarity.
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re
import numpy as np

# ---- Optional spaCy (fallback to regex when unavailable) ----
try:
    import spacy  # type: ignore
    _DEFAULT_SPACY_MODEL = "en_core_web_sm"  # ringan; ganti bila perlu
    try:
        _nlp = spacy.load(_DEFAULT_SPACY_MODEL, disable=["tagger", "ner", "lemmatizer"])
    except Exception:
        # model not available or failed to load; set to None and fallback to regex
        _nlp = None
except Exception:
    _nlp = None

# ---- TF-IDF + cosine similarity (sklearn) ----
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    _SKLEARN_AVAILABLE = True
except Exception:
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore
    _SKLEARN_AVAILABLE = False


@dataclass
class ChunkerConfig:
    target_chars: int = 500       # target panjang chunk (karakter)
    min_chars: int = 200          # minimum panjang chunk
    max_chars: int = 800          # maksimum panjang chunk
    overlap_ratio: float = 0.10   # overlap berdasar jumlah kalimat (0..0.3)
    use_spacy: bool = True        # pakai spaCy bila tersedia
    spacy_model: Optional[str] = None  # set kalau mau pakai model selain default
    # TF-IDF
    max_features: int = 2000
    ngram_range: Tuple[int, int] = (1, 2)


class ImprovedSemanticChunker:
    """
    Clean semantic chunker:
      - Split kalimat dengan spaCy (jika ada) atau regex yang aman.
      - Cari titik potong dekat target_chars menggunakan sinyal ukuran + similarity antar kalimat.
      - Tambahkan overlap kalimat kecil untuk konteks.
      - Semua method pure (tidak mengubah state global secara tak terduga).
    """

    _ABBREV = {
        # umum EN; tambahkan sesuai kebutuhan
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Jr.", "Sr.",
        "etc.", "e.g.", "i.e.", "vs.", "Fig.", "No.", "pp.", "p.",
        "a.m.", "p.m.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.",
        "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.",
        # beberapa singkatan ID yang sering muncul (opsional)
        "No.", "Ds.", "Jl."
    }

    _HEADER_PATTERNS = [
        re.compile(r"^\s*#{1,6}\s+\S.+$", re.MULTILINE),            # markdown header
        re.compile(r"^\s*\d+(\.\d+)*\s+\S.+$", re.MULTILINE),       # 1., 1.2, dst
        re.compile(r"^[A-Z][A-Z\s]{2,}$", re.MULTILINE),            # ALL CAPS
        re.compile(r"^\s*[A-Z][^.!?]{1,80}$", re.MULTILINE),        # Judul pendek tanpa tanda baca akhir
    ]

    def __init__(self, cfg: ChunkerConfig = ChunkerConfig()):
        self.cfg = cfg
        self._nlp = self._init_spacy(cfg) if cfg.use_spacy else None
        self._vectorizer: Optional["TfidfVectorizer"] = None

    # ---------- Public API ----------
    def chunk_text(self, text: str) -> List[str]:
        """
        Main entry: returns list of chunk strings.
        """
        text = (text or "").strip()
        if not text:
            return []

        if len(text) <= self.cfg.min_chars:
            return [text]

        sents = self._sentences(text)
        if not sents:
            return [text]
        if len(sents) == 1:
            return [sents[0]]

        sim = self._similarity_matrix(sents)
        boundaries = self._choose_boundaries(sents, sim)
        chunks = self._build_chunks(sents, boundaries)
        return chunks

    # ---------- Sentence splitting ----------
    def _init_spacy(self, cfg: ChunkerConfig):
        try:
            if cfg.spacy_model:
                import spacy  # type: ignore
                return spacy.load(cfg.spacy_model, disable=["tagger", "ner", "lemmatizer"])
            return _nlp  # default loaded above (may be None)
        except Exception:
            return None

    def _sentences(self, text: str) -> List[str]:
        # spaCy path (lebih akurat untuk EN)
        if self._nlp is not None:
            try:
                doc = self._nlp(text)
                out = []
                for s in doc.sents:
                    st = s.text.strip()
                    if st and len(st) > 2:
                        out.append(st)
                return out
            except Exception:
                # fallback to regex below if spaCy fails unexpectedly
                pass

        # fallback regex: split pada akhir kalimat diikuti spasi/newline + kapital
        # jaga agar tidak memotong setelah singkatan
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\“\"'])", text.replace("\r", ""))
        out: List[str] = []
        for p in parts:
            st = p.strip()
            if not st:
                continue
            # hindari boundary tepat setelah singkatan umum
            last_word = st.split()[-1] if st.split() else ""
            if last_word in self._ABBREV and out:
                out[-1] = out[-1] + " " + st
            else:
                out.append(st)
        return out

    # ---------- Similarity ----------
    def _similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Return NxN similarity matrix. If TF-IDF not available or fails,
        fallback to simple adjacency identity matrix.
        """
        n = len(sentences)
        if n < 2 or not _SKLEARN_AVAILABLE:
            return np.eye(n, dtype=float)

        try:
            # lazy vectorizer
            if self._vectorizer is None:
                self._vectorizer = TfidfVectorizer(
                    max_features=self.cfg.max_features,
                    ngram_range=self.cfg.ngram_range,
                    stop_words="english",
                    token_pattern=r"\b\w+\b",
                    max_df=0.95,
                    min_df=1,
                )
            # If there are fewer sentences than features, tfidf still works.
            X = self._vectorizer.fit_transform(sentences)
            sim = cosine_similarity(X)
            # numerical stability
            sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(sim, 1.0)
            return sim
        except Exception:
            # fallback safe identity
            return np.eye(n, dtype=float)

    # ---------- Boundary selection ----------
    def _choose_boundaries(self, sents: List[str], sim: np.ndarray) -> List[Tuple[int, int]]:
        tgt = self.cfg.target_chars
        mn = self.cfg.min_chars
        mx = self.cfg.max_chars

        # precompute lengths
        lens = [len(s) for s in sents]

        boundaries: List[Tuple[int, int]] = []
        start = 0
        cur_len = 0

        for i, L in enumerate(lens):
            # if adding this sentence exceeds max and we already have some content -> cut before i
            if cur_len + L > mx and i > start:
                cut = self._best_cut(sents, sim, start, i)
                boundaries.append((start, cut))
                start = cut
                # recompute cur_len from start to current i
                cur_len = sum(lens[start:i+1]) if i >= start else 0
                continue

            cur_len += L

            # if reached target and length is ok, find cut near i+1
            if cur_len >= tgt and (i + 1 - start) >= 1 and cur_len >= mn:
                cut = self._best_cut(sents, sim, start, i + 1)
                boundaries.append((start, cut))
                start = cut
                cur_len = 0

        # tail
        if start < len(sents):
            boundaries.append((start, len(sents)))

        # merge too-small chunks with previous if possible
        boundaries = self._merge_small(sents, boundaries, mn, mx)
        return boundaries

    def _best_cut(self, sents: List[str], sim: np.ndarray, start: int, end: int) -> int:
        """
        Pilih titik potong (index kalimat) di (start+1 .. end) yang:
          - tidak menghasilkan chunk < min_chars
          - sedekat mungkin dengan target_chars
          - dan punya 'semantic gap' relatif besar di boundary (sim rendah)
        """
        mn, mx, tgt = self.cfg.min_chars, self.cfg.max_chars, self.cfg.target_chars
        best_idx = max(start + 1, min(end, start + 1))
        best_score = float("-inf")

        # candidate cuts: range from start+1 .. end (inclusive end)
        # but we bias to prefer cuts closer to 'end'
        cand_start = max(start + 1, end - 5)
        cand_end = end
        for cut in range(cand_start, cand_end + 1):
            left_len = sum(len(s) for s in sents[start:cut])
            # enforce min/max
            if left_len < mn or left_len > mx:
                continue

            # gap semantik: 1 - similarity antar kalimat boundary (if available)
            gap = 0.0
            if 0 < cut < len(sents):
                try:
                    gap = 1.0 - float(sim[cut - 1, cut])
                    if gap < 0.0:
                        gap = 0.0
                except Exception:
                    gap = 0.0

            # closeness to target (neg abs)
            size_pref = -abs(left_len - tgt) / max(tgt, 1)

            # weighted score
            score = gap * 0.7 + size_pref * 0.3
            if score > best_score:
                best_score, best_idx = score, cut

        # ensure at least start+1
        return max(start + 1, min(best_idx, end))

    def _merge_small(self, sents: List[str], bounds: List[Tuple[int, int]], mn: int, mx: int):
        if not bounds:
            return bounds
        merged: List[Tuple[int, int]] = []
        for b in bounds:
            if not merged:
                merged.append(b)
                continue
            prev = merged[-1]
            cur_len = sum(len(s) for s in sents[b[0]:b[1]])
            if cur_len < mn:
                # try to combine with previous if still <= mx
                combined = (prev[0], b[1])
                comb_len = sum(len(s) for s in sents[combined[0]:combined[1]])
                if comb_len <= mx:
                    merged[-1] = combined
                else:
                    merged.append(b)
            else:
                merged.append(b)
        return merged

    # ---------- Build text with overlap ----------
    def _build_chunks(self, sents: List[str], bounds: List[Tuple[int, int]]) -> List[str]:
        ov_ratio = max(0.0, min(self.cfg.overlap_ratio, 0.30))
        chunks: List[str] = []
        for i, (st, ed) in enumerate(bounds):
            # compute overlap in number of sentences (bounded to at most 3)
            ov_sents = 0
            if i > 0 and ov_ratio > 0:
                span = ed - st
                # prefer at least 1 sentence overlap if span big enough
                ov_sents = min(3, max(0, int(round(span * ov_ratio))))
            real_start = max(0, st - ov_sents) if i > 0 else st

            piece = self._assemble(sents[real_start:ed]).strip()
            if piece:
                chunks.append(piece)
        return chunks

    def _assemble(self, sents: List[str]) -> str:
        if not sents:
            return ""
        # join sentences with a single space, then normalize whitespace and newlines
        txt = " ".join(s.strip() for s in sents)
        # keep paragraph breaks reasonable
        txt = re.sub(r"\s+\n", "\n", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        txt = re.sub(r" {2,}", " ", txt)
        return txt
