# chunker.py
# Clean semantic chunker: optional spaCy sentence split + TF-IDF (sklearn) similarity.
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re
import numpy as np

# ---- Optional spaCy (fallback to regex when unavailable) ----
try:
    import spacy
    _DEFAULT_SPACY_MODEL = "en_core_web_sm"  # ringan; ganti bila perlu
    try:
        _nlp = spacy.load(_DEFAULT_SPACY_MODEL, disable=["tagger", "ner", "lemmatizer"])
    except Exception:
        _nlp = None
except Exception:
    _nlp = None

# ---- TF-IDF + cosine similarity (sklearn) ----
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
      - Tanpa output berlebih; semua method pure.
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
        self._vectorizer: Optional[TfidfVectorizer] = None

    # ---------- Public API ----------
    def chunk_text(self, text: str) -> List[str]:
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
                return spacy.load(cfg.spacy_model, disable=["tagger", "ner", "lemmatizer"])
            return _nlp  # default loaded above
        except Exception:
            return None

    def _sentences(self, text: str) -> List[str]:
        # spaCy path (lebih akurat untuk EN)
        if self._nlp is not None:
            doc = self._nlp(text)
            out = []
            for s in doc.sents:
                st = s.text.strip()
                if st and len(st) > 2:
                    out.append(st)
            return out

        # fallback regex: split pada akhir kalimat diikuti spasi/newline + kapital
        # jaga agar tidak memotong setelah singkatan
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.replace("\r", ""))
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
        if len(sentences) < 2:
            return np.eye(1, dtype=float)

        # lazy vectorizer
        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(
                max_features=self.cfg.max_features,
                ngram_range=self.cfg.ngram_range,
                stop_words="english",  # aman untuk EN; untuk ID sebagian besar tetap ok
                token_pattern=r"\b\w+\b",
                max_df=0.95,
                min_df=1,
            )
        X = self._vectorizer.fit_transform(sentences)
        sim = cosine_similarity(X)
        np.fill_diagonal(sim, 1.0)
        return sim

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
            # jika menambah kalimat ini melebihi max dan kita sudah punya isi, potong sebelum i
            if cur_len + L > mx and i > start:
                cut = self._best_cut(sents, sim, start, i)
                boundaries.append((start, cut))
                start = cut
                cur_len = sum(lens[start:i+1])
                continue

            cur_len += L

            # bila sudah melewati target dan panjang saat ini memadai, cari cut terbaik di sekitar i+1
            if cur_len >= tgt and (i + 1 - start) >= 1 and cur_len >= mn:
                cut = self._best_cut(sents, sim, start, i + 1)
                boundaries.append((start, cut))
                start = cut
                cur_len = 0

        # tail
        if start < len(sents):
            boundaries.append((start, len(sents)))

        # gabungkan potongan yang terlalu pendek
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
        best_idx = end - 1
        best_score = float("-inf")

        # cari sekitar belakang (lebih dekat end) agar chunk tak terlalu pendek
        for cut in range(max(start + 1, end - 5), end + 1):
            left_len = sum(len(s) for s in sents[start:cut])
            right_len = sum(len(s) for s in sents[cut:end])

            if left_len < mn or left_len > mx:
                continue

            # gap semantik: 1 - similarity antar kalimat boundary
            gap = 0.0
            if 0 < cut < len(sents):
                gap = 1.0 - float(sim[cut - 1, cut])

            # kedekatan ke target (semakin dekat semakin baik)
            size_pref = -abs(left_len - tgt) / max(tgt, 1)

            score = gap * 0.7 + size_pref * 0.3
            if score > best_score:
                best_score, best_idx = score, cut

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
                # coba gabung dengan sebelumnya
                combined = (prev[0], b[1])
                if sum(len(s) for s in sents[combined[0]:combined[1]]) <= mx:
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
            # hitung overlap dalam jumlah kalimat (maks 3)
            ov_sents = 0
            if i > 0 and ov_ratio > 0:
                span = ed - st
                ov_sents = min(3, max(0, int(round(span * ov_ratio))))
            real_start = max(0, st - ov_sents) if i > 0 else st

            piece = self._assemble(sents[real_start:ed]).strip()
            if piece:
                chunks.append(piece)
        return chunks

    def _assemble(self, sents: List[str]) -> str:
        if not sents:
            return ""
        # gabung kalimat dengan spasi, jaga paragraf ganda
        txt = " ".join(s.strip() for s in sents)
        txt = re.sub(r"\s+\n", "\n", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        txt = re.sub(r" {2,}", " ", txt)
        return txt
