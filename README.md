# RAG Indexer Service 🚀

> **Backend Service untuk Sistem Question Answering Mata Kuliah Manajemen Proyek**

Repositori ini berisi kode sumber untuk **Indexer Service**, komponen inti dari sistem *Retrieval-Augmented Generation* (RAG) yang dikembangkan sebagai bagian dari Skripsi S-1 Teknik Informatika Universitas Padjadjaran.

Layanan ini bertanggung jawab untuk memproses dokumen pembelajaran (PDF), mengubahnya menjadi representasi vektor (*embedding*), dan menangani pencarian semantik (*retrieval*) dengan mekanisme *Two-Stage Retrieval* (Search + Reranking).

---

## ✨ Fitur Utama

* **📄 PDF Ingestion & Extraction**: Ekstraksi teks otomatis dari file materi kuliah.
* **🧩 Sentence-Aware Chunking**: Algoritma pemotongan teks cerdas yang menjaga keutuhan kalimat dan konteks (Default: ±600 karakter, overlap aktif).
* **🔢 Embedding Generation**: Menggunakan model `intfloat/e5-base-v2` untuk representasi semantik berkualitas tinggi.
* **🔍 Two-Stage Retrieval**:
    1.  **Recall**: Pencarian vektor cepat menggunakan Supabase `pgvector` (Cosine Similarity).
    2.  **Precision**: Pemeringkatan ulang (*Reranking*) menggunakan `cross-encoder/ms-marco-MiniLM-L-6-v2`.
* **⚡ FastAPI Powered**: Dibangun di atas framework Python yang cepat dan modern.

---

## 🛠️ Teknologi yang Digunakan

* **Language**: Python 3.9+
* **Framework**: FastAPI / Uvicorn
* **Database**: Supabase (PostgreSQL + `pgvector`)
* **ML Libraries**: `sentence-transformers`, `numpy`, `PyPDF2`
* **Architecture**: Decoupled Service (Indexer terpisah dari Frontend).

---

## ⚙️ Instalasi dan Konfigurasi

### 1. Clone Repositori
```bash
git clone [https://github.com/TegarSimanjuntak/indexer.git](https://github.com/TegarSimanjuntak/indexer.git)
cd indexer

```

### 2. Install Dependensi

Pastikan Anda menggunakan virtual environment untuk menjaga kebersihan dependensi.

```bash
python -m venv venv
source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
pip install -r requirements.txt

```

### 3. Konfigurasi Environment Variables (`.env`)

Buat file `.env` di root directory dan sesuaikan parameter berikut:

```env
# Server Config
PORT=8000

# Supabase Config
SUPABASE_URL=[https://your-project.supabase.co](https://your-project.supabase.co)
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_BUCKET=documents

# RAG Hyperparameters
CHUNK_SIZE=600
CHUNK_OVERLAP=90
CHUNK_BY_SENTENCES=true
CHUNK_SENTENCES=3
CHUNK_SENTENCE_OVERLAP=1
CHUNK_MAX_CHARS=800

# AI Models
EMBED_MODEL_NAME=intfloat/e5-base-v2
EMBED_BATCH_SIZE=64
ENABLE_RERANK=true
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

```

### 4. Menjalankan Server

```bash
# Menjalankan script langsung
python indexer_rag.py

# Atau menggunakan uvicorn
uvicorn indexer_rag:app --reload

```

---

## 🔌 API Endpoints

Layanan ini mengekspos dua endpoint utama untuk integrasi:

| Method | Endpoint | Deskripsi | Payload Contoh |
| --- | --- | --- | --- |
| **POST** | `/indexing` | Proses PDF, chunking, embedding, & simpan ke DB. | `{"document_id": "uuid", "public_url": "..."}` |
| **POST** | `/search` | Mencari chunks relevan berdasarkan query. | `{"query": "Apa itu WBS?", "k": 3}` |

---

## 📊 Performa Sistem

Berdasarkan pengujian skripsi menggunakan dokumen **PMBOK® Guide – Seventh Edition**, sistem ini mencapai kinerja sebagai berikut:

| Metrik Evaluasi | Skor |
| --- | --- |
| **Precision@3** | 0.78 |
| **Recall@5** | 0.87 |
| **Mean Reciprocal Rank (MRR)** | 0.81 |

> **Note:** Implementasi RAG ini terbukti meningkatkan skor **Faithfulness** jawaban secara signifikan dari **2.7 (Tanpa RAG)** menjadi **4.3 (Dengan RAG)** dalam skala 5.

---

## 👤 Author

**Tegar Posma Diaz Simanjuntak**

* **NPM**: 140810220085
* **Program Studi**: Teknik Informatika, Universitas Padjadjaran

*Repositori ini dikembangkan sebagai bagian dari tugas akhir (Skripsi) tahun 2026.*

```

Apakah ada bagian spesifik lainnya yang ingin kamu tambahkan atau sesuaikan?

```
---

## 👤 Dosen Pembimbing

**Dr. Afrida Helen, S.T., M.Kom.**

*

**Dr. Intan Nurma Yulita, S.T., M.T**

* 

---
