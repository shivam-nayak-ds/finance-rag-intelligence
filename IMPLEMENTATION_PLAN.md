# SyllAIq — Complete Implementation Plan (Phase 1–10)
### CS Engineering RAG Intelligence System

---

> **IMPORTANT**: Full Rewrite Strategy — Saara existing finance-domain code hataya jayega.
> Architecture same rahega (Hybrid RAG), lekin har file CS domain ke liye naya likhenge.

---

## Current Codebase Audit

### Files to REWRITE (Keep structure, change domain logic)
| Existing File | Action | Reason |
|---|---|---|
| `src/ingestion/pdf_loader.py` | **REWRITE** | Finance metadata → CS metadata (subject, book, chapter, page) |
| `src/ingestion/data_cleaner.py` | **REWRITE** | Finance-specific cleaning → Generic academic text cleaning |
| `src/chunking/recursive_chunker.py` | **REWRITE** | Rename class + CS-aware chunk sizes |
| `src/chunking/semantic_chunker.py` | **REWRITE** | Rename class |
| `src/retrieval/hybrid_retriever.py` | **REWRITE** | Return full metadata with chunks (not just strings) |
| `src/retrieval/dense_retriever.py` | **REWRITE** | Return Document objects with metadata |
| `src/retrieval/sparse_retriever.py` | **REWRITE** | Return Document objects with metadata |
| `src/reranking/cohere_reranker.py` | **REWRITE** | Return scores alongside chunks for confidence scoring |
| `src/generation/prompt_templates.py` | **REWRITE** | Finance persona → SyllAIq CS tutor persona + citation format |
| `src/generation/llm_chain.py` | **REWRITE** | Return structured response: answer + sources + confidence |
| `src/generation/memory.py` | **REWRITE** | Session-aware memory for student context |
| `src/embedding/hf_embedder.py` | **REWRITE** | Rename class |
| `src/vectorstore/chromadb_store.py` | **REWRITE** | Store + retrieve full metadata |
| `src/vectorstore/faiss_store.py` | **REWRITE** | CS domain |
| `src/pipeline/injestion_pipeline.py` | **REWRITE** | Recursive folder ingestion + CS metadata tagging |
| `src/pipeline/rag_pipeline.py` | **REWRITE** | Full advanced RAG orchestration |
| `src/pipeline/self_rag_pipeline.py` | **REWRITE** | Complete self-RAG with hallucination loop |
| `src/evaluation/ragas_eval.py` | **REWRITE** | 4-metric RAGAS evaluation |
| `app/streamlit_app.py` | **REWRITE** | CS student UI (subject filter, source citation, confidence) |
| `app/main.py` | **REWRITE** | FastAPI backend |
| `tests/test_pipeline.py` | **REWRITE** | CS-domain test cases |
| `requirements.txt` | **REWRITE** | Add new dependencies |
| `README.md` | **REWRITE** | SyllAIq documentation |
| `.env` | **REWRITE** | Clean keys with proper naming |
| `docker/Dockerfile` | **REWRITE** | SyllAIq Docker config |
| `docker/docker-compose.yml` | **REWRITE** | Updated compose |

### Files to DELETE
| File | Reason |
|---|---|
| `src/tools/stock_tool.py` | Finance-specific, not needed |
| `notebooks/experiments.ipynb` | Finance experiments, will recreate |

### New Files to CREATE
| New File | Phase | Purpose |
|---|---|---|
| `src/tools/query_rewriter.py` | Phase 6 | LLM-based query improvement |
| `src/tools/confidence_scorer.py` | Phase 6 | Reranker score → confidence % |
| `src/security/input_guardrails.py` | Phase 8 | Prompt injection + scope filter |
| `src/security/rate_limiter.py` | Phase 8 | Per-user request throttling |
| `src/security/output_validator.py` | Phase 8 | Answer safety + PII filter |
| `src/security/__init__.py` | Phase 8 | Module init |
| `src/tools/__init__.py` | Phase 6 | Module init |
| `data/raw/CS_Engineering/` | Phase 2 | CS subject folder hierarchy |
| `scripts/setup_folders.py` | Phase 2 | Auto-create data folder structure |
| `notebooks/syllaiq_experiments.ipynb` | Phase 9 | CS RAG experiments |
| `config/settings.py` | Phase 1 | Centralized config (paths, model names, thresholds) |
| `config/__init__.py` | Phase 1 | Module init |

---

## Phase-by-Phase Implementation Plan

---

### PHASE 1 — Project Restructure & Config [DONE]
**Goal**: Finance references hataao, SyllAIq identity set karo, central config banao.

- [x] README.md — SyllAIq full documentation
- [x] .env — Clean API keys with proper variable names
- [x] requirements.txt — Updated with langgraph, groq, cohere, sentence-transformers, pdfplumber, fastapi, slowapi
- [x] config/__init__.py — Created
- [x] config/settings.py — Centralized settings (50+ settings)
- [x] Deleted stock_tool.py (finance-specific)
- [x] All new packages installed and verified

---

### PHASE 2 — Data Layer Setup
**Goal**: CS Engineering ka proper folder hierarchy banana.

- [ ] scripts/setup_folders.py — Auto-create complete folder structure

```
data/raw/CS_Engineering/
|-- Operating_Systems/
|   |-- textbooks/        <- OS_Galvin_10th_Ed.pdf
|   `-- pyqs/             <- OS_AKTU_PYQ_2019_2024.pdf
|-- DBMS/
|   |-- textbooks/        <- DBMS_Navathe_7th_Ed.pdf
|   `-- pyqs/
|-- Computer_Networks/
|   |-- textbooks/        <- CN_Forouzan_6th_Ed.pdf
|   `-- pyqs/
|-- Data_Structures/
|   |-- textbooks/        <- DSA_Cormen_CLRS.pdf
|   `-- pyqs/
|-- OOP/
|   |-- textbooks/
|   `-- pyqs/
|-- Placement_Prep/
|   |-- system_design/
|   |-- aptitude/
|   `-- interview_questions/
`-- Syllabus/
```

---

### PHASE 3 — Ingestion Pipeline (PDF → Clean Text + Rich Metadata)
**Goal**: Recursive PDF loader jo CS metadata auto-extract kare.

- [ ] src/ingestion/pdf_loader.py — REWRITE
  - glob("*.pdf") → rglob("*.pdf") (recursive)
  - Auto-extract: subject, doc_type, book_name, university, year, page
- [ ] src/ingestion/data_cleaner.py — REWRITE
  - Remove headers/footers, page numbers, watermarks
  - Preserve mathematical formulas and code snippets

---

### PHASE 4 — Advanced Chunking (Metadata-Preserving)
**Goal**: Har chunk apna source metadata carry kare.

- [ ] src/chunking/recursive_chunker.py — REWRITE
  - Class rename: FinanceRecursiveChunker → CSRecursiveChunker
  - Each chunk carries: subject, doc_type, book_name, chapter, page
- [ ] src/chunking/semantic_chunker.py — REWRITE
  - Class rename: FinanceSemanticChunker → CSSemanticChunker

---

### PHASE 5 — Embedding + VectorStore (Metadata-Aware)
**Goal**: Chunks ko embed karke ChromaDB me metadata ke saath store karo.

- [ ] src/embedding/hf_embedder.py — REWRITE
  - Model: sentence-transformers/all-MiniLM-L6-v2
- [ ] src/vectorstore/chromadb_store.py — REWRITE
  - Store full metadata with each vector
  - Support metadata filtering by subject, doc_type
- [ ] src/vectorstore/faiss_store.py — REWRITE

---

### PHASE 6 — Advanced RAG Core (Query Rewriting + Source Citation + Confidence)
**Goal**: Reliable, trustworthy, cited answers.

- [ ] src/tools/__init__.py — NEW
- [ ] src/tools/query_rewriter.py — NEW
  - Input: "deadlock kya hota hai"
  - Output: "Explain Deadlock in OS: necessary conditions, Banker's Algorithm..."
- [ ] src/tools/confidence_scorer.py — NEW
  - Reranker scores → Weighted average → Confidence %
  - Score > 0.85 → High | 0.60-0.85 → Medium | < 0.60 → Low
- [ ] src/retrieval/dense_retriever.py — REWRITE (return Document objects)
- [ ] src/retrieval/sparse_retriever.py — REWRITE (return Document objects)
- [ ] src/retrieval/hybrid_retriever.py — REWRITE (return List[Document] with metadata)
- [ ] src/reranking/cohere_reranker.py — REWRITE (return chunks + scores)
- [ ] src/generation/prompt_templates.py — REWRITE (CS tutor persona + citation format)
- [ ] src/generation/llm_chain.py — REWRITE (return RAGResult dataclass)
- [ ] src/generation/memory.py — REWRITE

---

### PHASE 7 — Self-RAG Pipeline (LangGraph)
**Goal**: System khud verify kare ki answer dena chahiye ya nahi.

```
Query → Retrieve → [Relevance Check]
    YES → Generate → [Faithfulness Check]
                YES → Return Answer + Sources
                NO  → Retry with different chunks
    NO  → Re-query with rewritten query (max 2 retries)
         → If still no good chunks: "Not in knowledge base"
```

- [ ] src/pipeline/self_rag_pipeline.py — REWRITE (LangGraph graph)
- [ ] src/pipeline/rag_pipeline.py — REWRITE (main LangGraph orchestrator)
- [ ] src/pipeline/ingestion_pipeline.py — REWRITE

---

### PHASE 8 — Security Layer (3-Layer Protection)
**Goal**: Production-ready safety for student-facing system.

- [ ] src/security/__init__.py — NEW
- [ ] src/security/input_guardrails.py — NEW
  - Block: prompt injection, jailbreak, off-topic queries
- [ ] src/security/rate_limiter.py — NEW
  - Max 20 requests/minute per session
- [ ] src/security/output_validator.py — NEW
  - Scope check, PII filter, confidence gate

---

### PHASE 9 — Streamlit UI (Student-Facing Interface)
**Goal**: Beautiful, intuitive CS student study companion UI.

- [ ] app/streamlit_app.py — REWRITE
  - Subject Filter Sidebar (OS, DBMS, CN, DSA, OOP, Placement)
  - Chat Interface with conversation history
  - Source Citations panel (book, chapter, page)
  - Confidence Score badge (Green/Yellow/Red)
  - Low confidence warning banner
  - Related PYQs section
- [ ] app/main.py — REWRITE (FastAPI REST endpoint: POST /ask)
- [ ] notebooks/syllaiq_experiments.ipynb — NEW

---

### PHASE 10 — Evaluation + Docker + Documentation
**Goal**: Measure accuracy, containerize, document everything.

- [ ] src/evaluation/ragas_eval.py — REWRITE (4 RAGAS metrics)
- [ ] tests/test_pipeline.py — REWRITE (CS domain test cases)
- [ ] docker/Dockerfile — REWRITE
- [ ] docker/docker-compose.yml — REWRITE
- [ ] README.md — Final polish

---

## Complete New File Tree

```
syllaiq/  (renamed from finance-rag-intelligence)
|
|-- .env                          REWRITE (done)
|-- .gitignore                    UPDATED (done)
|-- requirements.txt              REWRITE (done)
|-- README.md                     REWRITE (done)
|
|-- config/
|   |-- __init__.py               NEW (done)
|   `-- settings.py               NEW (done)
|
|-- data/
|   |-- raw/CS_Engineering/       NEW (Phase 2)
|   |-- processed/                KEEP
|   `-- vectorstore/              KEEP
|
|-- scripts/
|   `-- setup_folders.py          NEW (Phase 2)
|
|-- src/
|   |-- __init__.py               KEEP
|   |-- ingestion/
|   |   |-- pdf_loader.py         REWRITE (Phase 3)
|   |   `-- data_cleaner.py       REWRITE (Phase 3)
|   |-- chunking/
|   |   |-- recursive_chunker.py  REWRITE (Phase 4)
|   |   `-- semantic_chunker.py   REWRITE (Phase 4)
|   |-- embedding/
|   |   `-- hf_embedder.py        REWRITE (Phase 5)
|   |-- vectorstore/
|   |   |-- chromadb_store.py     REWRITE (Phase 5)
|   |   `-- faiss_store.py        REWRITE (Phase 5)
|   |-- retrieval/
|   |   |-- dense_retriever.py    REWRITE (Phase 6)
|   |   |-- sparse_retriever.py   REWRITE (Phase 6)
|   |   `-- hybrid_retriever.py   REWRITE (Phase 6)
|   |-- reranking/
|   |   `-- cohere_reranker.py    REWRITE (Phase 6)
|   |-- generation/
|   |   |-- prompt_templates.py   REWRITE (Phase 6)
|   |   |-- llm_chain.py          REWRITE (Phase 6)
|   |   `-- memory.py             REWRITE (Phase 6)
|   |-- tools/
|   |   |-- __init__.py           NEW (Phase 6)
|   |   |-- query_rewriter.py     NEW (Phase 6)
|   |   `-- confidence_scorer.py  NEW (Phase 6)
|   |-- security/
|   |   |-- __init__.py           NEW (Phase 8)
|   |   |-- input_guardrails.py   NEW (Phase 8)
|   |   |-- rate_limiter.py       NEW (Phase 8)
|   |   `-- output_validator.py   NEW (Phase 8)
|   |-- pipeline/
|   |   |-- ingestion_pipeline.py REWRITE (Phase 7)
|   |   |-- rag_pipeline.py       REWRITE (Phase 7)
|   |   `-- self_rag_pipeline.py  REWRITE (Phase 7)
|   `-- evaluation/
|       `-- ragas_eval.py         REWRITE (Phase 10)
|
|-- app/
|   |-- streamlit_app.py          REWRITE (Phase 9)
|   `-- main.py                   REWRITE (Phase 9)
|
|-- tests/
|   `-- test_pipeline.py          REWRITE (Phase 10)
|
|-- notebooks/
|   `-- syllaiq_experiments.ipynb NEW (Phase 9)
|
`-- docker/
    |-- Dockerfile                REWRITE (Phase 10)
    `-- docker-compose.yml        REWRITE (Phase 10)
```

---

## Execution Order

```
Phase 1  →  Project setup + Config          [DONE]
Phase 2  →  Data folder structure
Phase 3  →  Ingestion (PDF → Metadata)
Phase 4  →  Chunking (Metadata-preserving)
Phase 5  →  Embedding + VectorStore
Phase 6  →  Advanced RAG Core
Phase 7  →  Self-RAG Pipeline (LangGraph)
Phase 8  →  Security Layer
Phase 9  →  Streamlit UI
Phase 10 →  Evaluation + Docker + Docs
```

---

## Tech Stack

| Component | Technology |
|---|---|
| RAG Orchestration | LangChain + LangGraph |
| LLM | Groq (LLaMA 3.3 70B) / Gemini 1.5 Flash |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB (primary) / FAISS (fallback) |
| Reranking | Cohere Rerank v3 |
| PDF Parsing | PyPDF + pdfplumber |
| UI | Streamlit |
| API | FastAPI |
| Security | slowapi (rate limiting) |
| Evaluation | RAGAS |
| Containerization | Docker |

---

**Total Files**: 26 Rewrites + 9 New + 2 Deletes = Complete production-grade SyllAIq system
