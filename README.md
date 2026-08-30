# 🎓 SyllAIq — AI-Powered Exam Preparation for RGPV

> **AI-Powered Exam Preparation for RGPV CSE Students**
> *Concept explanations • PYQ retrieval • Topic importance • Source-cited answers*

---

## 🎯 What is SyllAIq?

SyllAIq is a **production-grade Agentic RAG system** built specifically for **RGPV CSE students**.

It solves a real problem:

> RGPV students have syllabus, textbooks, and PYQs scattered across different places. They don't know what to study, which topics are important, or whether the information they received is trustworthy.

SyllAIq answers complex exam questions by:
- Retrieving knowledge from **real RGPV textbooks, syllabus, and PYQs**
- Providing **source citations** (book, chapter, page)
- **Self-verifying** every answer before returning it
- Showing **PYQ frequency analysis** — which topics appear most in exams
- Being **honest about uncertainty** — never fabricating answers

---

## 💡 Example Queries

```
"Deadlock ke 4 necessary conditions kya hain?"
→ Answer from: Galvin OS Ch.7, Page 318 | Confidence: 94% ✅

"RGPV mein deadlock ke PYQs dikhao"
→ 6 questions found (2018–2024) with year, marks, unit

"OS ke sabse important topics kya hain?"
→ Deadlock (7/7 years), Process Scheduling (6/7), Memory Management (5/7)

"Unit 3 mein kya hai RGPV OS syllabus mein?"
→ RGPV OS Unit 3: Deadlock — conditions, prevention, avoidance, detection
```

---

## 🏗️ Architecture

```
Student Query
      │
      ▼
  FastAPI (input validation + rate limiting)
      │
      ▼
  LangGraph State Machine
  ┌─────────────────────────────────────┐
  │  analyze_query → classify_intent   │
  │         ↓                          │
  │     route_query                    │
  │   ┌──────┬──────┬──────┐           │
  │ concept  pyq  import syllabus      │
  │         ↓                          │
  │   Hybrid Retrieval                 │
  │   Dense (ChromaDB)                 │
  │   + BM25                           │
  │   + RRF Fusion                     │
  │         ↓                          │
  │   grade_documents (NLI)            │
  │   ├── relevant → rerank → generate │
  │   └── irrelevant → rewrite → retry │
  │         ↓                          │
  │   verify_answer (Self-RAG)         │
  │   ├── grounded → finalize          │
  │   └── hallucination → retry        │
  │         ↓                          │
  │   Final Answer + Citations         │
  └─────────────────────────────────────┘
      │
      ▼
  Streamlit UI
```

---

## 📚 V1 Knowledge Base (Operating Systems)

| Source | Content |
|---|---|
| RGPV OS Syllabus | Official unit-wise syllabus |
| Galvin OS 10th Ed | Primary textbook |
| RGPV OS PYQs 2018–2024 | 7 years of previous year questions |

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|---|---|---|
| Orchestration | LangGraph | Explicit state machine, debuggable |
| LLM | Groq (LLaMA 3.3 70B) | Free tier, fast, Hinglish capable |
| Embeddings | sentence-transformers/MiniLM | Free, fast, 384-dim |
| Vector DB | ChromaDB → pgvector | Local first, migrate later |
| Sparse Retrieval | BM25 (rank-bm25) | Keyword matching for PYQs |
| Reranking | Cohere Rerank v3 | Best API reranker |
| NLI Grading | cross-encoder/nli-deberta | Groundedness check |
| API | FastAPI | Async, Pydantic, auto-docs |
| UI | Streamlit | Quick, student-friendly |
| Database | SQLite → PostgreSQL | Structured PYQ data |
| Deployment | Docker + Railway/Render | Simple, affordable |

---

## 🚀 Project Status

| Phase | Description | Status |
|---|---|---|
| Phase 0 | Project setup & data inventory | 🔄 In Progress |
| Phase 1 | PDF ingestion + metadata | ⏳ Pending |
| Phase 2 | Chunking strategy | ⏳ Pending |
| Phase 3 | Embeddings + ChromaDB | ⏳ Pending |
| Phase 4 | Basic RAG (dense) | ⏳ Pending |
| Phase 5 | BM25 + Hybrid + RRF | ⏳ Pending |
| Phase 6 | Cohere Reranking + NLI | ⏳ Pending |
| Phase 7 | Evaluation baseline | ⏳ Pending |
| Phase 8 | LangGraph Agentic Routing | ⏳ Pending |
| Phase 9 | Self-RAG verification | ⏳ Pending |
| Phase 10 | PYQ Intelligence | ⏳ Pending |
| Phase 11 | Web Search | ⏳ Pending |
| Phase 12 | FastAPI + Streamlit UI | ⏳ Pending |
| Phase 13–16 | Observability, Security, Docker | ⏳ Pending |

---

## 📁 Project Structure

```
finance-rag-intelligence/   ← (will rename to syllaiq)
│
├── agents/                 ← LangGraph state machine nodes
│   ├── state.py
│   ├── graph.py
│   └── nodes/
├── api/                    ← FastAPI routes
├── retrieval/              ← Dense, BM25, RRF, Hybrid
├── ingestion/              ← PDF loader, cleaner, tagger
├── chunking/               ← Chunking strategies
├── embedding/              ← HuggingFace embedder
├── vectorstore/            ← ChromaDB operations
├── generation/             ← Prompts + LLM chain
├── models/                 ← Pydantic schemas
├── services/               ← PYQ analyzer
├── evaluation/             ← RAGAS + retrieval metrics
├── database/               ← SQLAlchemy models
├── security/               ← Input/output validation
├── tools/                  ← Web search, query rewriter
├── utils/                  ← Logger, tracer
├── config/                 ← Centralized settings
├── data/
│   ├── raw/os/             ← PDFs (not in git)
│   ├── processed/          ← Chunks JSON (not in git)
│   └── evaluation/         ← Benchmark dataset ✅
├── tests/
├── docs/
└── docker/
```

---

## ⚡ Quick Start

```bash
# 1. Clone & enter project
git clone https://github.com/yourusername/syllaiq
cd syllaiq

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 5. Add your data
# Place PDFs in data/raw/os/textbook/, data/raw/os/pyqs/, data/raw/os/syllabus/

# 6. Build the index (Phase 3+)
python scripts/build_index.py

# 7. Run the API
uvicorn api.main:app --reload

# 8. Run the UI
streamlit run app/streamlit_app.py
```

---

## 📊 Evaluation Targets

| Metric | Target |
|---|---|
| Recall@5 | ≥ 0.70 |
| Precision@5 | ≥ 0.65 |
| MRR | ≥ 0.65 |
| Faithfulness (RAGAS) | ≥ 0.80 |
| Answer Relevancy | ≥ 0.75 |

---

## 🙋 About

Built for RGPV CSE students. Focused on depth over breadth.
V1 covers Operating Systems only — done right, with measurable quality.

---

*SyllAIq — Because students deserve better than a generic chatbot.*
