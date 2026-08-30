"""
SyllAIq — Centralized Configuration
=====================================
AI-Powered Exam Preparation for RGPV

All project-wide settings, paths, model names, and thresholds.
Import from this module — never hardcode values elsewhere.
"""

from pathlib import Path
from typing import Final

# ─────────────────────────────────────────────────────────────
# Project Identity
# ─────────────────────────────────────────────────────────────
PROJECT_NAME: Final[str]      = "SyllAIq"
PROJECT_TAGLINE: Final[str]   = "AI-Powered Exam Preparation for RGPV"
PROJECT_VERSION: Final[str]   = "0.1.0"
TARGET_UNIVERSITY: Final[str] = "RGPV"
TARGET_BRANCH: Final[str]     = "CSE"

# ─────────────────────────────────────────────────────────────
# Project Root & Paths
# ─────────────────────────────────────────────────────────────
ROOT_DIR: Final[Path] = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR            = ROOT_DIR / "data"
DATA_RAW_PATH       = DATA_DIR / "raw"
DATA_PROCESSED_PATH = DATA_DIR / "processed"
DATA_EVAL_PATH      = DATA_DIR / "evaluation"
VECTORSTORE_PATH    = DATA_DIR / "vectorstore"

# V1 scope: Operating Systems only
OS_SYLLABUS_PATH = DATA_RAW_PATH / "os" / "syllabus"
OS_TEXTBOOK_PATH = DATA_RAW_PATH / "os" / "textbook"
OS_PYQS_PATH     = DATA_RAW_PATH / "os" / "pyqs"

# Processed output files
RAW_DOCUMENTS_JSON = DATA_PROCESSED_PATH / "raw_documents.json"
CHUNKS_JSON        = DATA_PROCESSED_PATH / "chunks.json"
BM25_INDEX_PATH    = DATA_PROCESSED_PATH / "bm25_index.pkl"

# Evaluation files
BENCHMARK_DATASET = DATA_EVAL_PATH / "benchmark_dataset.json"
EVAL_RESULTS_DIR  = DATA_EVAL_PATH / "results"

# ─────────────────────────────────────────────────────────────
# V1 Scope: RGPV CSE — Operating Systems
# ─────────────────────────────────────────────────────────────
V1_SUBJECT      = "Operating Systems"
V1_SUBJECT_CODE = "OS"

# RGPV OS Syllabus Units (Official)
RGPV_OS_UNITS: Final[dict[int, str]] = {
    1: "Introduction to Operating Systems: Function, Evolution, Types, Characteristics, OS Services, Utility Programs, System Calls",
    2: "File Systems: File Concept, Disk/Tape Organization, File System Modules, Disk Space Allocation (Contiguous, Linked, Indexed), Directory Structures, File Protection, System Calls, Disk Scheduling Algorithms",
    3: "CPU Scheduling: Process Concept, Scheduling Concepts, Types of Schedulers, Process State Diagram, Scheduling Algorithms, Algorithm Evaluation, System Calls; Multiple Processor Scheduling; Threads. Memory Management: Partitioning, Swapping, Segmentation, Paging, Paged Segmentation, Overlay, Dynamic Linking/Loading, Virtual Memory, Demand Paging",
    4: "I/O: Principles, I/O Problems, Asynchronous Operations, Speed Gap, I/O Interfaces, Programmed/Interrupt-Driven/Concurrent I/O. Concurrent Processes: Mutual Exclusion, Synchronization, IPC, Critical Section, Semaphores (Binary & Counting), WAIT/SIGNAL. Deadlocks: Characterization, Prevention, Avoidance, Recovery",
    5: "Introduction to Network, Distributed and Multiprocessor OS. Case Studies: Unix/Linux, Windows and Contemporary Operating Systems",
}

# Detailed unit topics for metadata tagging and PYQ mapping
RGPV_OS_UNIT_TOPICS: Final[dict[int, list[str]]] = {
    1: [
        "functions of operating system",
        "evolution of operating system",
        "types of operating system",
        "batch os", "time sharing os", "real time os", "distributed os",
        "characteristics of os",
        "os services",
        "utility programs",
        "system calls",
    ],
    2: [
        "file concept",
        "file system",
        "disk organization",
        "tape organization",
        "contiguous allocation",
        "linked allocation",
        "indexed allocation",
        "disk space allocation",
        "directory structure",
        "file protection",
        "disk scheduling",
        "fcfs disk scheduling",
        "sstf disk scheduling",
        "scan disk scheduling",
        "c-scan disk scheduling",
    ],
    3: [
        "process concept",
        "process state diagram",
        "cpu scheduling",
        "scheduling algorithms",
        "fcfs scheduling",
        "sjf scheduling",
        "round robin scheduling",
        "priority scheduling",
        "multilevel queue scheduling",
        "multiple processor scheduling",
        "threads",
        "memory management",
        "partitioning",
        "swapping",
        "segmentation",
        "paging",
        "paged segmentation",
        "overlay",
        "dynamic linking",
        "virtual memory",
        "demand paging",
        "page replacement algorithms",
        "thrashing",
    ],
    4: [
        "input output",
        "i/o principles",
        "asynchronous operations",
        "programmed io",
        "interrupt driven io",
        "concurrent io",
        "concurrent processes",
        "mutual exclusion",
        "synchronization",
        "inter process communication",
        "ipc",
        "critical section problem",
        "semaphores",
        "binary semaphore",
        "counting semaphore",
        "wait signal operations",
        "deadlock",
        "deadlock characterization",
        "deadlock prevention",
        "deadlock avoidance",
        "bankers algorithm",
        "deadlock detection",
        "deadlock recovery",
    ],
    5: [
        "network operating system",
        "distributed operating system",
        "multiprocessor operating system",
        "unix",
        "linux",
        "windows",
        "contemporary operating systems",
    ],
}

# PYQ year range
PYQ_YEARS: Final[list[int]] = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# ─────────────────────────────────────────────────────────────
# ChromaDB — Separate collections per source type
# ─────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = str(VECTORSTORE_PATH / "chroma")

CHROMA_COLLECTION_TEXTBOOK = "os_textbook"
CHROMA_COLLECTION_PYQS     = "os_pyqs"
CHROMA_COLLECTION_SYLLABUS = "os_syllabus"

ALL_CHROMA_COLLECTIONS: Final[list[str]] = [
    CHROMA_COLLECTION_TEXTBOOK,
    CHROMA_COLLECTION_PYQS,
    CHROMA_COLLECTION_SYLLABUS,
]

# ─────────────────────────────────────────────────────────────
# Embedding Model
# ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION  = 384
EMBEDDING_BATCH_SIZE = 64   # Batch size for efficient embedding

# ─────────────────────────────────────────────────────────────
# Chunking Settings
# ─────────────────────────────────────────────────────────────
TEXTBOOK_CHUNK_SIZE    = 800   # Characters per chunk
TEXTBOOK_CHUNK_OVERLAP = 150   # Overlap between chunks to preserve context
# NOTE: PYQs = 1 question = 1 chunk (never split)
# NOTE: Syllabus = 1 unit = 1 chunk (never split)

# ─────────────────────────────────────────────────────────────
# Retrieval Settings
# ─────────────────────────────────────────────────────────────
DENSE_TOP_K = 20   # Fetch top-20 from ChromaDB before reranking
BM25_TOP_K  = 20   # Fetch top-20 from BM25 before fusion
RRF_K       = 60   # Reciprocal Rank Fusion constant (standard value)
RERANK_TOP_N = 5   # Keep top-5 after Cohere reranking

# NLI grading thresholds
NLI_RELEVANCE_THRESHOLD    = 0.4  # Doc relevance gate
NLI_GROUNDEDNESS_THRESHOLD = 0.6  # Answer groundedness gate

# ─────────────────────────────────────────────────────────────
# LLM Settings
# ─────────────────────────────────────────────────────────────
GROQ_MODEL      = "llama-3.3-70b-versatile"
GEMINI_MODEL    = "gemini-1.5-flash"
LLM_TEMPERATURE = 0.1    # Low = factual, grounded answers
LLM_MAX_TOKENS  = 1024

PRIMARY_LLM  = "groq"
FALLBACK_LLM = "gemini"

# ─────────────────────────────────────────────────────────────
# Self-RAG Settings
# ─────────────────────────────────────────────────────────────
MAX_RETRIEVAL_RETRIES  = 2   # Max query rewrites before giving up
MAX_GENERATION_RETRIES = 2   # Max regeneration attempts

# ─────────────────────────────────────────────────────────────
# Confidence Thresholds
# ─────────────────────────────────────────────────────────────
CONFIDENCE_HIGH_THRESHOLD   = 0.85   # Green ✅
CONFIDENCE_MEDIUM_THRESHOLD = 0.60   # Yellow ⚠️
# Below medium → Low confidence, answer is flagged with warning

# ─────────────────────────────────────────────────────────────
# Intent Labels
# ─────────────────────────────────────────────────────────────
INTENT_CONCEPT_EXPLANATION = "concept"
INTENT_PYQ_RETRIEVAL       = "pyq"
INTENT_TOPIC_IMPORTANCE    = "importance"
INTENT_SYLLABUS_LOOKUP     = "syllabus"
INTENT_UNKNOWN             = "unknown"

ALL_INTENTS: Final[list[str]] = [
    INTENT_CONCEPT_EXPLANATION,
    INTENT_PYQ_RETRIEVAL,
    INTENT_TOPIC_IMPORTANCE,
    INTENT_SYLLABUS_LOOKUP,
    INTENT_UNKNOWN,
]

# ─────────────────────────────────────────────────────────────
# Web Search (Phase 11 — disabled in V1)
# ─────────────────────────────────────────────────────────────
WEB_SEARCH_ENABLED = False

SOURCE_TRUST_TIERS: Final[dict[int, list[str]]] = {
    1: ["rgpv.ac.in", "mp.gov.in"],
    2: ["aicte-india.org", "ugc.gov.in"],
    3: ["nptel.ac.in", "geeksforgeeks.org"],
    4: ["github.com", "stackoverflow.com"],
    5: ["unknown"],
}

# ─────────────────────────────────────────────────────────────
# Security / Rate Limiting
# ─────────────────────────────────────────────────────────────
MAX_REQUESTS_PER_MINUTE = 20
MAX_REQUESTS_PER_HOUR   = 100
MAX_QUERY_LENGTH        = 500  # Characters

INJECTION_PATTERNS: Final[list[str]] = [
    "ignore previous instructions",
    "ignore all instructions",
    "you are now",
    "pretend you are",
    "act as",
    "jailbreak",
    "dan mode",
    "do anything now",
    "disregard your",
    "forget your instructions",
]

OFF_TOPIC_KEYWORDS: Final[list[str]] = [
    "cricket", "movie", "song", "weather", "stock price",
    "recipe", "sports", "celebrity", "news", "politics",
]

# ─────────────────────────────────────────────────────────────
# FastAPI Settings
# ─────────────────────────────────────────────────────────────
API_HOST        = "0.0.0.0"
API_PORT        = 8000
API_PREFIX      = "/api/v1"
API_TITLE       = "SyllAIq API"
API_DESCRIPTION = "AI-Powered Exam Preparation for RGPV Students"

# ─────────────────────────────────────────────────────────────
# Streamlit UI Settings
# ─────────────────────────────────────────────────────────────
APP_TITLE        = "SyllAIq 🎓"
APP_SUBTITLE     = "AI-Powered Exam Preparation for RGPV"
APP_ICON         = "🎓"
MAX_CHAT_HISTORY = 10

# ─────────────────────────────────────────────────────────────
# Database (SQLite for V1 → PostgreSQL later)
# ─────────────────────────────────────────────────────────────
DATABASE_URL = f"sqlite:///{ROOT_DIR / 'data' / 'syllaiq.db'}"

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

# ─────────────────────────────────────────────────────────────
# Evaluation Targets
# ─────────────────────────────────────────────────────────────
EVAL_TARGET_FAITHFULNESS      = 0.80
EVAL_TARGET_ANSWER_RELEVANCY  = 0.75
EVAL_TARGET_CONTEXT_PRECISION = 0.75
EVAL_TARGET_CONTEXT_RECALL    = 0.70
EVAL_TARGET_RECALL_AT_5       = 0.70
EVAL_TARGET_PRECISION_AT_5    = 0.65
EVAL_TARGET_MRR               = 0.65
