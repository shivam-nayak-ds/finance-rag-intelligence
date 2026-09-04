"""Centralized configuration for SyllAIq."""

import os
from pathlib import Path
from typing import Final, Optional

# Project Identity
PROJECT_NAME: Final[str] = "SyllAIq"
PROJECT_TAGLINE: Final[str] = "AI-Powered Exam Preparation for RGPV"
PROJECT_VERSION: Final[str] = "0.1.0"
TARGET_UNIVERSITY: Final[str] = "RGPV"
TARGET_BRANCH: Final[str] = "CSE"

# Project Root & Paths
ROOT_DIR: Final[Path] = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
DATA_RAW_PATH = DATA_DIR / "raw"
DATA_PROCESSED_PATH = DATA_DIR / "processed"
DATA_EVAL_PATH = DATA_DIR / "evaluation"
VECTORSTORE_PATH = DATA_DIR / "vectorstore"

# V1 Scope: Operating Systems
OS_SYLLABUS_PATH = DATA_RAW_PATH / "os" / "syllabus"
OS_TEXTBOOK_PATH = DATA_RAW_PATH / "os" / "textbook"
OS_PYQS_PATH = DATA_RAW_PATH / "os" / "pyqs"

RAW_DOCUMENTS_JSON = DATA_PROCESSED_PATH / "raw_documents.json"
CHUNKS_JSON = DATA_PROCESSED_PATH / "chunks.json"
BM25_INDEX_PATH = DATA_PROCESSED_PATH / "bm25_index.pkl"

BENCHMARK_DATASET = DATA_EVAL_PATH / "benchmark_dataset.json"
EVAL_RESULTS_DIR = DATA_EVAL_PATH / "results"

V1_SUBJECT = "Operating Systems"
V1_SUBJECT_CODE = "OS"

RGPV_OS_UNITS: Final[dict[int, str]] = {
    1: "Introduction to Operating Systems: Function, Evolution, Types, Characteristics, OS Services, Utility Programs, System Calls",
    2: "File Systems: File Concept, Disk/Tape Organization, File System Modules, Disk Space Allocation (Contiguous, Linked, Indexed), Directory Structures, File Protection, System Calls, Disk Scheduling Algorithms",
    3: "CPU Scheduling: Process Concept, Scheduling Concepts, Types of Schedulers, Process State Diagram, Scheduling Algorithms, Algorithm Evaluation, System Calls; Multiple Processor Scheduling; Threads. Memory Management: Partitioning, Swapping, Segmentation, Paging, Paged Segmentation, Overlay, Dynamic Linking/Loading, Virtual Memory, Demand Paging",
    4: "I/O: Principles, I/O Problems, Asynchronous Operations, Speed Gap, I/O Interfaces, Programmed/Interrupt-Driven/Concurrent I/O. Concurrent Processes: Mutual Exclusion, Synchronization, IPC, Critical Section, Semaphores (Binary & Counting), WAIT/SIGNAL. Deadlocks: Characterization, Prevention, Avoidance, Recovery",
    5: "Introduction to Network, Distributed and Multiprocessor OS. Case Studies: Unix/Linux, Windows and Contemporary Operating Systems",
}

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

PYQ_YEARS: Final[list[int]] = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# Vector Store (Qdrant)
QDRANT_PERSIST_DIR: Final[str] = str(VECTORSTORE_PATH / "qdrant")
QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL", None)
QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", None)

QDRANT_COLLECTION_TEXTBOOK: Final[str] = "os_textbook"
QDRANT_COLLECTION_PYQS: Final[str] = "os_pyqs"
QDRANT_COLLECTION_SYLLABUS: Final[str] = "os_syllabus"

ALL_QDRANT_COLLECTIONS: Final[list[str]] = [
    QDRANT_COLLECTION_TEXTBOOK,
    QDRANT_COLLECTION_PYQS,
    QDRANT_COLLECTION_SYLLABUS,
]

# Legacy aliases for backward compatibility
CHROMA_PERSIST_DIR = QDRANT_PERSIST_DIR
CHROMA_COLLECTION_TEXTBOOK = QDRANT_COLLECTION_TEXTBOOK
CHROMA_COLLECTION_PYQS = QDRANT_COLLECTION_PYQS
CHROMA_COLLECTION_SYLLABUS = QDRANT_COLLECTION_SYLLABUS
ALL_CHROMA_COLLECTIONS = ALL_QDRANT_COLLECTIONS

# Embedding Settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 64

# Chunking Settings
TEXTBOOK_CHUNK_SIZE = 800
TEXTBOOK_CHUNK_OVERLAP = 150

# Retrieval Settings
DENSE_TOP_K = 20
BM25_TOP_K = 20
RRF_K = 60
RERANK_TOP_N = 5

NLI_RELEVANCE_THRESHOLD = 0.4
NLI_GROUNDEDNESS_THRESHOLD = 0.6

# LLM Settings
GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-1.5-flash"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

PRIMARY_LLM = "groq"
FALLBACK_LLM = "gemini"

# Self-RAG
MAX_RETRIEVAL_RETRIES = 2
MAX_GENERATION_RETRIES = 2

# Confidence Gates
CONFIDENCE_HIGH_THRESHOLD = 0.85
CONFIDENCE_MEDIUM_THRESHOLD = 0.60

# Query Intents
INTENT_CONCEPT_EXPLANATION = "concept"
INTENT_PYQ_RETRIEVAL = "pyq"
INTENT_TOPIC_IMPORTANCE = "importance"
INTENT_SYLLABUS_LOOKUP = "syllabus"
INTENT_UNKNOWN = "unknown"

ALL_INTENTS: Final[list[str]] = [
    INTENT_CONCEPT_EXPLANATION,
    INTENT_PYQ_RETRIEVAL,
    INTENT_TOPIC_IMPORTANCE,
    INTENT_SYLLABUS_LOOKUP,
    INTENT_UNKNOWN,
]

# Web Search
WEB_SEARCH_ENABLED = False
SOURCE_TRUST_TIERS: Final[dict[int, list[str]]] = {
    1: ["rgpv.ac.in", "mp.gov.in"],
    2: ["aicte-india.org", "ugc.gov.in"],
    3: ["nptel.ac.in", "geeksforgeeks.org"],
    4: ["github.com", "stackoverflow.com"],
    5: ["unknown"],
}

# Rate Limiting & Security
MAX_REQUESTS_PER_MINUTE = 20
MAX_REQUESTS_PER_HOUR = 100
MAX_QUERY_LENGTH = 500

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

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_PREFIX = "/api/v1"
API_TITLE = "SyllAIq API"
API_DESCRIPTION = "Academic Retrieval API for RGPV Exam Preparation"

# UI Settings
APP_TITLE = "SyllAIq"
APP_SUBTITLE = "Exam Preparation Assistant for RGPV CSE"
APP_ICON = None
MAX_CHAT_HISTORY = 10

# Database
DATABASE_URL = f"sqlite:///{ROOT_DIR / 'data' / 'syllaiq.db'}"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

# Evaluation Thresholds
EVAL_TARGET_FAITHFULNESS = 0.80
EVAL_TARGET_ANSWER_RELEVANCY = 0.75
EVAL_TARGET_CONTEXT_PRECISION = 0.75
EVAL_TARGET_CONTEXT_RECALL = 0.70
EVAL_TARGET_RECALL_AT_5 = 0.70
EVAL_TARGET_PRECISION_AT_5 = 0.65
EVAL_TARGET_MRR = 0.65
