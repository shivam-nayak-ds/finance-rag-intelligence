# 📦 SyllAIq — Data Inventory & Catalog

**Target Domain**: RGPV University CSE — Operating Systems (V1 Scope)  
**Catalog Status**: Complete for Syllabus & PYQs (5 Years of authentic papers)

---

## 1. 📋 Syllabus Layer (`data/raw/os/syllabus/`)

| File | Type | Source | Unit Coverage | Format |
|---|---|---|---|---|
| [`rgpv_os_syllabus.txt`](file:///c:/finance-rag-intelligence/data/raw/os/syllabus/rgpv_os_syllabus.txt) | Official Syllabus Text | RGPV Academics | Unit 1 – 5 | Plain Text |
| [`rgpv_os_syllabus.md`](file:///c:/finance-rag-intelligence/data/raw/os/syllabus/rgpv_os_syllabus.md) | Formatted Markdown | RGPV Academics | Unit 1 – 5 | Markdown |
| [`rgpv_os_syllabus.json`](file:///c:/finance-rag-intelligence/data/raw/os/syllabus/rgpv_os_syllabus.json) | Structured Units & Topics | RGPV Academics | Unit 1 – 5 | JSON |

### Unit Breakdown:
- **Unit 1**: Introduction to OS, Evolution, Types, Characteristics, OS Services, System Calls.
- **Unit 2**: File Systems, Disk/Tape Organization, Contiguous/Linked/Indexed Allocation, Directory Structures, Disk Scheduling (FCFS, SSTF, SCAN, C-SCAN, LOOK).
- **Unit 3**: CPU Scheduling (FCFS, SJF, SRTF, RR, Priority, Multi-Queue), Process States, PCB, Threads, Memory Management (Partitioning, Paging, Segmentation, Inverted Page Tables), Virtual Memory (Demand Paging, Page Faults, Page Replacement: FIFO/LRU/Optimal, Thrashing, Working Set).
- **Unit 4**: I/O Principles, Programmed/Interrupt/DMA, Concurrency, Mutual Exclusion, Critical Section, Semaphores (Binary & Counting), Classical Synchronization (Producer-Consumer, Readers-Writers, Dining Philosophers), Deadlocks (4 Conditions, RAG, Banker's Algorithm, Prevention, Detection, Recovery).
- **Unit 5**: Distributed OS, Network OS, Failures in Distributed Systems, Case Studies: Unix/Linux and Windows OS.

---

## 2. 📜 Previous Year Question Papers (`data/raw/os/pyqs/`)

| Paper ID | Session | Code | Max Marks | Total Questions | Files |
|---|---|---|:---:|:---:|---|
| `RGPV_OS_JUN_2023` | June 2023 | AD/CD/CS-405 (GS) | 70 | 15 items | [JSON](file:///c:/finance-rag-intelligence/data/raw/os/pyqs/rgpv_os_pyqs_dataset.json), [MD](file:///c:/finance-rag-intelligence/data/raw/os/pyqs/rgpv_os_pyqs_all_years.md) |
| `RGPV_OS_NOV_2023` | Nov 2023 | AD/CD/CS-405 (GS) | 70 | 15 items | [JSON](file:///c:/finance-rag-intelligence/data/raw/os/pyqs/rgpv_os_pyqs_dataset.json), [MD](file:///c:/finance-rag-intelligence/data/raw/os/pyqs/rgpv_os_pyqs_all_years.md) |
| `RGPV_OS_JUN_2024` | June 2024 | AD/CD/CS/SD-405 (GS) | 70 | 16 items | [JSON](file:///c:/finance-rag-intelligence/data/raw/os/pyqs/rgpv_os_pyqs_dataset.json), [MD](file:///c:/finance-rag-intelligence/data/raw/os/pyqs/rgpv_os_pyqs_all_years.md) |
| `RGPV_OS_JUN_2025` | June 2025 | AD/CD/CS/SD-405 | 70 | 16 items | [JSON](file:///c:/finance-rag-intelligence/data/raw/os/pyqs/rgpv_os_pyqs_dataset.json), [MD](file:///c:/finance-rag-intelligence/data/raw/os/pyqs/rgpv_os_pyqs_all_years.md) |
| `RGPV_OS_JUN_2026` | June 2026 | CS405(GS) | 70 | 16 items | [JSON](file:///c:/finance-rag-intelligence/data/raw/os/pyqs/rgpv_os_pyqs_dataset.json), [MD](file:///c:/finance-rag-intelligence/data/raw/os/pyqs/rgpv_os_pyqs_all_years.md) |

**Total Transcribed Authentic Questions**: **78 Question Items** with bilingual English/Hindi text, mapped to specific Units and Topics.

---

## 3. 📚 Authoritative Textbook Layer (`data/raw/os/textbook/`)
- Target: *Operating System Concepts* — Silberschatz, Galvin, Gagne (10th Edition).
- Ready for chunking and ingestion into ChromaDB collection `os_textbook`.
