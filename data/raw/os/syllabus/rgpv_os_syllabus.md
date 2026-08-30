# RGPV CSE Operating Systems — Official Syllabus
**University**: Rajiv Gandhi Proudyogiki Vishwavidyalaya (RGPV), Bhopal  
**Course / Branch**: B.Tech / B.E. Computer Science & Engineering (CSE)  
**Subject**: Operating Systems (CS-405 / CS-503)

---

## UNIT 1: Introduction to Operating Systems
- **Function and Evolution**: Function of an Operating System, Evolution of Operating Systems, Different Types (Batch, Time-sharing, Real-Time, Distributed, Multiprocessor).
- **Characteristics**: Desirable Characteristics and features of an O/S.
- **Operating Systems Services**: Types of Services, Different ways of providing these Services – Utility Programs, System Calls (Types, Parameter passing, Execution).

---

## UNIT 2: File Systems & Storage Management
- **File Concept**: User’s and System Programmer’s view of File System, File attributes, operations, and file types.
- **Storage Organization**: Disk Organization, Tape Organization, Different Modules of a File System.
- **Disk Space Allocation Methods**: Contiguous Allocation, Linked Allocation, Indexed Allocation (Comparative analysis).
- **Directory & Protection**: Directory Structures (Single-level, Two-level, Tree, Acyclic Graph), File Protection and Access Control, System Calls for File Management.
- **Disk Scheduling Algorithms**: FCFS, SSTF, SCAN, C-SCAN, LOOK, C-LOOK.

---

## UNIT 3: CPU Scheduling & Memory Management
- **Process & CPU Scheduling**:
  - Process Concept, Process State Diagram, Process Control Block (PCB).
  - Scheduling Concepts, Types of Schedulers (Long-term, Short-term, Medium-term).
  - Scheduling Algorithms (FCFS, SJF Preemptive/Non-preemptive, Priority, Round Robin, Multilevel Queue, Multilevel Feedback Queue).
  - Algorithms Evaluation criteria & methods, System calls for Process Management.
  - Multiple Processor Scheduling, Concept of Threads (User-level vs Kernel-level).
- **Memory Management**:
  - Memory Management Techniques: Partitioning (Fixed & Dynamic), Swapping, Segmentation, Paging, Paged Segmentation, Comparison of memory management techniques.
  - Large Program Execution: Overlay, Dynamic Linking and Loading.
  - Virtual Memory: Concept, Implementation by Demand Paging, Page Fault handling, Page Replacement Algorithms (FIFO, Optimal, LRU, LFU), Thrashing & Working Set Model.

---

## UNIT 4: I/O Management, Concurrent Processes & Deadlocks
- **Input / Output**:
  - Principles and Programming of I/O, Input/Output Problems, Asynchronous Operations, Speed gap, Format conversion.
  - I/O Interfaces, Programmed Controlled I/O, Interrupt Driven I/O, Direct Memory Access (DMA), Concurrent I/O.
- **Concurrent Processes**:
  - Real and Virtual Concurrency, Mutual Exclusion, Synchronization, Inter-Process Communication (IPC: Shared Memory, Message Passing).
  - Critical Section Problem & Requirements (Mutual Exclusion, Progress, Bounded Waiting).
  - Solutions to Critical Section: Semaphores (Binary & Counting Semaphores), WAIT & SIGNAL Operations and their implementation, Classical Synchronization Problems (Producer-Consumer, Reader-Writer, Dining Philosophers).
- **Deadlocks**:
  - Deadlock Problems, Characterization (4 Necessary Conditions: Mutual Exclusion, Hold & Wait, No Preemption, Circular Wait).
  - Resource Allocation Graph (RAG).
  - Deadlock Handling: Deadlock Prevention, Deadlock Avoidance (Banker's Algorithm), Deadlock Detection, Deadlock Recovery.

---

## UNIT 5: Distributed Systems & Case Studies
- **Advanced Operating Systems**:
  - Introduction to Network Operating Systems.
  - Introduction to Distributed Operating Systems (Architecture, Design issues).
  - Introduction to Multiprocessor Operating Systems.
- **Case Studies**:
  - Unix / Linux Operating System (Architecture, Process management, File system, Kernel modules).
  - Windows Operating System (Architecture, Process & Thread management, Memory model, NTFS).
  - Other Contemporary Operating Systems.
