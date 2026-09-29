# MemoryOS — Submission & Viva Quick Guide

## 1. One-line explanation
MemoryOS is a self-healing memory reliability layer for AI agents that retrieves relevant memories, detects behavioral drift, identifies stale memory, and performs automatic recovery.

## 2. Demo flow
1. Start the Streamlit dashboard.
2. Open Overview and show the health KPIs.
3. Open Memory Explorer and add a memory with importance.
4. Open Semantic Retrieval and search for a related concept.
5. Open Memory Intelligence to explain the retrieval ranking.
6. Open Agent Memory and run a sample input to show agent-ready context.
7. Run Drift Analysis.
8. If the score crosses 0.45, demonstrate Auto-Heal.
9. Open Memory Lifecycle to show ACTIVE/STALE states.
10. Open Reports and export the health report.

## 3. Viva questions

### What problem does MemoryOS solve?
AI-agent memory can become stale or behavior can drift as sessions change. MemoryOS provides monitoring and a recovery layer around that memory.

### What is drift?
Drift measures how different current agent behavior/context is from a baseline. The prototype uses: drift = 1 - cosine similarity.

### Why 0.45?
0.45 is the configured prototype threshold used to trigger the auto-heal path. It should be described as an experimentally configured project threshold, not a universal industry value.

### How does retrieval work?
The system converts memory text to TF-IDF vectors, calculates cosine similarity with the query, and combines semantic similarity with memory importance to rank results.

### What is memory decay?
Relevance decreases with elapsed sessions using exponential decay. Importance and access count influence the retained-memory decision.

### What does Auto-Heal do?
It calculates memory relevance, marks low-relevance entries stale, prunes stale entries, retains high-value memories, re-injects the top memories through access updates, and records a heal event.

### Why use Streamlit?
It provides a fast interactive monitoring interface for an academic prototype without requiring a separate frontend stack.

### Why JSON instead of Redis/FAISS?
The current implementation is intentionally lightweight and reproducible. Local JSON avoids external infrastructure while TF-IDF provides a dependency-light retrieval baseline.

### What is the future scope?
Production deployment could add a persistent vector database, transformer embeddings, authentication, API services, stronger evaluation datasets, distributed agent memory, and model-provider integration.

## 4. Important presentation rule
Do not claim that Mem0, Redis, FAISS, or an external LLM is currently implemented unless the code is actually connected to it. Present the current system as a working academic prototype with TF-IDF, cosine similarity, JSON persistence, LangGraph prototype components, drift detection, retrieval, lifecycle management, and auto-healing.
