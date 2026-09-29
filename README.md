# memoryos-project
Memory - Self-Healing Memory Architecture for AI Agents

# 🧠 MemoryOS for AI Agents

> Self-Healing Memory Architecture with Real-Time Drift Detection

[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-green)](https://langgraph.com)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🚨 The Problem

AI agents deployed in production **silently fail** after 1-2 weeks.
- **65%** of failures caused by memory loss — not model capability
- **$800B** market problem by 2031 (Gartner)
- **0** open-source tools exist for agent memory health monitoring

This phenomenon is called **Context Rot**.

---

## ✅ The Solution — MemoryOS

MemoryOS is a lightweight monitoring, retrieval, lifecycle-management, drift-detection, and self-healing layer for AI-agent memory.

> Just as an OS manages RAM for a computer — MemoryOS manages memory for AI agents.

---

## 🏆 Benchmark Results

| Metric | WITHOUT MemoryOS | WITH MemoryOS |
|---|---|---|
| Average Drift Score | High | **90.9% lower** |
| Consistency | Baseline | **+64.7% better** |
| Heal Events | 0 | Auto-triggered |

---

## 🏗️ System Architecture

```
User Input
    ↓
LangGraph Agent
    ↓
Memory Retrieval + Lifecycle Layer (TF-IDF + JSON)
    ↓
Drift Detection Engine (cosine similarity)
    ↓
Decision: drift < 0.45 → Continue
          drift ≥ 0.45 → Auto-Heal Module
    ↓
Health Dashboard (Streamlit)
```

---

## 📁 Project Structure

```
memoryos-project/
├── agent.py            # Week 1: LangGraph agent
├── drift_detector.py   # Week 1: Drift detection engine
├── auto_heal.py        # Week 1: Auto-heal module
├── embeddings.py       # Week 1: TF-IDF embeddings
├── memoryos.py         # Week 1: Integrated system
├── dashboard.py        # Week 2: Streamlit dashboard
├── calibration.py      # Week 2: Threshold calibration
├── multi_agent.py      # Week 2: Multi-agent sync
├── benchmark.py        # Week 2: WITH vs WITHOUT
└── README.md           # This file
```

---

## ⚙️ Core Algorithms

**Drift Score:**
```
drift(t) = 1 - cosine_similarity(v_baseline, v_current)
```

**Memory Decay:**
```
relevance(m, t) = importance(m) × e^(-λ × Δt)
```

**Auto-Heal Trigger:**
```
if drift(t) ≥ θ (0.45):
    prune stale memories
    re-inject top-3 important memories
    log heal event
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph |
| Memory Store | Local JSON persistence |
| Embeddings | TF-IDF (scikit-learn) |
| Vector Index | TF-IDF / cosine similarity |
| Dashboard | Streamlit + Plotly |
| Version Control | Git + GitHub |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/shivampoddar12/memoryos-project.git
cd memoryos-project

# 2. Setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 3. Run Agent
python memoryos.py

# 4. Run Dashboard
python -m streamlit run dashboard.py
```

---

## 📚 References

- MemGPT (arXiv:2310.08560)
- Mem0 (arXiv:2504.19413)
- Agent Drift (arXiv:2601.04170)
- Zep (arXiv:2501.13956)

---

## 👨‍💻 Authors

- **Aaditya Rana** (2303051240001)
- **Shivam Kumar Poddar** (2303051240374)

B.Tech CSE-AI | Parul University | 2025-26

Supervisor: Ms. Kiran Sharma


## Current Dashboard

The Streamlit dashboard includes Overview, Memory Explorer, Memory Lifecycle, Semantic Retrieval, Memory Intelligence, Agent Memory, Drift Analytics, Auto-Heal, Multi-Agent, Benchmark, Reports, and Settings.

## Implementation note

The current repository is a lightweight academic prototype. It does not require external Mem0, Redis, or FAISS services; the implemented retrieval layer uses scikit-learn TF-IDF and cosine similarity with local JSON persistence.
