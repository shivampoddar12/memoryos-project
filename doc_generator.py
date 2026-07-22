# doc_generator.py — MemoryOS Week 2 Day 5
# Auto Documentation Generator

import json
import os
from datetime import datetime

print("📝 MemoryOS — Auto Documentation Generator")
print("="*50)

# ── READ ALL JSON RESULTS ─────────────────────────────
def load_json(filename):
    if os.path.exists(filename):
        with open(filename) as f:
            return json.load(f)
    return None

drift_data   = load_json("drift_history.json")
heal_data    = load_json("heal_log.json")
bench_data   = load_json("benchmark_results.json")
calib_data   = load_json("calibration_report.json")
sync_data    = load_json("sync_bus_log.json")
memory_data  = load_json("memory_log.json")

print("\n📂 Files found:")
for fname, data in [
    ("drift_history.json",    drift_data),
    ("heal_log.json",         heal_data),
    ("benchmark_results.json",bench_data),
    ("calibration_report.json",calib_data),
    ("sync_bus_log.json",     sync_data),
    ("memory_log.json",       memory_data),
]:
    status = "✅" if data else "❌"
    print(f"   {status} {fname}")

# ── GENERATE SUMMARY ─────────────────────────────────
print("\n\n📊 Generating Project Summary...")
print("-"*50)

summary = {
    "project": "MemoryOS for AI Agents",
    "generated_at": datetime.now().isoformat(),
    "authors": ["Aaditya Rana (2303051240001)", "Shivam Kumar Poddar (2303051240374)"],
    "supervisor": "Ms. Kiran Sharma",
    "university": "Parul University, CSE-AI",
}

# Drift stats
if drift_data:
    scores = [d.get("score", d.get("drift_score", 0)) for d in drift_data]
    summary["drift_analysis"] = {
        "total_sessions": len(scores),
        "average_drift": round(sum(scores)/len(scores), 4),
        "max_drift": round(max(scores), 4),
        "min_drift": round(min(scores), 4),
        "healthy_sessions": sum(1 for s in scores if s < 0.45),
    }
    print(f"✅ Drift Analysis: {len(scores)} sessions analyzed")

# Heal stats
if heal_data:
    summary["heal_analysis"] = {
        "total_heals": len(heal_data),
        "heal_sessions": [h.get("session", "?") for h in heal_data],
    }
    print(f"✅ Heal Analysis: {len(heal_data)} heal events")

# Benchmark stats
if bench_data:
    imp = bench_data.get("improvement", {})
    summary["benchmark"] = {
        "drift_reduction": f"{imp.get('drift_reduction_percent', 0)}%",
        "consistency_gain": f"{imp.get('consistency_gain_percent', 0)}%",
        "sessions_tested": bench_data.get("sessions_tested", 0),
        "threshold_used": bench_data.get("threshold_used", 0.45),
    }
    print(f"✅ Benchmark: {imp.get('drift_reduction_percent')}% drift reduction")

# Calibration stats
if calib_data:
    rec = calib_data.get("recommendation", {})
    summary["calibration"] = {
        "optimal_threshold": rec.get("optimal_threshold", 0.45),
        "thresholds_tested": len(calib_data.get("threshold_analysis", [])),
        "scenarios_tested": len(calib_data.get("scenario_scores", {})),
    }
    print(f"✅ Calibration: θ = {rec.get('optimal_threshold', 0.45)} optimal")

# Multi-agent stats
if sync_data:
    summary["multi_agent"] = {
        "connected_agents": len(sync_data.get("connected_agents", [])),
        "knowledge_items": len(sync_data.get("knowledge_graph", {})),
        "sync_events": len(sync_data.get("sync_log", [])),
    }
    print(f"✅ Multi-Agent: {len(sync_data.get('connected_agents', []))} agents connected")

# Memory stats
if memory_data:
    summary["memory_analysis"] = {
        "total_memories": len(memory_data),
        "first_session": memory_data[0].get("time", "?")[:10] if memory_data else "?",
        "last_session": memory_data[-1].get("time", "?")[:10] if memory_data else "?",
    }
    print(f"✅ Memory: {len(memory_data)} entries logged")

# ── GENERATE REPORT TEXT ─────────────────────────────
print("\n\n📄 Generating Text Report...")
print("-"*50)

report_text = f"""
MemoryOS for AI Agents — Project Summary Report
================================================
Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Authors: Aaditya Rana, Shivam Kumar Poddar
University: Parul University | CSE-AI | 2025-26
Supervisor: Ms. Kiran Sharma

EXECUTIVE SUMMARY
-----------------
MemoryOS is a self-healing memory architecture for LLM-based AI agents.
It monitors agent memory health in real-time, detects Context Rot via
cosine similarity drift scoring, and automatically heals degraded memory
without requiring agent restart.

KEY RESULTS
-----------
"""

if bench_data:
    imp = bench_data.get("improvement", {})
    report_text += f"""
Benchmark (WITH vs WITHOUT MemoryOS):
  • Drift Reduction:    {imp.get('drift_reduction_percent', 0)}%
  • Consistency Gain:  +{imp.get('consistency_gain_percent', 0)}%
  • Sessions Tested:   {bench_data.get('sessions_tested', 0)}
  • Heal Events:       {bench_data.get('with_memoryos', {}).get('heal_events', 0)}
"""

if calib_data:
    rec = calib_data.get("recommendation", {})
    report_text += f"""
Threshold Calibration:
  • Optimal θ:         {rec.get('optimal_threshold', 0.45)}
  • Thresholds Tested: {len(calib_data.get('threshold_analysis', []))}
  • Scenarios Tested:  {len(calib_data.get('scenario_scores', {}))}
"""

if sync_data:
    report_text += f"""
Multi-Agent Sync:
  • Agents Connected:  {len(sync_data.get('connected_agents', []))}
  • Knowledge Items:   {len(sync_data.get('knowledge_graph', {}))}
  • Sync Events:       {len(sync_data.get('sync_log', []))}
"""

report_text += f"""
MODULES BUILT
-------------
Week 1: agent.py, drift_detector.py, auto_heal.py, embeddings.py, memoryos.py
Week 2: dashboard.py, calibration.py, multi_agent.py, benchmark.py

TECH STACK
----------
Python 3.12 | LangGraph | scikit-learn | Streamlit | Plotly | Git

REFERENCES
----------
[1] MemGPT (arXiv:2310.08560)
[2] Mem0 (arXiv:2504.19413)
[3] Agent Drift (arXiv:2601.04170)
[4] Zep (arXiv:2501.13956)

GitHub: github.com/shivampoddar12/memoryos-project
"""

# Save everything
with open("project_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

with open("project_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print(report_text)
print("💾 project_summary.json saved!")
print("💾 project_report.txt saved!")
print("\n✅ Week 2 Day 5 — Documentation Complete!")