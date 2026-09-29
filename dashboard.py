import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from memory_engine import MemoryEngine

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

BASE = Path(__file__).resolve().parent
MEMORY_FILE = BASE / "memoryos_data.json"
DRIFT_FILE = BASE / "drift_history.json"
HEAL_FILE = BASE / "heal_log.json"
MEMORY_LOG_FILE = BASE / "memory_log.json"
SYNC_FILE = BASE / "sync_bus_log.json"
BENCHMARK_FILE = BASE / "benchmark_results.json"
THRESHOLD = 0.45
ENGINE = MemoryEngine(BASE, drift_threshold=THRESHOLD)

st.set_page_config(
    page_title="MemoryOS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: #07090d; color: #f5f7fb; }
section[data-testid="stSidebar"] { background: #0b0e14; border-right: 1px solid #1d2430; }
.block-container { max-width: 1450px; padding: 2rem 2.5rem 4rem; }
.brand { font-size: 26px; font-weight: 800; letter-spacing: -1px; margin-bottom: 2px; }
.muted { color: #8993a4; font-size: 13px; }
.hero { padding: 24px 28px; border: 1px solid #202938; border-radius: 20px;
        background: linear-gradient(135deg,#101621 0%,#0b0f16 60%,#111827 100%);
        margin-bottom: 22px; }
.hero h1 { margin: 0; font-size: 34px; letter-spacing: -1.5px; }
.hero p { color:#9ba6b7; margin:8px 0 0; }
.kpi { background:#0d121a; border:1px solid #202938; border-radius:16px; padding:18px; min-height:115px; }
.kpi-label { color:#8d98aa; font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
.kpi-value { font-size:29px; font-weight:800; margin-top:8px; }
.kpi-sub { color:#758195; font-size:12px; margin-top:4px; }
.card { background:#0d121a; border:1px solid #202938; border-radius:18px; padding:20px; margin-bottom:16px; }
.card h3 { margin:0 0 6px; font-size:17px; }
.badge { display:inline-block; padding:5px 9px; border-radius:999px; background:#151d2a;
         border:1px solid #2b3748; font-size:11px; color:#cbd5e1; }
div[data-testid="stMetric"] { background:#0d121a; border:1px solid #202938; padding:14px; border-radius:14px; }
.stButton > button { border-radius:10px; border:1px solid #2b3545; background:#121923; color:#fff; font-weight:600; }
.stButton > button:hover { border-color:#7f8da3; }
div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea,
div[data-baseweb="select"] > div { background:#0c1118; border-color:#273243; color:#fff; }
[data-testid="stDataFrame"] { border:1px solid #202938; border-radius:14px; overflow:hidden; }
@media(max-width:768px){
 .block-container{padding:1rem .8rem 3rem}
 .hero h1{font-size:27px}.hero{padding:20px}
 .kpi-value{font-size:23px}
}
</style>
""", unsafe_allow_html=True)

def read_json(path, default):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

def normalize_memories(raw):
    if isinstance(raw, dict):
        for key in ("memories", "memory_store", "data", "items"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
        else:
            raw = []
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, str):
            out.append({"content": item, "importance": 1.0})
        elif isinstance(item, dict):
            text = item.get("content") or item.get("text") or item.get("memory") or ""
            if text:
                out.append({
                    "content": str(text),
                    "importance": float(item.get("importance", 1.0)),
                    "access_count": int(item.get("access_count", 0)),
                    "is_stale": bool(item.get("is_stale", False)),
                })
    return out

def load_memories():
    return normalize_memories(read_json(MEMORY_FILE, []))

def drift_score(texts):
    if len(texts) < 2:
        return 0.0
    if not SKLEARN_OK:
        return 0.0
    try:
        vec = TfidfVectorizer(stop_words="english")
        matrix = vec.fit_transform(texts)
        sim = cosine_similarity(matrix[0:1], matrix[1:]).mean()
        return round(float(max(0.0, 1.0 - sim)), 4)
    except Exception:
        return 0.0

def drift_status(score):
    if score < 0.30: return ("HEALTHY", "🟢")
    if score < 0.50: return ("WARNING", "🟡")
    if score < 0.70: return ("CRITICAL", "🔴")
    return ("DANGER", "🚨")

def save_drift(score, count):
    history = read_json(DRIFT_FILE, [])
    if not isinstance(history, list):
        history = []
    status, _ = drift_status(score)
    history.append({
        "session": len(history) + 1,
        "timestamp": datetime.now().isoformat(),
        "drift_score": score,
        "status": status,
        "inputs_analyzed": count,
    })
    write_json(DRIFT_FILE, history)
    return history[-1]

def run_heal(memories, score):
    before = len(memories)
    session_count = len(read_json(DRIFT_FILE, []))
    kept = []
    pruned = 0
    for m in memories:
        importance = float(m.get("importance", 1.0))
        relevance = importance * np.exp(-0.1 * session_count)
        if relevance < 0.3 and len(memories) > 3:
            pruned += 1
            continue
        m["is_stale"] = False
        kept.append(m)
    kept = sorted(kept, key=lambda x: float(x.get("importance", 1)), reverse=True)
    kept = kept[:max(3, min(len(kept), 50))]
    entry = {
        "session": session_count,
        "timestamp": datetime.now().isoformat(),
        "drift_score_before": score,
        "memories_before": before,
        "memories_pruned": pruned,
        "memories_after": len(kept),
        "reinjected": [m["content"] for m in kept[:3]],
    }
    logs = read_json(HEAL_FILE, [])
    if not isinstance(logs, list): logs = []
    logs.append(entry)
    write_json(HEAL_FILE, logs)
    write_json(MEMORY_FILE, kept)
    return kept, entry

def add_memory(text, importance):
    memories = load_memories()
    memories.append({
        "content": text.strip(),
        "importance": float(importance),
        "access_count": 0,
        "is_stale": False,
    })
    write_json(MEMORY_FILE, memories)
    logs = read_json(MEMORY_LOG_FILE, [])
    if not isinstance(logs, list): logs = []
    logs.append({"timestamp": datetime.now().isoformat(), "action": "add", "content": text.strip()})
    write_json(MEMORY_LOG_FILE, logs)

def reset_demo():
    for path in (DRIFT_FILE, HEAL_FILE, MEMORY_LOG_FILE):
        write_json(path, [])
    write_json(MEMORY_FILE, [
        {"content":"what is machine learning","importance":0.9},
        {"content":"what is deep learning","importance":0.85},
        {"content":"what is neural network","importance":0.8},
        {"content":"what is python programming","importance":0.75},
        {"content":"what is data science","importance":0.7},
    ])

def kpi(label, value, sub=""):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True)

memories = load_memories()
drift_history = read_json(DRIFT_FILE, [])
heal_history = read_json(HEAL_FILE, [])
last_score = float(drift_history[-1].get("drift_score", 0)) if drift_history else 0
status, emoji = drift_status(last_score)

with st.sidebar:
    st.markdown('<div class="brand">🧠 MemoryOS</div><div class="muted">AI Memory Reliability Layer</div>', unsafe_allow_html=True)
    st.divider()
    page = st.radio("NAVIGATION", [
        "Overview", "Memory Explorer", "Drift Analytics",
        "Auto-Heal", "Multi-Agent", "Benchmark", "Reports", "Settings"
    ], label_visibility="visible")
    st.divider()
    st.markdown(f'<span class="badge">{emoji} {status}</span>', unsafe_allow_html=True)
    st.caption(f"Threshold: {THRESHOLD:.2f}")
    st.caption(f"Updated: {datetime.now().strftime('%d %b %Y • %H:%M')}")

st.markdown(f"""
<div class="hero">
  <div class="badge">MEMORY RELIABILITY ENGINE</div>
  <h1>{page}</h1>
  <p>Observe memory health, detect behavioral drift, and recover stale context from one control surface.</p>
</div>
""", unsafe_allow_html=True)

if page == "Overview":
    cols = st.columns(4)
    with cols[0]: kpi("Memories", len(memories), "active memory items")
    with cols[1]: kpi("Drift Score", f"{last_score:.2f}", status)
    with cols[2]: kpi("Heal Events", len(heal_history), "recovery operations")
    with cols[3]: kpi("Sessions", len(drift_history), "tracked sessions")

    st.markdown("### Quick actions")
    a,b,c,d = st.columns(4)
    with a:
        if st.button("🔍 Run Drift Analysis", use_container_width=True):
            sample = [m["content"] for m in memories]
            baseline = sample[:max(1, min(4, len(sample)))]
            current = sample[-max(1, min(4, len(sample))):]
            result = ENGINE.drift(baseline, current)
            st.success(f"Analysis complete • drift {result['score']:.2f} • {result['status']}")
            st.rerun()
    with b:
        if st.button("🛠 Run Auto-Heal", use_container_width=True):
            if last_score >= THRESHOLD:
                entry = ENGINE.heal(last_score)
                st.success(f"Healed • pruned {entry['memories_pruned']} memories")
                st.rerun()
            else:
                st.info("No heal required below threshold.")
    with c:
        if st.button("↻ Refresh", use_container_width=True):
            st.rerun()
    with d:
        if st.button("↺ Reset Demo", use_container_width=True):
            reset_demo()
            st.success("Demo data restored.")
            st.rerun()

    left, right = st.columns([1.5,1])
    with left:
        st.markdown('<div class="card"><h3>Drift timeline</h3><div class="muted">Session-by-session behavioral drift</div></div>', unsafe_allow_html=True)
        if drift_history:
            df = pd.DataFrame(drift_history)
            if "session" in df and "drift_score" in df:
                st.line_chart(df.set_index("session")["drift_score"], height=280)
        else:
            st.info("No drift sessions yet. Run Drift Analysis to populate the timeline.")
    with right:
        st.markdown('<div class="card"><h3>System health</h3><div class="muted">Current reliability signals</div></div>', unsafe_allow_html=True)
        st.metric("Current status", f"{emoji} {status}")
        st.metric("Heal threshold", f"{THRESHOLD:.2f}")
        st.metric("Memory coverage", f"{len(memories)} items")

elif page == "Memory Explorer":
    st.markdown("### Memory Explorer")
    q = st.text_input("Search memory", placeholder="Search by content...")
    min_imp = st.slider("Minimum importance", 0.0, 1.0, 0.0, 0.05)
    filtered = [m for m in ENGINE.search(q, limit=100) if m["importance"] >= min_imp]
    st.caption(f"{len(filtered)} memories shown")
    if filtered:
        df = pd.DataFrame(filtered)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No memories match your filters.")

    st.markdown("### Add memory")
    with st.form("add_memory"):
        text = st.text_area("Memory content", placeholder="e.g. User prefers concise explanations")
        importance = st.slider("Importance", 0.0, 1.0, 0.8, 0.05)
        submitted = st.form_submit_button("Add Memory", use_container_width=True)
        if submitted:
            if text.strip():
                add_memory(text, importance)
                st.success("Memory added.")
                st.rerun()
            else:
                st.warning("Enter memory content first.")

elif page == "Drift Analytics":
    st.markdown("### Drift Analytics")
    a,b,c = st.columns(3)
    with a: kpi("Latest", f"{last_score:.3f}", status)
    with b: kpi("Threshold", f"{THRESHOLD:.2f}", "auto-heal trigger")
    with c: kpi("Sessions", len(drift_history), "observations")
    if drift_history:
        df = pd.DataFrame(drift_history)
        st.line_chart(df.set_index("session")["drift_score"], height=340)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No analytics data yet.")

    st.markdown("### Analyze custom session")
    raw = st.text_area("Enter one input per line", height=150)
    if st.button("Calculate Drift", use_container_width=False):
        lines = [x.strip() for x in raw.splitlines() if x.strip()]
        if len(lines) >= 2:
            score = drift_score(lines)
            s,e = drift_status(score)
            st.success(f"{e} {s} • Drift score: {score:.3f}")
        else:
            st.warning("Add at least two inputs.")

elif page == "Auto-Heal":
    st.markdown("### Auto-Heal Center")
    if last_score >= THRESHOLD:
        st.error(f"Threshold breached: {last_score:.2f} ≥ {THRESHOLD:.2f}")
    else:
        st.success(f"System stable: {last_score:.2f} < {THRESHOLD:.2f}")
    if st.button("🛠 Execute Auto-Heal", use_container_width=True):
        if last_score >= THRESHOLD:
            _, entry = run_heal(memories, last_score)
            st.success(f"Recovery complete. {entry['memories_pruned']} memories pruned.")
            st.rerun()
        else:
            st.info("Auto-heal is not required right now.")
    if heal_history:
        st.markdown("### Recovery history")
        st.dataframe(pd.DataFrame(heal_history), use_container_width=True, hide_index=True)
    else:
        st.info("No heal events recorded.")

elif page == "Multi-Agent":
    st.markdown("### Multi-Agent Sync")
    logs = read_json(SYNC_FILE, [])
    if isinstance(logs, dict):
        logs = logs.get("events", logs.get("messages", []))
    if logs:
        df = pd.DataFrame(logs if isinstance(logs, list) else [])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.success(f"{len(df)} sync events loaded.")
    else:
        st.info("No sync bus events found yet. Run the multi-agent workflow to populate this view.")
    st.markdown('<div class="card"><h3>Shared memory concept</h3><div class="muted">Agents can exchange memory events through the project sync bus. This page surfaces the persisted event stream without changing the underlying agent implementation.</div></div>', unsafe_allow_html=True)

elif page == "Benchmark":
    st.markdown("### Benchmark")
    data = read_json(BENCHMARK_FILE, {})
    if isinstance(data, dict) and data:
        if "results" in data and isinstance(data["results"], list):
            st.dataframe(pd.DataFrame(data["results"]), use_container_width=True, hide_index=True)
        else:
            rows = [{"metric": k, "value": v} for k,v in data.items() if not isinstance(v,(dict,list))]
            if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else: st.json(data)
    else:
        st.info("No benchmark_results.json found. Run the benchmark script first.")
    st.markdown("### Project benchmark files")
    files = sorted([p.name for p in BASE.glob("*benchmark*.json")])
    st.write(files if files else "No benchmark JSON files detected.")

elif page == "Reports":
    st.markdown("### Reports & Export")
    report = {
        "generated_at": datetime.now().isoformat(),
        "memory_count": len(memories),
        "latest_drift": last_score,
        "status": status,
        "drift_sessions": len(drift_history),
        "heal_events": len(heal_history),
    }
    st.json(report)
    st.download_button(
        "⬇ Export Health Report",
        data=json.dumps(report, indent=2),
        file_name="memoryos_health_report.json",
        mime="application/json",
        use_container_width=True,
    )
    pdf = BASE / "MemoryOS_Final_Report.pdf"
    if pdf.exists():
        with open(pdf, "rb") as f:
            st.download_button("📄 Download Final PDF Report", f, file_name=pdf.name, mime="application/pdf", use_container_width=True)

elif page == "Settings":
    st.markdown("### Settings")
    st.metric("Drift threshold", THRESHOLD)
    st.caption("The project AutoHealEngine uses the same 0.45 default threshold.")
    st.markdown("### Data files")
    rows = []
    for p in (MEMORY_FILE, DRIFT_FILE, HEAL_FILE, MEMORY_LOG_FILE, SYNC_FILE, BENCHMARK_FILE):
        rows.append({"file": p.name, "exists": p.exists(), "size_kb": round(p.stat().st_size/1024,2) if p.exists() else 0})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("### Demo controls")
    if st.button("Reset all demo data", type="secondary"):
        reset_demo()
        st.success("Demo data reset.")
        st.rerun()
