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
html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.stApp { background:#080b11; color:#f7f8fb; }
section[data-testid="stSidebar"] { width:300px !important; background:#0b0f16; border-right:1px solid #242d3b; }
section[data-testid="stSidebar"] > div { width:300px !important; }
.block-container { max-width:1480px; padding:2.2rem 3rem 5rem; }
.brand { font-size:30px; font-weight:800; letter-spacing:-1.2px; margin-bottom:3px; color:#ffffff; }
.muted { color:#aab4c3; font-size:14px; line-height:1.55; }
.hero { position:relative; overflow:hidden; padding:34px 36px; border:1px solid #303b4d; border-radius:22px;
        background:linear-gradient(135deg,#151d2a 0%,#0e141d 58%,#17132a 100%);
        box-shadow:0 18px 55px rgba(0,0,0,.28); margin-bottom:28px; }
.hero:after { content:""; position:absolute; width:300px; height:300px; right:-90px; top:-140px;
              border-radius:50%; background:rgba(124,58,237,.18); filter:blur(16px); }
.hero h1 { margin:8px 0 0; font-size:44px; line-height:1.12; letter-spacing:-2px; position:relative; z-index:1; color:#ffffff; }
.hero p { color:#c1cad7; margin:12px 0 0; max-width:850px; position:relative; z-index:1; font-size:16px; line-height:1.55; }
.kpi { background:#101720; border:1px solid #2a3647; border-radius:17px; padding:20px; min-height:128px;
       box-shadow:0 10px 28px rgba(0,0,0,.18); transition:.2s; }
.kpi:hover { transform:translateY(-2px); border-color:#53647c; }
.kpi-label { color:#b8c2d0; font-size:13px; text-transform:none; letter-spacing:.01em; font-weight:700; }
.kpi-value { font-size:34px; line-height:1.1; color:#ffffff; font-weight:800; margin-top:10px; letter-spacing:-.8px; }
.kpi-sub { color:#8f9bad; font-size:13px; margin-top:7px; }
.card { background:#0f151e; border:1px solid #2a3647; border-radius:18px; padding:23px; margin-bottom:18px;
        box-shadow:0 12px 30px rgba(0,0,0,.16); }
.card:hover { border-color:#3b4b61; }
.card h3 { margin:0 0 8px; font-size:20px; color:#ffffff; }
.badge { display:inline-flex; align-items:center; gap:5px; padding:7px 11px; border-radius:999px; background:#182231;
         border:1px solid #394a61; font-size:12px; color:#e5eaf1; font-weight:700; }
.live-badge { display:inline-flex; align-items:center; gap:7px; padding:8px 13px; border-radius:999px;
              background:#dc2626; border:1px solid #ef4444; color:#fff; font-size:12px; font-weight:800;
              letter-spacing:.04em; box-shadow:0 7px 22px rgba(220,38,38,.22); }
.health-card { background:#111923; border:1px solid #2d394b; border-radius:15px; padding:16px 17px; margin:8px 0; }
.health-label { color:#9da9b9; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
.health-value { color:#fff; font-size:22px; font-weight:800; margin-top:6px; }
.health-sub { color:#8793a5; font-size:12px; margin-top:3px; }
div[data-testid="stMetric"] { background:#101720; border:1px solid #2a3647; padding:16px; border-radius:16px; }
.stButton > button { border-radius:11px; border:1px solid #3a485c; background:#151e2a; color:#ffffff; font-size:14px; font-weight:700; min-height:48px; transition:.2s; }
.stButton > button:hover { border-color:#9a7cff; background:#1b2636; color:#fff; transform:translateY(-1px); box-shadow:0 8px 20px rgba(124,58,237,.18); }
div[data-testid="stFormSubmitButton"] > button { border-radius:11px; min-height:48px; font-weight:700; }
div[data-testid="stRadio"] > label { color:#aeb8c6 !important; font-size:12px !important; font-weight:700 !important; letter-spacing:.09em; }
div[data-testid="stRadio"] label { display:flex; align-items:center; border-radius:10px; padding:9px 10px; margin:3px 0; color:#e5eaf1 !important; }
div[data-testid="stRadio"] label p { font-size:15px !important; color:#d8dee8 !important; font-weight:600 !important; }
div[data-testid="stRadio"] label:hover { background:#151d28; }
div[data-testid="stRadio"] [aria-checked="true"] { background:#171f2d; border-left:3px solid #8b5cf6; }
div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea { font-size:15px !important; color:#f8fafc !important; }
div[data-baseweb="select"] > div { min-height:44px; }
[data-testid="stDataFrame"] { border:1px solid #2a3647; border-radius:14px; overflow:hidden; }
[data-testid="stCaptionContainer"] p { color:#aab4c3 !important; font-size:13px !important; }
hr { border-color:#263142; }
@media(max-width:900px){
 section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div { width:270px !important; }
 .block-container{padding:1.4rem 1rem 3rem}
 .hero h1{font-size:34px}.hero{padding:25px}
 .kpi-value{font-size:28px}
}
@media(max-width:600px){
 .hero h1{font-size:30px}.hero p{font-size:14px}
 .kpi{min-height:105px;padding:15px}
}
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
        "Overview", "Memory Explorer", "Memory Lifecycle", "Semantic Retrieval", "Memory Intelligence", "Agent Memory", "Drift Analytics",
        "Auto-Heal", "Multi-Agent", "Benchmark", "Reports", "Settings"
    ], label_visibility="visible")
    st.divider()
    st.markdown(f'<div class="card" style="padding:12px 14px;margin:12px 0 8px"><span class="badge">{emoji} {status}</span></div>', unsafe_allow_html=True)
    st.caption(f"● System online  •  Threshold {THRESHOLD:.2f}")
    st.caption(f"Updated: {datetime.now().strftime('%d %b %Y • %H:%M')}")

st.markdown(f"""
<div class="hero">
  <div class="live-badge">● LIVE MEMORYOS CONTROL CENTER</div>
  <h1>{page}</h1>
  <p>AI-agent memory, semantic retrieval, drift detection and self-healing — presented in a modern command-center interface.</p>
</div>
""", unsafe_allow_html=True)

if page == "Overview":
    cols = st.columns(4)
    with cols[0]: kpi("🧠 Memories", len(memories), "Active memory items")
    with cols[1]: kpi("📈 Drift Score", f"{last_score:.2f}", status)
    with cols[2]: kpi("🛠 Heal Events", len(heal_history), "Recovery operations")
    with cols[3]: kpi("◉ Sessions", len(drift_history), "Tracked sessions")

    st.markdown("## Quick actions")
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
        st.markdown('<div class="card"><h3>📈 Drift timeline</h3><div class="muted">Session-by-session behavioral drift. Higher scores indicate greater change from the baseline.</div></div>', unsafe_allow_html=True)
        if drift_history:
            df = pd.DataFrame(drift_history)
            if "session" in df and "drift_score" in df:
                st.line_chart(df.set_index("session")["drift_score"], height=280)
        else:
            st.info("No drift sessions yet. Run Drift Analysis to populate the timeline.")
    with right:
        st.markdown('<div class="card"><h3>💚 System health</h3><div class="muted">Current reliability signals and recovery readiness.</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="health-card"><div class="health-label">Current Status</div><div class="health-value">{emoji} {status}</div><div class="health-sub">Overall memory reliability state</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="health-card"><div class="health-label">Heal Threshold</div><div class="health-value">{THRESHOLD:.2f}</div><div class="health-sub">Auto-heal trigger level</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="health-card"><div class="health-label">Memory Coverage</div><div class="health-value">{len(memories)} items</div><div class="health-sub">Currently available memory records</div></div>', unsafe_allow_html=True)

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

elif page == "Memory Lifecycle":
    st.markdown("### Memory Lifecycle")
    lifecycle = ENGINE.lifecycle()
    if lifecycle:
        df = pd.DataFrame(lifecycle)
        show = [x for x in ["content","importance","access_count","relevance","state"] if x in df.columns]
        st.dataframe(df[show], use_container_width=True, hide_index=True)
        active = sum(x.get("state") == "ACTIVE" for x in lifecycle)
        stale = sum(x.get("state") == "STALE" for x in lifecycle)
        x,y = st.columns(2)
        with x: kpi("Active", active, "healthy memories")
        with y: kpi("Stale", stale, "candidates for pruning")
    else:
        st.info("No memories available.")

elif page == "Semantic Retrieval":
    st.markdown("### Semantic Memory Retrieval")
    q = st.text_input("Ask your memory", placeholder="What do you know about machine learning?")
    limit = st.slider("Results", 1, 10, 5)
    if st.button("Retrieve Context", use_container_width=True):
        result = ENGINE.retrieve(q, limit=limit, min_similarity=0.0)
        if result["results"]:
            for i, item in enumerate(result["results"], 1):
                st.markdown(
                    f'<div class="card"><span class="badge">#{i} • score {item["retrieval_score"]:.3f}</span>'
                    f'<p style="margin:12px 0 0">{item["content"]}</p>'
                    f'<div class="muted">semantic {item["similarity"]:.3f} • importance {item["importance"]:.2f}</div></div>',
                    unsafe_allow_html=True)
            st.text_area("Agent-ready context", ENGINE.context(q, limit), height=140)
        else:
            st.info("No relevant memories found.")

elif page == "Memory Intelligence":
    st.markdown("### Memory Intelligence")
    st.caption("Inspect why memories are retrieved and monitor the current memory-health snapshot.")
    q = st.text_input("Explain a memory query", placeholder="e.g. machine learning")
    if st.button("Analyze Retrieval", use_container_width=True):
        result = ENGINE.explain_retrieval(q, limit=5)
        if result["results"]:
            for i, item in enumerate(result["results"], 1):
                st.markdown(
                    f'<div class="card"><span class="badge">#{i} • {item["reason"]}</span>'
                    f'<h3>{item["content"]}</h3>'
                    f'<div class="muted">semantic similarity: {item["signals"]["semantic_similarity"]:.3f} '
                    f'• importance: {item["signals"]["importance"]:.2f} '
                    f'• combined: {item["signals"]["combined_score"]:.3f}</div></div>',
                    unsafe_allow_html=True)
        else:
            st.info("No matching memories found.")
    st.markdown("### Health snapshot")
    snap = ENGINE.snapshot()
    a,b,c,d = st.columns(4)
    with a: kpi("Memory Health", snap["status"], "latest state")
    with b: kpi("Active", snap["active_memories"], "usable memories")
    with c: kpi("Stale", snap["stale_memories"], "decay candidates")
    with d: kpi("Drift", f'{snap["drift_score"]:.2f}', "latest score")


elif page == "Agent Memory":
    st.markdown("### Agent Memory Console")
    st.caption("Test the same retrieval layer used by the MemoryOS agent before sending a message.")
    prompt = st.text_area("Agent input", placeholder="Ask something related to stored memory...", height=120)
    if st.button("Run Memory Pipeline", use_container_width=True):
        if prompt.strip():
            result = ENGINE.explain_retrieval(prompt, limit=5)
            context = ENGINE.context(prompt, limit=5)
            st.markdown("#### Retrieved memories")
            if result["results"]:
                st.dataframe(
                    pd.DataFrame([{
                        "memory": x["content"],
                        "similarity": x["signals"]["semantic_similarity"],
                        "importance": x["signals"]["importance"],
                        "score": x["signals"]["combined_score"],
                    } for x in result["results"]]),
                    use_container_width=True, hide_index=True)
            else:
                st.info("No matching memory found.")
            st.markdown("#### Context sent to agent")
            st.text_area("Context", context or "No context retrieved.", height=160)
        else:
            st.warning("Enter an agent input first.")

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
            entry = ENGINE.heal(last_score)
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
