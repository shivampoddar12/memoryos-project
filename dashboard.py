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
SERVER_IMAGE = "https://images.unsplash.com/photo-1695668548342-c0c1ad479aee?auto=format&fit=crop&fm=jpg&q=80&w=1400"
WORK_IMAGE = "https://images.unsplash.com/photo-1553484771-cc0d9b8c2b33?auto=format&fit=crop&fm=jpg&q=80&w=1400"
HERO_IMAGE = "https://images.unsplash.com/photo-1694261321131-8157dce8e288?auto=format&fit=crop&fm=jpg&q=80&w=1800"

st.set_page_config(
    page_title="MemoryOS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.stApp{background:#080f1c;color:#f6f7f9}
section[data-testid="stSidebar"]{width:285px!important;background:#0a1423;border-right:1px solid #1c2a3d}
section[data-testid="stSidebar"]>div{width:285px!important}
.block-container{max-width:1440px;padding:1.5rem 3.2rem 5rem}
/* Streamlit chrome reset: keep the app itself flush to the top. */
header[data-testid="stHeader"]{background:transparent!important;height:0!important;min-height:0!important}
div[data-testid="stToolbar"]{display:none!important}
div[data-testid="stDecoration"]{display:none!important}
#MainMenu{visibility:hidden!important}
footer{visibility:hidden!important}
.stAppViewContainer{padding-top:0!important}
.main .block-container{padding-top:1rem!important}

.desktop-nav{display:block;margin:0 0 24px;padding:12px 16px;border:1px solid #263952;border-radius:16px;background:linear-gradient(180deg,#101d2e,#0c1726);box-shadow:0 10px 28px rgba(0,0,0,.20)}
.desktop-nav-label{color:#718197;font-size:9px;font-weight:800;letter-spacing:.16em;margin:0 0 9px 3px}
.desktop-nav [data-testid="stRadio"]{padding:0!important;margin:0!important}
.desktop-nav [data-testid="stRadio"]>label{display:none!important}
.desktop-nav [role="radiogroup"]{gap:5px!important;flex-wrap:nowrap!important;overflow-x:auto!important;scrollbar-width:none!important;padding-bottom:1px!important}
.desktop-nav [role="radiogroup"]::-webkit-scrollbar{display:none!important}
.desktop-nav [role="radio"]{
  flex:0 0 auto!important;display:inline-flex!important;align-items:center!important;
  border-radius:9px!important;padding:8px 12px!important;
  color:#aebacc!important;font-size:11px!important;font-weight:700!important;
  border:1px solid transparent!important;background:transparent!important;
  transition:all .18s ease!important;white-space:nowrap!important;
}
.desktop-nav [role="radio"]>div:first-child{display:none!important}
.desktop-nav [role="radio"] p{color:inherit!important;font-size:11px!important;font-weight:700!important;margin:0!important}
.desktop-nav [role="radio"]:hover{background:#17263a!important;color:#fff!important;border-color:#2b4059!important}
.desktop-nav [role="radio"][aria-checked="true"]{background:#f4774b!important;color:#fff!important;border-color:#ff9a72!important;box-shadow:0 5px 14px rgba(244,119,75,.20)!important}
@media(min-width:901px){section[data-testid="stSidebar"]{display:none!important}.desktop-nav{display:block}}
@media(max-width:900px){.desktop-nav{display:none!important}}

.brand{font-size:28px;font-weight:800;letter-spacing:-1.2px;color:#fff;margin-bottom:2px}
.muted{color:#a7b1c0;font-size:13px;line-height:1.55}
.hero{position:relative;overflow:hidden;padding:38px 46px;min-height:225px;border:1px solid #22334a;border-radius:24px;
background-image:linear-gradient(90deg,rgba(7,14,25,.97) 0%,rgba(7,14,25,.86) 42%,rgba(7,14,25,.35) 100%),url('https://images.unsplash.com/photo-1694261321131-8157dce8e288?auto=format&fit=crop&fm=jpg&q=80&w=1800');
background-size:cover;background-position:center;
box-shadow:0 24px 60px rgba(0,0,0,.28);margin-bottom:28px}
.hero:before{content:"";position:absolute;width:430px;height:430px;right:-180px;top:-210px;border-radius:50%;border:1px solid rgba(246,126,74,.16)}
.hero h1{margin:18px 0 0;font-size:48px;line-height:1.05;letter-spacing:-2.4px;color:#fff;position:relative;z-index:2}
.hero p{color:#aeb9c8;margin:13px 0 0;max-width:760px;font-size:15px;line-height:1.65;position:relative;z-index:2}
.hero-top{display:flex;align-items:center;justify-content:space-between;gap:16px;position:relative;z-index:2}
.hero-meta{color:#8290a3;font-size:10px;font-weight:800;letter-spacing:.12em}
.live-badge{display:inline-flex;align-items:center;gap:7px;padding:8px 13px;border-radius:999px;background:#f4774b;border:1px solid #ff9a72;color:#fff;font-size:11px;font-weight:800;letter-spacing:.06em;box-shadow:0 8px 24px rgba(244,119,75,.22)}
.kpi{background:#101b2b;border:1px solid #22334a;border-radius:17px;padding:20px;min-height:128px;box-shadow:0 12px 30px rgba(0,0,0,.18);transition:.2s}
.kpi:hover{transform:translateY(-2px);border-color:#40536d}
.kpi-label{color:#94a1b3;font-size:11px;text-transform:uppercase;letter-spacing:.08em;font-weight:700}
.kpi-value{font-size:34px;line-height:1.1;color:#fff;font-weight:800;margin-top:10px;letter-spacing:-.8px}
.kpi-sub{color:#7e8b9d;font-size:12px;margin-top:7px}
.card{background:#0f1928;border:1px solid #22334a;border-radius:18px;padding:23px;margin-bottom:18px;box-shadow:0 12px 32px rgba(0,0,0,.16)}
.card:hover{border-color:#30445f}
.image-feature{position:relative;overflow:hidden;min-height:230px;border-radius:20px;border:1px solid #22334a;background:#101b2b;box-shadow:0 14px 36px rgba(0,0,0,.2)}
.image-feature:before{content:"";position:absolute;inset:0;background:linear-gradient(90deg,rgba(8,15,27,.96) 0%,rgba(8,15,27,.78) 52%,rgba(8,15,27,.18) 100%),var(--feature-image);background-size:cover;background-position:center}
.image-feature-content{position:relative;z-index:2;padding:28px;max-width:590px}
.image-feature-kicker{color:#f58a62;font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase}
.image-feature h3{font-size:28px!important;line-height:1.15;margin:8px 0!important}
.image-feature p{color:#b8c3d1;font-size:13px;line-height:1.6;max-width:500px}
.image-feature .mini-badge{display:inline-flex;padding:6px 9px;border-radius:999px;background:rgba(244,119,75,.12);border:1px solid rgba(244,119,75,.35);color:#ffb397;font-size:11px;font-weight:700}

.card h3{margin:0 0 8px;font-size:19px;color:#fff}
.badge{display:inline-flex;align-items:center;gap:5px;padding:7px 11px;border-radius:999px;background:#162437;border:1px solid #2b405a;font-size:11px;color:#dce3ec;font-weight:700}
.health-card{background:#101b2b;border:1px solid #253750;border-radius:15px;padding:16px 17px;margin:8px 0}
.health-label{color:#8796aa;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.07em}
.health-value{color:#fff;font-size:22px;font-weight:800;margin-top:6px}
.health-sub{color:#718096;font-size:12px;margin-top:3px}
.section-head{display:flex;align-items:flex-end;justify-content:space-between;gap:18px;margin:5px 0 14px}
.section-title{color:#fff;font-size:20px;font-weight:800;letter-spacing:-.4px}
.section-note{color:#7f8da0;font-size:12px}
div[data-testid="stMetric"]{background:#101b2b;border:1px solid #253750;padding:16px;border-radius:16px}
.stButton>button{border-radius:11px;border:1px solid #31445c;background:#162437;color:#fff;font-size:14px;font-weight:700;min-height:48px;transition:.2s}
.stButton>button:hover{border-color:#f4774b;background:#1d2c40;color:#fff;transform:translateY(-1px);box-shadow:0 8px 20px rgba(244,119,75,.12)}
div[data-testid="stFormSubmitButton"]>button{border-radius:11px;min-height:48px;font-weight:700}
div[data-testid="stRadio"]>label{color:#8290a3!important;font-size:11px!important;font-weight:700!important;letter-spacing:.09em}
div[data-testid="stRadio"] label{display:flex;align-items:center;border-radius:10px;padding:9px 10px;margin:3px 0;color:#d6dde7!important}
div[data-testid="stRadio"] label p{font-size:14px!important;color:#cbd4df!important;font-weight:600!important}
div[data-testid="stRadio"] label:hover{background:#101d2e}
div[data-testid="stRadio"] [aria-checked="true"]{background:#18263a;border-left:3px solid #f4774b}
div[data-testid="stTextInput"] input,div[data-testid="stTextArea"] textarea{background:#0d1725!important;color:#f8fafc!important;border-color:#273b54!important;font-size:15px!important}
div[data-baseweb="select"]>div{background:#0d1725!important;border-color:#273b54!important;min-height:44px}
[data-testid="stDataFrame"]{border:1px solid #253750;border-radius:14px;overflow:hidden}
[data-testid="stCaptionContainer"] p{color:#8b98aa!important;font-size:13px!important}
[data-testid="stAlert"]{border-radius:12px}
hr{border-color:#1f3045}
@media(max-width:900px){section[data-testid="stSidebar"],section[data-testid="stSidebar"]>div{width:260px!important;display:block!important}.block-container{padding:1.2rem 1rem 3rem}.hero h1{font-size:36px}.hero{padding:30px}.kpi-value{font-size:28px}}
@media(max-width:600px){.hero h1{font-size:30px}.hero p{font-size:14px}.hero-meta{display:none}.kpi{min-height:105px;padding:15px}}
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

NAV_ITEMS = [
    "Overview", "Memory Explorer", "Memory Lifecycle", "Semantic Retrieval",
    "Memory Intelligence", "Agent Memory", "Drift Analytics", "Auto-Heal",
    "Multi-Agent", "Benchmark", "Reports", "Settings"
]

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Overview"

def sync_desktop_nav():
    st.session_state.nav_page = st.session_state.desktop_nav

def sync_mobile_nav():
    st.session_state.nav_page = st.session_state.mobile_nav

# Desktop: website-style top navigation. Mobile: keep the compact sidebar navigation.
st.markdown('<div class="desktop-nav"><div class="desktop-nav-label">MEMORYOS</div>', unsafe_allow_html=True)
st.radio(
    "Desktop navigation",
    NAV_ITEMS,
    key="desktop_nav",
    index=NAV_ITEMS.index(st.session_state.nav_page),
    horizontal=True,
    label_visibility="collapsed",
    on_change=sync_desktop_nav,
)
st.markdown('</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="brand">🧠 MemoryOS</div><div class="muted">AI Memory Reliability Layer</div>', unsafe_allow_html=True)
    st.divider()
    st.radio(
        "NAVIGATION",
        NAV_ITEMS,
        key="mobile_nav",
        index=NAV_ITEMS.index(st.session_state.nav_page),
        label_visibility="visible",
        on_change=sync_mobile_nav,
    )
    st.divider()
    st.markdown(f'<div class="card" style="padding:12px 14px;margin:12px 0 8px"><span class="badge">{emoji} {status}</span></div>', unsafe_allow_html=True)
    st.caption(f"● System online  •  Threshold {THRESHOLD:.2f}")
    st.caption(f"Updated: {datetime.now().strftime('%d %b %Y • %H:%M')}")

page = st.session_state.nav_page

st.markdown(f"""
<div class="hero">
  <div class="hero-top">
    <div class="live-badge">● LIVE SYSTEM</div>
    <div class="hero-meta">MEMORYOS PLATFORM&nbsp;&nbsp;•&nbsp;&nbsp;AI AGENT INFRASTRUCTURE</div>
  </div>
  <h1>{page}</h1>
  <p>Monitor memory health, understand retrieval, detect drift and recover stale context from one intelligent workspace.</p>
</div>
""", unsafe_allow_html=True)

if page == "Overview":
    cols = st.columns(4)
    with cols[0]:
        kpi("🧠 Memories", len(memories), "Active memory items")
    with cols[1]:
        kpi("📈 Drift Score", f"{last_score:.2f}", status)
    with cols[2]:
        kpi("🛠 Heal Events", len(heal_history), "Recovery operations")
    with cols[3]:
        kpi("◉ Sessions", len(drift_history), "Tracked sessions")

    st.markdown(
        '<div class="section-head"><div><div class="section-title">Quick actions</div>'
        '<div class="section-note">Run the core MemoryOS operations from here.</div></div></div>',
        unsafe_allow_html=True,
    )
    a, b, c, d = st.columns(4)
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

    left, right = st.columns([1.5, 1])
    with left:
        st.markdown(
            f"""<div class="image-feature" style="--feature-image:url('{SERVER_IMAGE}')">
<div class="image-feature-content">
<div class="image-feature-kicker">Memory Infrastructure</div>
<h3>Reliable context, built for intelligent agents.</h3>
<p>MemoryOS monitors drift, ranks relevant memories and keeps the agent context useful as sessions evolve.</p>
<span class="mini-badge">SELF-HEALING MEMORY</span>
</div></div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-head"><div><div class="section-title">Drift timeline</div>'
            '<div class="section-note">Session-by-session behavioral drift</div></div></div>',
            unsafe_allow_html=True,
        )
        if drift_history:
            df = pd.DataFrame(drift_history)
            if "session" in df and "drift_score" in df:
                st.line_chart(df.set_index("session")["drift_score"], height=280)
        else:
            st.info("No drift sessions yet. Run Drift Analysis to populate the timeline.")

    with right:
        st.markdown(
            f"""<div class="image-feature" style="--feature-image:url('{WORK_IMAGE}');min-height:180px;margin-bottom:16px">
<div class="image-feature-content" style="padding:22px">
<div class="image-feature-kicker">Agent Workspace</div>
<h3 style="font-size:22px!important">Memory that works with your agent.</h3>
<p>Retrieve the right context before every interaction.</p>
</div></div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-head"><div><div class="section-title">System health</div>'
            '<div class="section-note">Reliability and recovery readiness</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="health-card"><div class="health-label">Current Status</div>'
            f'<div class="health-value">{emoji} {status}</div>'
            f'<div class="health-sub">Overall memory reliability state</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="health-card"><div class="health-label">Heal Threshold</div>'
            f'<div class="health-value">{THRESHOLD:.2f}</div>'
            f'<div class="health-sub">Auto-heal trigger level</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="health-card"><div class="health-label">Memory Coverage</div>'
            f'<div class="health-value">{len(memories)} items</div>'
            f'<div class="health-sub">Currently available memory records</div></div>',
            unsafe_allow_html=True,
        )

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
