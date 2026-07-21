# dashboard.py — MemoryOS Week 2 — Live Dashboard
import streamlit as st
import json
import os
import time
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="MemoryOS Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ── STYLING ──────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1E2530;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
}
.healthy { color: #3FB950; }
.warning { color: #F0C040; }
.critical { color: #FF8C00; }
.danger  { color: #FF4444; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────────────────
def load_data():
    data = {
        "drift_history": [],
        "heal_log": [],
        "memory_log": [],
        "total_sessions": 0
    }
    if os.path.exists("memoryos_data.json"):
        with open("memoryos_data.json") as f:
            data = json.load(f)
    elif os.path.exists("drift_history.json"):
        with open("drift_history.json") as f:
            data["drift_history"] = json.load(f)
    if os.path.exists("heal_log.json"):
        with open("heal_log.json") as f:
            data["heal_log"] = json.load(f)
    if os.path.exists("memory_log.json"):
        with open("memory_log.json") as f:
            data["memory_log"] = json.load(f)
    return data

def get_status(score):
    if score < 0.3:   return "🟢 HEALTHY",  "healthy"
    elif score < 0.45: return "🟡 WARNING",  "warning"
    elif score < 0.7:  return "🔴 CRITICAL", "critical"
    else:              return "🚨 DANGER",   "danger"

# ── HEADER ───────────────────────────────────────────
st.title("🧠 MemoryOS — Agent Health Dashboard")
st.caption("Real-time drift detection & auto-heal monitoring")
st.divider()

# Auto-refresh
auto = st.sidebar.checkbox("Auto Refresh (5s)", value=False)
if auto:
    time.sleep(5)
    st.rerun()

data = load_data()
drift_hist  = data.get("drift_history", [])
heal_log    = data.get("heal_log", [])
memory_log  = data.get("memory_log", [])
total_sess  = data.get("total_sessions", len(drift_hist))

# ── KPI CARDS ────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

last_drift = drift_hist[-1].get("score", drift_hist[-1].get("drift_score", 0.0)) if drift_hist else 0.0
status_label, status_class = get_status(last_drift)

with c1:
    st.metric("📊 Last Drift Score", f"{last_drift:.3f}", help="0=stable, 1=drifted")
with c2:
    health_pct = max(0, int((1 - last_drift) * 100))
    st.metric("💚 Memory Health", f"{health_pct}%")
with c3:
    st.metric("💊 Total Heals", len(heal_log))
with c4:
    st.metric("🔄 Sessions", total_sess)

# Status banner
if drift_hist:
    color = {"healthy":"#3FB950","warning":"#F0C040","critical":"#FF8C00","danger":"#FF4444"}[status_class]
    st.markdown(f"""
    <div style='background:{color}22; border:2px solid {color};
    border-radius:8px; padding:10px; text-align:center; margin:8px 0'>
    <h3 style='color:{color}; margin:0'>{status_label}</h3>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── DRIFT TIMELINE CHART ─────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📈 Drift Score Timeline")
    if drift_hist:
        sessions = list(range(1, len(drift_hist)+1))
        scores   = scores = [d.get("score", d.get("drift_score", 0.0)) for d in drift_hist]
        statuses = [d.get("status","HEALTHY") for d in drift_hist]

        colors = []
        for sc in scores:
            if sc < 0.3:    colors.append("#3FB950")
            elif sc < 0.45: colors.append("#F0C040")
            elif sc < 0.7:  colors.append("#FF8C00")
            else:           colors.append("#FF4444")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sessions, y=scores,
            marker_color=colors,
            name="Drift Score",
            hovertemplate="Session %{x}<br>Drift: %{y:.3f}<extra></extra>"
        ))
        # Threshold line
        fig.add_hline(y=0.45, line_dash="dash", line_color="#FF8C00",
                      annotation_text="Heal Threshold (0.45)")
        fig.update_layout(
            xaxis_title="Session",
            yaxis_title="Drift Score",
            yaxis_range=[0, 1.05],
            plot_bgcolor="#0D1117",
            paper_bgcolor="#0D1117",
            font_color="#E6EDF3",
            showlegend=False,
            height=300,
            margin=dict(l=20, r=20, t=20, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run memoryos.py first to generate drift data!")

# ── HEAL EVENTS ──────────────────────────────────────
with col_right:
    st.subheader("💊 Heal Events")
    if heal_log:
        for h in reversed(heal_log[-5:]):
            sc = h.get("drift_score_before", h.get("drift_score", 0))
            st.markdown(f"""
            <div style='background:#1E2530; border-radius:8px;
            padding:8px; margin:5px 0; border-left:3px solid #FF4444'>
            <b>Heal #{h.get('heal_number',h.get('heal_count','?'))}</b><br>
            Drift before: <span style='color:#FF4444'>{sc:.3f}</span><br>
            Memories: {h.get('memories_before','?')} → {h.get('memories_after','?')}
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No heal events yet")

st.divider()

# ── MEMORY LOG TABLE ─────────────────────────────────
col_mem, col_stats = st.columns([2,1])

with col_mem:
    st.subheader("🧠 Memory Log")
    if memory_log:
        import pandas as pd
        rows = []
        for m in memory_log[-10:]:
            rows.append({
                "Session": m.get("session","?"),
                "Input":   m.get("input","?")[:50]+"..." if len(m.get("input",""))>50 else m.get("input","?"),
                "Time":    m.get("time","?")[:19] if m.get("time") else "?"
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No memory entries yet")

with col_stats:
    st.subheader("📊 Health Stats")
    if drift_hist:
        scores = [d.get("score", d.get("drift_score", 0.0)) for d in drift_hist]
        st.metric("Average Drift",  f"{sum(scores)/len(scores):.3f}")
        st.metric("Max Drift",      f"{max(scores):.3f}")
        st.metric("Min Drift",      f"{min(scores):.3f}")
        healthy = sum(1 for sc in scores if sc < 0.45)
        st.metric("Healthy Sessions", f"{healthy}/{len(scores)}")
    else:
        st.info("No stats yet")

# ── FOOTER ───────────────────────────────────────────
st.divider()
st.caption(f"MemoryOS v1.0 | Last updated: {datetime.now().strftime('%H:%M:%S')} | github.com/shivampoddar12/memoryos-project")