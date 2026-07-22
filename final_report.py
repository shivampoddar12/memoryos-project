# final_report.py — MemoryOS Week 3 Day 1
# Final Project Report Generator

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import json
from datetime import datetime

print("📄 Generating Final Project Report...")

# Load data
def load(f):
    try:
        return json.load(open(f))
    except:
        return None

bench = load("benchmark_results.json")
calib = load("calibration_report.json")
stats = load("statistical_report.json")

doc = SimpleDocTemplate(
    "MemoryOS_Final_Report.pdf",
    pagesize=A4,
    rightMargin=20*mm, leftMargin=20*mm,
    topMargin=18*mm, bottomMargin=18*mm
)

W = 170*mm
BLK = colors.black
RED = colors.HexColor("#CC0000")
ORG = colors.HexColor("#E05A00")
BLU = colors.HexColor("#1565C0")
GRN = colors.HexColor("#2E7D32")
LGR = colors.HexColor("#F5F5F5")
MGR = colors.HexColor("#E0E0E0")

def ps(name, **kw):
    return ParagraphStyle(name, **kw)

def P(text, style):
    return Paragraph(text, style)

story = []

# ── TITLE ────────────────────────────────────────────
title_s  = ps("T", fontSize=18, fontName="Helvetica-Bold",
               alignment=TA_CENTER, textColor=RED, leading=24)
sub_s    = ps("S", fontSize=12, fontName="Helvetica",
               alignment=TA_CENTER, textColor=colors.HexColor("#444444"), leading=16)
auth_s   = ps("A", fontSize=10, fontName="Helvetica",
               alignment=TA_CENTER, textColor=BLK, leading=14)
body_s   = ps("B", fontSize=10, fontName="Helvetica",
               alignment=TA_JUSTIFY, leading=15, textColor=BLK)
head_s   = ps("H", fontSize=13, fontName="Helvetica-Bold",
               textColor=RED, leading=18, spaceBefore=8)
subh_s   = ps("SH", fontSize=11, fontName="Helvetica-Bold",
               textColor=BLU, leading=16, spaceBefore=4)
code_s   = ps("C", fontSize=9, fontName="Courier",
               textColor=BLK, leading=13,
               backColor=LGR, leftIndent=8, rightIndent=8)
small_s  = ps("SM", fontSize=8.5, fontName="Helvetica",
               textColor=colors.HexColor("#555"), leading=12)

# Header box
hdr = Table([[
    P("MemoryOS for AI Agents", title_s),
]], colWidths=[W])
hdr.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#FFF3E0")),
    ("TOPPADDING",(0,0),(-1,-1),12),
    ("BOTTOMPADDING",(0,0),(-1,-1),6),
    ("BOX",(0,0),(-1,-1),1.5,ORG),
]))
story.append(hdr)

story.append(P("Self-Healing Memory Architecture with Real-Time Drift Detection", sub_s))
story.append(Spacer(1,4*mm))
story.append(P("B.Tech CSE-AI Major Project Report  |  Parul University  |  2025-26", auth_s))
story.append(P("Aaditya Rana (2303051240001)  &  Shivam Kumar Poddar (2303051240374)", auth_s))
story.append(P("Supervisor: Ms. Kiran Sharma", auth_s))
story.append(HRFlowable(width="100%", thickness=2, color=ORG, spaceAfter=6))

# ── ABSTRACT ─────────────────────────────────────────
story.append(P("Abstract", head_s))
story.append(HRFlowable(width="100%", thickness=1, color=RED, spaceAfter=4))
story.append(P(
    "MemoryOS is an open-source, operating-system-inspired memory management layer for "
    "LLM-based AI agents. It addresses the critical problem of <i>Context Rot</i> — the "
    "silent degradation of agent memory that causes 65% of production AI failures. "
    "MemoryOS provides real-time drift detection using cosine similarity, zero-downtime "
    "auto-healing via relevance decay pruning, and multi-agent knowledge synchronization. "
    "Benchmark results demonstrate <b>90.9% drift reduction</b> and <b>64.7% consistency "
    "improvement</b> over agents without memory management. All code is open-sourced at "
    "github.com/shivampoddar12/memoryos-project.",
    body_s))

# ── PROBLEM ──────────────────────────────────────────
story.append(Spacer(1,4*mm))
story.append(P("1. Problem Statement", head_s))
story.append(HRFlowable(width="100%", thickness=1, color=RED, spaceAfter=4))

stats_data = [
    [P("<b>Statistic</b>",ps("th",fontSize=9,fontName="Helvetica-Bold",textColor=colors.white)),
     P("<b>Value</b>",ps("th",fontSize=9,fontName="Helvetica-Bold",textColor=colors.white)),
     P("<b>Source</b>",ps("th",fontSize=9,fontName="Helvetica-Bold",textColor=colors.white))],
    ["Agent failures due to memory loss", "65%",        "Industry Research 2026"],
    ["Market problem by 2031",            "$800 Billion","Gartner, 2026"],
    ["Pilots never reaching production",  "75%",        "McKinsey, 2026"],
    ["Open-source memory health tools",   "0",          "This Work"],
]
st = Table(stats_data, colWidths=[85*mm, 35*mm, 50*mm])
st.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),RED),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LGR,colors.white]),
    ("GRID",(0,0),(-1,-1),0.5,MGR),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),6),
    ("FONTSIZE",(0,1),(-1,-1),9),
    ("FONTNAME",(0,1),(-1,-1),"Helvetica"),
]))
story.append(st)
story.append(Spacer(1,3*mm))
story.append(P(
    "AI agents deployed in production exhibit a consistent failure pattern: reliable "
    "performance during evaluation, followed by silent quality decay after sustained use. "
    "This phenomenon — termed <i>Context Rot</i> — occurs when critical earlier context "
    "is displaced by newer tokens. No existing open-source tool monitors or heals agent "
    "memory health.",
    body_s))

# ── METHODOLOGY ──────────────────────────────────────
story.append(Spacer(1,4*mm))
story.append(P("2. Methodology", head_s))
story.append(HRFlowable(width="100%", thickness=1, color=RED, spaceAfter=4))

story.append(P("2.1 Drift Detection Algorithm", subh_s))
story.append(P(
    "Each agent session is embedded using TF-IDF (500 features, bigram support). "
    "A baseline vector is captured at deployment time. Per-session drift is computed as:",
    body_s))
story.append(Spacer(1,2*mm))

code_box = Table([[P("drift(t) = 1 - cosine_similarity(v_baseline, v_current)\n\nIf drift(t) >= θ (0.45): trigger Auto-Heal", code_s)]],
                 colWidths=[W])
code_box.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,-1),LGR),
    ("BOX",(0,0),(-1,-1),1,MGR),
    ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
    ("LEFTPADDING",(0,0),(-1,-1),10),
]))
story.append(code_box)
story.append(Spacer(1,3*mm))

story.append(P("2.2 Auto-Heal Module", subh_s))
story.append(P(
    "On threshold breach, the Auto-Heal Module executes a three-step transaction "
    "without restarting the agent: (1) compute relevance decay for all memory items "
    "using rel(m,t) = importance(m) × e^(−λ·Δt), (2) prune items below relevance "
    "threshold 0.3, (3) re-inject top-3 highest-importance memories into working context.",
    body_s))

story.append(P("2.3 Threshold Calibration", subh_s))
story.append(P(
    "The drift threshold θ was calibrated empirically across 11 candidate values "
    "(0.20 to 0.70) using 4 scenario types: similar, borderline, different, and extreme. "
    "θ = 0.45 was identified as optimal, correctly classifying all 4 scenario types.",
    body_s))

# ── RESULTS ──────────────────────────────────────────
story.append(Spacer(1,4*mm))
story.append(P("3. Results", head_s))
story.append(HRFlowable(width="100%", thickness=1, color=RED, spaceAfter=4))

imp = bench.get("improvement", {}) if bench else {}
w_out = bench.get("without_memoryos", {}) if bench else {}
w_in  = bench.get("with_memoryos", {}) if bench else {}

res_data = [
    [P("<b>Metric</b>",ps("th2",fontSize=9,fontName="Helvetica-Bold",textColor=colors.white)),
     P("<b>WITHOUT MemoryOS</b>",ps("th2",fontSize=9,fontName="Helvetica-Bold",textColor=colors.white)),
     P("<b>WITH MemoryOS</b>",ps("th2",fontSize=9,fontName="Helvetica-Bold",textColor=colors.white)),
     P("<b>Improvement</b>",ps("th2",fontSize=9,fontName="Helvetica-Bold",textColor=colors.white))],
    ["Average Drift Score",
     str(w_out.get('avg_drift','N/A')),
     str(w_in.get('avg_drift','N/A')),
     f"{imp.get('drift_reduction_percent','N/A')}% reduction"],
    ["Avg Consistency",
     f"{w_out.get('avg_consistency','N/A')}%",
     f"{w_in.get('avg_consistency','N/A')}%",
     f"+{imp.get('consistency_gain_percent','N/A')}%"],
    ["Heal Events",      "0",  str(w_in.get('heal_events','N/A')), "Auto-triggered"],
    ["Sessions Tested",  str(bench.get('sessions_tested','N/A') if bench else 'N/A'),
     str(bench.get('sessions_tested','N/A') if bench else 'N/A'), "10 long-horizon"],
]

rt = Table(res_data, colWidths=[52*mm, 36*mm, 36*mm, 46*mm])
rt.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),GRN),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LGR,colors.white]),
    ("GRID",(0,0),(-1,-1),0.5,MGR),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),6),
    ("FONTSIZE",(0,1),(-1,-1),9),
    ("FONTNAME",(0,1),(-1,-1),"Helvetica"),
]))
story.append(rt)

# ── TECH STACK ───────────────────────────────────────
story.append(Spacer(1,4*mm))
story.append(P("4. Technology Stack", head_s))
story.append(HRFlowable(width="100%", thickness=1, color=RED, spaceAfter=4))

tech_data = [
    [P("<b>Layer</b>",ps("th3",fontSize=9,fontName="Helvetica-Bold",textColor=colors.white)),
     P("<b>Technology</b>",ps("th3",fontSize=9,fontName="Helvetica-Bold",textColor=colors.white)),
     P("<b>Purpose</b>",ps("th3",fontSize=9,fontName="Helvetica-Bold",textColor=colors.white))],
    ["Agent Framework",  "LangGraph 1.2+",    "Stateful agent graph orchestration"],
    ["Embeddings",       "TF-IDF (sklearn)",  "500-dim text vectorization + bigrams"],
    ["Drift Detection",  "NumPy / SciPy",     "Cosine similarity computation"],
    ["Dashboard",        "Streamlit + Plotly","Real-time health visualization"],
    ["Multi-Agent",      "Shared Bus (custom)","Knowledge synchronization"],
    ["Version Control",  "Git + GitHub",      "Reproducible experiments"],
]
tt = Table(tech_data, colWidths=[45*mm, 50*mm, 75*mm])
tt.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),BLU),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LGR,colors.white]),
    ("GRID",(0,0),(-1,-1),0.5,MGR),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),6),
    ("FONTSIZE",(0,1),(-1,-1),9),
    ("FONTNAME",(0,1),(-1,-1),"Helvetica"),
]))
story.append(tt)

# ── REFERENCES ───────────────────────────────────────
story.append(Spacer(1,4*mm))
story.append(P("5. References", head_s))
story.append(HRFlowable(width="100%", thickness=1, color=RED, spaceAfter=4))

refs = [
    "[1] C. Packer et al., \"MemGPT: Towards LLMs as Operating Systems,\" arXiv:2310.08560, 2023.",
    "[2] P. Chhikara et al., \"Mem0: Building Production-Ready AI Agents,\" arXiv:2504.19413, ECAI 2025.",
    "[3] A. Rath, \"Agent Drift: Quantifying Behavioral Degradation,\" arXiv:2601.04170, 2026.",
    "[4] P. Rasmussen et al., \"Zep: A Temporal Knowledge Graph for Agent Memory,\" arXiv:2501.13956, 2025.",
    "[5] M. Kang et al., \"ACON: Context Compression for Long-Horizon Agents,\" arXiv:2510.00615, 2026.",
    "[6] Y. Hu et al., \"MemoryAgentBench: Evaluation Framework,\" arXiv:2507.05257, 2025.",
]
ref_s = ps("RS", fontSize=9, fontName="Helvetica", leading=13,
           textColor=BLK, leftIndent=10, firstLineIndent=-10)
for r in refs:
    story.append(P(r, ref_s))
    story.append(Spacer(1,1.5*mm))

# ── FOOTER ───────────────────────────────────────────
story.append(Spacer(1,4*mm))
foot = Table([[P(
    f"GitHub: github.com/shivampoddar12/memoryos-project  |  Generated: {datetime.now().strftime('%d/%m/%Y')}",
    ps("FT", fontSize=8, fontName="Helvetica", textColor=colors.white, alignment=TA_CENTER)
)]], colWidths=[W])
foot.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,-1),ORG),
    ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
]))
story.append(foot)

doc.build(story)
print("✅ MemoryOS_Final_Report.pdf generated!")
print("✅ Week 3 Day 1 — Final Report Complete!")