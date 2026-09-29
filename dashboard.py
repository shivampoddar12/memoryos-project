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
html,body,[class*="css"],.stApp,.stApp *,.stMarkdown,.stButton>button,.stTextInput input,.stTextArea textarea,.stSelectbox,.stSlider,.stDataFrame,.stCaption,.stAlert{
  font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif!important;
}
.stApp{background:#080f1c;color:#f6f7f9}
section[data-testid="stSidebar"]{width:285px!important;background:#0a1423;border-right:1px solid #1c2a3d}
section[data-testid="stSidebar"]>div{width:285px!important}
.block-container{max-width:1440px;padding:1.5rem 3.2rem 5rem}
/* Streamlit chrome reset */
header[data-testid="stHeader"]{background:transparent!important;height:0!important;min-height:0!important}
div[data-testid="stToolbar"]{display:none!important}
div[data-testid="stDecoration"]{display:none!important}
#MainMenu{visibility:hidden!important}
footer{visibility:hidden!important}
.stAppViewContainer{padding-top:0!important}
.main .block-container{padding-top:.65rem!important}

.desktop-nav-shell{background:#0b1422;border:1px solid #24354b;border-top:0;border-radius:0 0 14px 14px;padding:0 0 4px;margin-bottom:24px}
.desktop-nav-shell .desktop-nav-row{padding:10px 18px}
.global-search-panel{margin:-12px 0 18px;padding:14px 18px;border:1px solid #26374b;border-radius:10px;background:#0b1420}
.global-search-panel .stTextInput input{background:#0d1725!important;color:#e6edf5!important;border:1px solid #33485f!important}
.global-search-panel .stButton>button{min-height:40px!important;background:#f4774b!important;color:#fff!important;border:1px solid #f4774b!important}
.search-result{display:flex;justify-content:space-between;gap:20px;padding:11px 13px;margin:7px 0;border:1px solid #26374b;border-radius:7px;background:#0d1725;color:#dbe2ea;font-size:11px}
.search-result span{color:#8391a2;white-space:nowrap}
.desktop-site-header{display:block;margin:0 0 24px;background:#0b1422;border:1px solid #24354b;border-radius:0 0 14px 14px;box-shadow:0 10px 30px rgba(0,0,0,.22);overflow:visible;position:relative;z-index:20}
.desktop-site-header .stHorizontalBlock{margin:0!important;padding:0 24px 16px!important;align-items:center!important}
.desktop-site-header .stButton{margin:0!important}
.desktop-site-header .stHorizontalBlock{margin:0!important;padding:8px 18px!important;align-items:center!important;gap:6px!important}
.desktop-brand-inline{font-size:20px;font-weight:800;letter-spacing:-1px;color:#f2eee8;white-space:nowrap;line-height:40px}
.desktop-brand-inline span{color:#f4774b}
.desktop-site-header .stButton>button{min-height:40px!important;height:40px!important;margin:0!important;padding:0 7px!important;border-radius:7px!important;font-size:9.5px!important}
.desktop-site-header .stHorizontalBlock .stButton>button[aria-label="Search MemoryOS"]{padding:0!important;font-size:18px!important;line-height:1!important;color:#f4774b!important;background:#0d1725!important;border:1px solid #40536b!important}
.desktop-site-header .stHorizontalBlock .stButton>button[aria-label="Search MemoryOS"]:hover{color:#fff!important;border-color:#f4774b!important;background:#17273a!important}



.desktop-nav-shell{position:relative;z-index:10;clear:both;margin-top:0!important;margin-bottom:8px!important}
.desktop-nav-heading{padding:9px 18px 0;color:#718096;font-size:8px;font-weight:800;letter-spacing:.16em}
.main .block-container{overflow-x:hidden!important}
.desktop-nav-item,.desktop-nav-more{min-width:0!important}
.desktop-nav-row{display:block!important;overflow:visible!important}


.desktop-topbar{display:flex;align-items:center;gap:30px;padding:14px 24px;border-bottom:1px solid #1d2b3e}
.desktop-brand{font-size:23px;font-weight:800;letter-spacing:-1px;color:#f2eee8;white-space:nowrap}
.desktop-brand span{color:#f4774b}
.desktop-search{margin-left:auto;min-width:190px;border:1px solid #293b52;border-radius:7px;padding:8px 12px;color:#718095;font-size:10px;background:#0e1928}
.desktop-nav-row{display:flex;align-items:center;gap:8px;padding:10px 18px;overflow:visible}
.desktop-nav-row>div[data-testid="column"]{min-width:0}
.desktop-nav-item{width:100%}
.desktop-nav-more{width:100%}
.desktop-nav-more.active{border-radius:7px}
.more-title{font-size:9px;font-weight:800;letter-spacing:.14em;color:#8d9aab;margin:2px 0 10px}
.desktop-nav-more .stButton>button{
  min-height:40px!important;height:40px!important;width:100%!important;
  padding:0 12px!important;border:1px solid #40536b!important;border-radius:7px!important;
  background:#0d1725!important;background-color:#0d1725!important;
  color:#e3e8ee!important;font-family:Inter,system-ui,sans-serif!important;
  font-size:10px!important;font-weight:700!important;letter-spacing:.03em!important;
  box-shadow:none!important;opacity:1!important;
}
.desktop-nav-more .stButton>button:hover{background:#17273a!important;background-color:#17273a!important;color:#fff!important;border-color:#f4774b!important}
.desktop-nav-more.active .stButton>button{border-color:#f4774b!important;color:#fff!important}
.more-nav-panel{margin:4px 18px 8px;padding:10px 0 4px;border-top:1px solid #26374b}
.more-nav-panel .stButton>button{
  min-height:36px!important;height:36px!important;padding:0 10px!important;
  border:1px solid #24364b!important;border-radius:6px!important;
  background:#0d1725!important;color:#c1cbd6!important;font-size:10px!important;font-weight:600!important;
}
.more-nav-panel .stButton>button:hover{background:#17273a!important;color:#fff!important;border-color:#f4774b!important}
.desktop-nav-label{display:none!important}
@media(min-width:901px){section[data-testid="stSidebar"]{display:none!important}.main .block-container{max-width:1500px;padding-left:3.2rem;padding-right:3.2rem}}
.landing-hero{position:relative;overflow:hidden;min-height:720px;margin:0 0 55px;padding:32px 38px 26px;border:1px solid #26374b;background:radial-gradient(circle at 72% 44%,#18283b 0%,#0a111c 31%,#050a10 74%);box-shadow:0 25px 80px rgba(0,0,0,.35)}
.landing-top{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #1d2a3a;padding-bottom:16px;position:relative;z-index:5}
.landing-kicker,.landing-eyebrow{font-size:9px;font-weight:800;letter-spacing:.18em;color:#8f9baa}
.landing-status{font-size:9px;letter-spacing:.12em;color:#b5bfcb}.landing-status span{display:inline-block;width:6px;height:6px;background:#f4774b;border-radius:50%;margin-right:7px;box-shadow:0 0 12px #f4774b}
.landing-copy{position:relative;z-index:4;margin-top:68px;width:52%}.landing-copy .landing-eyebrow{color:#f0b19b;margin-bottom:12px}.landing-copy h1{font-size:clamp(72px,9vw,138px);line-height:.78;letter-spacing:-.075em;font-weight:500;color:#e7e0d8;margin:0}.landing-copy h1 em{font-style:normal;color:#8d8f92}.landing-copy p{max-width:470px;color:#b1bbc7;font-size:13px;line-height:1.7;margin:32px 0 0}
.landing-side{position:absolute;right:7%;top:35%;width:210px;z-index:5;color:#c0c8d2;font-size:10px;line-height:1.6}.landing-side strong{display:block;color:#e5e7eb;font-size:11px;margin:8px 0 4px}.side-line{width:32px;height:1px;background:#f4774b}
.landing-cta-row{position:absolute;left:38px;bottom:90px;z-index:7;display:flex;align-items:center;gap:15px}.fake-cta{background:#e9e4dc;border:0;color:#0a0e13;border-radius:3px;padding:12px 18px;font-size:11px;font-weight:800}.fake-cta span{margin-left:18px}.cta-note{font-size:9px;color:#697687;letter-spacing:.08em}
.memory-core{position:absolute;right:15%;top:20%;width:380px;height:380px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:radial-gradient(circle at 40% 35%,#6f7377,#272b2e 28%,#0a0f14 64%);box-shadow:inset -35px -35px 75px rgba(0,0,0,.8),0 35px 100px rgba(0,0,0,.5);z-index:2}.core-ring{position:absolute;border:1px solid rgba(239,228,216,.22);border-radius:50%}.ring-one{width:430px;height:430px}.ring-two{width:330px;height:330px;border-color:rgba(244,119,75,.25)}.core-mark{position:relative;width:92px;height:92px}.core-mark i{position:absolute;width:36px;height:36px;border:7px solid #e8e0d8;border-radius:13px;transform:rotate(45deg)}.core-mark i:nth-child(1){left:8px;top:28px}.core-mark i:nth-child(2){left:48px;top:28px}.core-mark i:nth-child(3){left:28px;top:8px}.core-mark i:nth-child(4){left:28px;top:48px}.core-label{position:absolute;bottom:-90px;text-align:center;color:#a8b0ba;font-size:8px;letter-spacing:.2em;line-height:1.5}
.landing-orbit{position:absolute;z-index:6;border:1px solid #354253;color:#9ea8b4;background:#0b1119;border-radius:50%;width:68px;height:68px;display:flex;align-items:center;justify-content:center;font-size:8px;letter-spacing:.12em}.orbit-one{right:9%;top:28%}.orbit-two{right:7%;top:57%}.orbit-three{right:25%;bottom:15%}
.landing-bottom{position:absolute;left:38px;right:38px;bottom:25px;border-top:1px solid #1e2b3b;padding-top:13px;display:flex;gap:55px;color:#9aa6b4;font-size:9px;letter-spacing:.08em}.landing-bottom b{color:#e0d8cf;margin-right:8px}
.landing-section-title{margin:0 0 22px}.landing-section-title span{font-size:9px;letter-spacing:.16em;color:#f4774b;font-weight:800}.landing-section-title h2{font-size:42px;line-height:1.05;letter-spacing:-.04em;color:#e6e1db;font-weight:500;margin:10px 0 28px}
.cap-card{min-height:210px;border:1px solid #26374b;background:#0b1420;padding:21px;position:relative;transition:.2s}.cap-card:hover{border-color:#f4774b;transform:translateY(-3px)}
div[data-testid="stHorizontalBlock"] .stButton>button{transition:all .18s ease!important}
div[data-testid="stHorizontalBlock"] .stButton>button:hover{border-color:#f4774b!important;box-shadow:0 0 0 1px rgba(244,119,75,.35)!important}
.cap-card .stButton{margin-top:12px}.cap-card .stButton>button{min-height:34px!important;height:34px!important;padding:0 10px!important;border-radius:6px!important;background:#101d2d!important;border:1px solid #30455d!important;color:#f4774b!important;font-size:9px!important;font-weight:800!important;letter-spacing:.08em!important}.cap-card .stButton>button:hover{background:#f4774b!important;color:#fff!important;border-color:#f4774b!important}.cap-card>span{color:#f4774b;font-size:9px;letter-spacing:.12em}.cap-card h3{font-size:17px!important;margin:58px 0 9px!important;color:#e6e1db}.cap-card p{font-size:11px;line-height:1.65;color:#a3afbd}.cap-card>b{position:absolute;right:18px;bottom:18px;color:#c5ccd4}
.landing-split{display:grid;grid-template-columns:1fr 1fr;gap:45px;align-items:center;margin:95px 0;padding:30px 0}.split-copy{padding:10px}.split-copy h2{font-size:45px;line-height:1.02;font-weight:500;letter-spacing:-.04em;color:#e6e1db;margin:11px 0 20px}.split-copy>p{color:#a5b0bd;font-size:12px;line-height:1.8;max-width:500px}.split-points{margin-top:28px}.split-points div{padding:13px 0;border-top:1px solid #223145;color:#a6b1bd;font-size:11px}.split-points b{color:#f4774b;margin-right:16px}.split-visual{height:430px;position:relative;overflow:hidden;border:1px solid #27374a;background:radial-gradient(circle,#1d2b3a 0,#0a111a 50%,#05090e 100%)}.visual-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.04) 1px,transparent 1px);background-size:35px 35px;transform:perspective(400px) rotateX(58deg) scale(1.5);transform-origin:center bottom}.visual-core{position:absolute;left:50%;top:45%;transform:translate(-50%,-50%);width:135px;height:135px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:55px;color:#e9e1d8;background:radial-gradient(circle,#53585c,#151a1f 65%);box-shadow:0 0 70px rgba(244,119,75,.13)}.split-visual span{position:absolute;left:20px;bottom:18px;color:#7e8b9b;font-size:8px;letter-spacing:.18em;line-height:1.5}
.landing-metrics{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid #26374b;border-bottom:1px solid #26374b;margin:40px 0 90px}.landing-metrics div{padding:28px 22px;border-right:1px solid #26374b}.landing-metrics div:last-child{border-right:0}.landing-metrics strong{display:block;color:#e8e1d9;font-size:38px;font-weight:500;letter-spacing:-.05em}.landing-metrics span{font-size:8px;letter-spacing:.16em;color:#778495}
.landing-bottom-cta{width:100%;margin:78px 0 0;padding:0;border-top:1px solid #26374b}
.bottom-cta-inner{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:60px;align-items:end;padding:46px 8px 82px}
.landing-bottom-cta h2{font-size:48px;line-height:1.02;letter-spacing:-.04em;color:#e6e1db;font-weight:500;margin:9px 0 0;max-width:760px}
.landing-cta-copy{max-width:320px;color:#788596;font-size:11px;line-height:1.7;padding-bottom:4px}
@media(max-width:900px){.desktop-site-header{display:none!important}}
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
div[data-testid="stHorizontalBlock"] .stButton>button{
  border-radius:7px!important;
}
div[data-testid="stFormSubmitButton"]>button{border-radius:11px;min-height:48px;font-weight:700}
.desktop-nav-row .stButton>button{
  min-height:40px!important;height:40px!important;width:100%!important;padding:0 10px!important;
  border:1px solid #24364b!important;border-radius:7px!important;background:#0d1725!important;
  color:#c1cbd6!important;font-family:Inter,system-ui,sans-serif!important;font-size:10.5px!important;
  font-weight:600!important;box-shadow:none!important;transform:none!important;
}
.desktop-nav-row .stButton>button:hover{background:#17273a!important;border-color:#40536b!important;color:#fff!important}

.desktop-nav-row .stButton>button{
  min-height:40px!important;height:40px!important;width:100%!important;
  padding:0 14px!important;border:1px solid #24364b!important;
  border-radius:7px!important;background:#0d1725!important;
  color:#c1cbd6!important;font-family:Inter,system-ui,sans-serif!important;
  font-size:11px!important;font-weight:600!important;
  box-shadow:none!important;transform:none!important;
}
.desktop-nav-row .stButton>button:hover{
  background:#17273a!important;border-color:#40536b!important;color:#fff!important;
  transform:none!important;box-shadow:none!important;
}
.desktop-nav-row .stButton>button:focus{
  outline:none!important;box-shadow:0 0 0 1px #f4774b!important;
}

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
@media(max-width:900px){section[data-testid="stSidebar"],section[data-testid="stSidebar"]>div{width:260px!important;display:block!important}.block-container{padding:1.2rem 1rem 3rem}.hero h1{font-size:36px}.hero{padding:30px}.kpi-value{font-size:28px}.landing-hero{min-height:650px;padding:22px 20px}.landing-copy{width:100%;margin-top:55px}.landing-copy h1{font-size:70px}.landing-side{display:none}.memory-core{width:245px;height:245px;right:5%;top:37%}.ring-one{width:275px;height:275px}.ring-two{width:215px;height:215px}.landing-orbit{width:54px;height:54px}.landing-cta-row{left:20px;bottom:78px}.landing-bottom{left:20px;right:20px;gap:15px;overflow:hidden}.landing-split{grid-template-columns:1fr;margin:60px 0}.landing-section-title h2,.split-copy h2{font-size:34px}.landing-metrics{grid-template-columns:1fr 1fr}.landing-metrics div{border-bottom:1px solid #26374b}.landing-bottom-cta{display:block}.bottom-cta-inner{display:block;padding:38px 0 55px}.landing-bottom-cta h2{font-size:36px}.landing-cta-copy{margin-top:20px}}
@media(max-width:600px){.hero h1{font-size:30px}.hero p{font-size:14px}.hero-meta{display:none}.kpi{min-height:105px;padding:15px}.landing-hero{min-height:620px}.landing-copy h1{font-size:56px}.memory-core{width:190px;height:190px;top:39%;right:2%}.ring-one{width:215px;height:215px}.ring-two{width:170px;height:170px}.landing-orbit{display:none}.landing-bottom{font-size:8px;gap:9px}.landing-bottom div:nth-child(2){display:none}.landing-metrics strong{font-size:30px}}
.overview-chart-head{display:flex;justify-content:space-between;align-items:end;margin:42px 0 12px;padding-top:22px;border-top:1px solid #24354b}
.overview-chart-head span{font-size:8px;letter-spacing:.16em;color:#f4774b;font-weight:800}
.overview-chart-head h3{margin:6px 0 0;color:#e6e1db;font-size:25px;font-weight:600}
.chart-threshold{font-size:9px;color:#8f9baa;border:1px solid #33485f;border-radius:999px;padding:7px 10px}
.chart-caption{font-size:9px;color:#718096;margin:-4px 0 26px}

/* Unified MemoryOS typography */
*, *::before, *::after{font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif!important}

/* MemoryOS hero video slideshow */
.landing-hero{isolation:isolate}
.hero-video-slideshow{position:absolute;inset:0;z-index:0;overflow:hidden;background:#050a10}
.hero-video{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0;animation:memoryHeroVideo 18s infinite;filter:saturate(.78) contrast(1.04)}
.hero-video-1{animation-delay:0s}
.hero-video-2{animation-delay:6s}
.hero-video-3{animation-delay:12s}
.hero-video-slideshow:after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,rgba(5,10,16,.94) 0%,rgba(5,10,16,.70) 42%,rgba(5,10,16,.52) 100%),linear-gradient(180deg,rgba(5,10,16,.20),rgba(5,10,16,.70))}
.landing-hero>.landing-glow,.landing-hero>.landing-top,.landing-hero>.landing-copy,.landing-hero>.landing-side,.landing-hero>.landing-cta-row,.landing-hero>.landing-orbit,.landing-hero>.memory-core{z-index:2}
@keyframes memoryHeroVideo{
  0%,28%{opacity:0}
  7%,21%{opacity:.48}
  34%,100%{opacity:0}
}

/* Floating technology logos */
.tech-logo-cloud{position:absolute;inset:0;z-index:1;pointer-events:none;overflow:hidden;opacity:.22}
.tech-logo-cloud img{position:absolute;width:30px;height:30px;filter:grayscale(1) brightness(1.9);animation:techFloat 10s ease-in-out infinite}
.tech-logo-cloud img:nth-child(1){left:8%;top:24%;animation-delay:-1s}
.tech-logo-cloud img:nth-child(2){left:25%;top:12%;animation-delay:-4s}
.tech-logo-cloud img:nth-child(3){left:43%;top:20%;animation-delay:-7s}
.tech-logo-cloud img:nth-child(4){left:66%;top:13%;animation-delay:-2s}
.tech-logo-cloud img:nth-child(5){left:87%;top:25%;animation-delay:-5s}
.tech-logo-cloud img:nth-child(6){left:12%;top:70%;animation-delay:-6s}
.tech-logo-cloud img:nth-child(7){left:35%;top:78%;animation-delay:-3s}
.tech-logo-cloud img:nth-child(8){left:58%;top:72%;animation-delay:-8s}
.tech-logo-cloud img:nth-child(9){left:78%;top:67%;animation-delay:-1s}
.tech-logo-cloud img:nth-child(10){left:92%;top:78%;animation-delay:-6s}
@keyframes techFloat{0%,100%{transform:translate3d(0,0,0) rotate(0deg)}50%{transform:translate3d(0,-12px,0) rotate(4deg)}}
@media(max-width:600px){.tech-logo-cloud{opacity:.13}.tech-logo-cloud img{width:22px;height:22px}}

/* Full-page floating technology logo background */
.site-tech-logo-cloud{position:fixed;inset:0;width:100vw;height:100vh;z-index:1;pointer-events:none;overflow:hidden;opacity:.13}
.site-tech-logo-cloud:after{content:"";position:absolute;inset:0;background:radial-gradient(circle at center,transparent 18%,rgba(5,10,16,.25) 62%,rgba(5,10,16,.72) 100%)}
.site-tech-logo-cloud img{position:absolute;width:34px;height:34px;filter:grayscale(1) brightness(1.9);animation:siteTechFloat 12s ease-in-out infinite}
.site-tech-logo-cloud img:nth-child(1){left:5%;top:13%;animation-delay:-2s}.site-tech-logo-cloud img:nth-child(2){left:18%;top:31%;animation-delay:-7s}.site-tech-logo-cloud img:nth-child(3){left:31%;top:10%;animation-delay:-4s}.site-tech-logo-cloud img:nth-child(4){left:46%;top:28%;animation-delay:-9s}.site-tech-logo-cloud img:nth-child(5){left:62%;top:11%;animation-delay:-5s}.site-tech-logo-cloud img:nth-child(6){left:78%;top:30%;animation-delay:-1s}.site-tech-logo-cloud img:nth-child(7){left:91%;top:15%;animation-delay:-8s}.site-tech-logo-cloud img:nth-child(8){left:9%;top:58%;animation-delay:-6s}.site-tech-logo-cloud img:nth-child(9){left:24%;top:76%;animation-delay:-3s}.site-tech-logo-cloud img:nth-child(10){left:43%;top:61%;animation-delay:-10s}.site-tech-logo-cloud img:nth-child(11){left:68%;top:78%;animation-delay:-4s}.site-tech-logo-cloud img:nth-child(12){left:88%;top:60%;animation-delay:-7s}
@keyframes siteTechFloat{0%,100%{transform:translate3d(0,0,0) rotate(0deg)}50%{transform:translate3d(8px,-16px,0) rotate(5deg)}}
@media(max-width:600px){.site-tech-logo-cloud{opacity:.08}.site-tech-logo-cloud img{width:23px;height:23px}}
.main .block-container{position:relative;z-index:2}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="site-tech-logo-cloud" aria-hidden="true">
  <img src="https://cdn.simpleicons.org/python" alt="">
  <img src="https://cdn.simpleicons.org/react" alt="">
  <img src="https://cdn.simpleicons.org/streamlit" alt="">
  <img src="https://cdn.simpleicons.org/langchain" alt="">
  <img src="https://cdn.simpleicons.org/github" alt="">
  <img src="https://cdn.simpleicons.org/numpy" alt="">
  <img src="https://cdn.simpleicons.org/pandas" alt="">
  <img src="https://cdn.simpleicons.org/scikitlearn" alt="">
  <img src="https://cdn.simpleicons.org/flask" alt="">
  <img src="https://cdn.simpleicons.org/docker" alt="">
  <img src="https://cdn.simpleicons.org/tensorflow" alt="">
  <img src="https://cdn.simpleicons.org/openai" alt="">
</div>
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
PRIMARY_NAV = ["Overview", "Memory Explorer", "Memory Lifecycle", "Semantic Retrieval", "Memory Intelligence", "Agent Memory"]
SECONDARY_NAV = ["Drift Analytics", "Auto-Heal", "Multi-Agent", "Benchmark", "Reports", "Settings"]

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Overview"
if "show_more_nav" not in st.session_state:
    st.session_state.show_more_nav = False
if "search_open" not in st.session_state:
    st.session_state.search_open = False

def set_primary():
    st.session_state.nav_page = st.session_state.primary_nav

def set_secondary():
    st.session_state.nav_page = st.session_state.secondary_nav

# Desktop: single-line website navigation. Extra sections live under "More".
VISIBLE_NAV = ["Overview", "Memory Explorer", "Memory Lifecycle", "Semantic Retrieval", "Memory Intelligence", "Agent Memory"]
MORE_NAV = ["Drift Analytics", "Auto-Heal", "Multi-Agent", "Benchmark", "Reports", "Settings"]

def render_single_nav():
    # Single desktop website row: brand + primary navigation + search icon.
    cols = st.columns([1.25, 0.82, 1.05, 1.05, 1.10, 1.12, 0.98, 0.88, 0.32], gap="small")

    with cols[0]:
        st.markdown('<div class="desktop-brand-inline">🧠 Memory<span>OS</span></div>', unsafe_allow_html=True)

    for i, item in enumerate(VISIBLE_NAV):
        with cols[i + 1]:
            if st.button(item, key=f"topnav_{i}_{item}", use_container_width=True):
                st.session_state.nav_page = item
                st.rerun()

    with cols[7]:
        if st.button("SEE MORE  +", key="topnav_more", use_container_width=True):
            st.session_state.show_more_nav = not st.session_state.get("show_more_nav", False)
            st.rerun()

    with cols[8]:
        if st.button("🔍", key="global_search_toggle", help="Search MemoryOS", use_container_width=True):
            st.session_state.search_open = not st.session_state.search_open
            st.rerun()

    if st.session_state.get("show_more_nav", False):
        st.markdown('<div class="more-nav-panel"><div class="more-title">MORE MEMORYOS</div>', unsafe_allow_html=True)
        more_cols = st.columns(6, gap="small")
        for i, item in enumerate(MORE_NAV):
            with more_cols[i]:
                if st.button(item, key=f"more_item_{i}_{item}", use_container_width=True):
                    st.session_state.nav_page = item
                    st.session_state.show_more_nav = False
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# One-line desktop website navigation: brand + navigation + search icon.
st.markdown('<div class="desktop-site-header">', unsafe_allow_html=True)
st.markdown('<div class="desktop-nav-heading">MEMORYOS</div>', unsafe_allow_html=True)
render_single_nav()
st.markdown('</div>', unsafe_allow_html=True)


if st.session_state.search_open:
    st.markdown('<div class="global-search-panel">', unsafe_allow_html=True)
    qcol, bcol = st.columns([5, 1], gap="small")
    with qcol:
        query = st.text_input("Search MemoryOS", placeholder="Search memories, pages or concepts…", label_visibility="collapsed", key="global_search_query")
    with bcol:
        do_search = st.button("SEARCH", key="global_search_submit", use_container_width=True)
    if do_search:
        q = query.strip()
        if not q:
            st.warning("Type something to search.")
        else:
            page_hits = [p for p in NAV_ITEMS if q.lower() in p.lower()]
            memory_hits = ENGINE.search(q, limit=8)
            if page_hits:
                st.markdown("**Pages**")
                page_cols = st.columns(min(4, len(page_hits)), gap="small")
                for i, p in enumerate(page_hits):
                    with page_cols[i]:
                        if st.button(p, key=f"search_page_{i}_{p}", use_container_width=True):
                            st.session_state.nav_page = p
                            st.session_state.search_open = False
                            st.rerun()
            st.markdown("**Memory results**")
            if memory_hits:
                for hit in memory_hits:
                    st.markdown(f'<div class="search-result"><b>{hit.get("content","")}</b><span>similarity {float(hit.get("similarity",0)):.2f} • importance {float(hit.get("importance",0)):.2f}</span></div>', unsafe_allow_html=True)
            else:
                st.info("No matching memories found.")
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
        on_change=lambda: st.session_state.update(nav_page=st.session_state.mobile_nav),
    )
    st.divider()
    st.markdown(f'<div class="card" style="padding:12px 14px;margin:12px 0 8px"><span class="badge">{emoji} {status}</span></div>', unsafe_allow_html=True)
    st.caption(f"● System online  •  Threshold {THRESHOLD:.2f}")
    st.caption(f"Updated: {datetime.now().strftime('%d %b %Y • %H:%M')}")

page = st.session_state.nav_page

if page != "Overview":
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
    # Marketing-style landing page inspired by the supplied reference, while keeping MemoryOS functionality live.
    st.markdown(
        f"""
        <section class="landing-hero">
          <div class="hero-video-slideshow" aria-hidden="true">
            <video class="hero-video hero-video-1" autoplay muted loop playsinline preload="metadata">
              <source src="https://videos.pexels.com/video-files/3129977/3129977-uhd_3840_2160_30fps.mp4" type="video/mp4">
            </video>
            <video class="hero-video hero-video-2" autoplay muted loop playsinline preload="metadata">
              <source src="https://videos.pexels.com/video-files/5028622/5028622-uhd_3840_2160_25fps.mp4" type="video/mp4">
            </video>
            <video class="hero-video hero-video-3" autoplay muted loop playsinline preload="metadata">
              <source src="https://videos.pexels.com/video-files/1085656/1085656-uhd_3840_2160_25fps.mp4" type="video/mp4">
            </video>
          </div>
          <div class="landing-glow glow-a"></div>
          <div class="landing-glow glow-b"></div>
          <div class="landing-top">
            <div class="landing-kicker">MEMORYOS / AI MEMORY INFRASTRUCTURE</div>
            <div class="landing-status"><span></span> SYSTEM ONLINE</div>
          </div>
          <div class="landing-copy">
            <div class="landing-eyebrow">RELIABLE CONTEXT FOR AI AGENTS</div>
            <h1>Future<br><em>Memory.</em></h1>
            <p>Build intelligent AI agents that remember what matters, detect memory drift, retrieve the right context and recover stale knowledge automatically.</p>
          </div>
          <div class="landing-side">
            <div class="side-line"></div>
            <strong>Autonomous Memory</strong>
            <span>Semantic retrieval, drift detection and self-healing memory in one intelligent layer.</span>
          </div>
          <div class="landing-cta-row">
            <span class="cta-note">Use the navigation to explore the live memory workspace.</span>
            <span class="cta-note">Built for intelligent agents</span>
          </div>
          <div class="landing-orbit orbit-one">RETRIEVE</div>
          <div class="landing-orbit orbit-two">DRIFT</div>
          <div class="landing-orbit orbit-three">HEAL</div>
          <div class="memory-core">
            <div class="core-ring ring-one"></div>
            <div class="core-ring ring-two"></div>
            <div class="core-mark"><i></i><i></i><i></i><i></i></div>
            <div class="core-label">MEMORY<br>ENGINE</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Explore MemoryOS  ↗", key="hero_explore", use_container_width=False):
        st.session_state.nav_page = "Memory Explorer"
        st.rerun()

    st.markdown('<div class="landing-section-title"><span>MEMORYOS CAPABILITIES</span><h2>Everything your agent needs<br>to remember intelligently.</h2></div>', unsafe_allow_html=True)

    f1,f2,f3,f4 = st.columns(4)
    cards = [
        ("01","Semantic Retrieval","Find the most relevant memories using TF-IDF similarity and weighted relevance."),
        ("02","Drift Detection","Measure how memory context changes across sessions and flag unhealthy drift."),
        ("03","Memory Lifecycle","Track active and stale memories as relevance changes over time."),
        ("04","Auto-Heal","Prune low-value context and preserve high-value memories when drift crosses the threshold."),
    ]
    card_routes = {
        "Semantic Retrieval": "Semantic Retrieval",
        "Drift Detection": "Drift Analytics",
        "Memory Lifecycle": "Memory Lifecycle",
        "Auto-Heal": "Auto-Heal",
    }
    for idx,(col,(num,title,desc)) in enumerate(zip((f1,f2,f3,f4),cards)):
        with col:
            st.markdown(f'<div class="cap-card"><span>{num}</span><h3>{title}</h3><p>{desc}</p></div>',unsafe_allow_html=True)
            if st.button("OPEN  ↗", key=f"capability_{idx}_{title}", use_container_width=True):
                st.session_state.nav_page = card_routes[title]
                st.session_state.search_open = False
                st.rerun()

    if drift_history:
        st.markdown('<div class="overview-chart-head"><div><span>MEMORY HEALTH</span><h3>Drift Timeline</h3></div><div class="chart-threshold">Auto-heal threshold 0.45</div></div>', unsafe_allow_html=True)
        chart_df = pd.DataFrame(drift_history)
        if "session" in chart_df.columns and "drift_score" in chart_df.columns:
            chart_df = chart_df[["session", "drift_score"]].copy().set_index("session")
            st.line_chart(chart_df, height=280, use_container_width=True)
        st.markdown('<div class="chart-caption">Session-by-session memory drift. Higher values indicate greater context change.</div>', unsafe_allow_html=True)

    st.markdown('<div class="landing-split"><div class="split-copy"><div class="landing-eyebrow">MEMORY INFRASTRUCTURE</div><h2>Context that stays useful as your agent evolves.</h2><p>MemoryOS connects retrieval, reliability monitoring and recovery into one lightweight memory layer. Test the pipeline, inspect retrieval signals and keep stale context under control.</p><div class="split-points"><div><b>01</b>Relevant memories before every interaction</div><div><b>02</b>Explainable retrieval signals</div><div><b>03</b>Persistent local memory state</div></div></div><div class="split-visual"><div class="visual-grid"></div><div class="visual-core">✣</div><span>AI MEMORY<br>INFRASTRUCTURE</span></div></div>',unsafe_allow_html=True)

    st.markdown('<div class="landing-metrics"><div><strong>'+str(len(memories))+'</strong><span>MEMORY ITEMS</span></div><div><strong>'+f'{last_score:.2f}'+'</strong><span>DRIFT SCORE</span></div><div><strong>'+str(len(heal_history))+'</strong><span>HEAL EVENTS</span></div><div><strong>'+str(len(drift_history))+'</strong><span>SESSIONS</span></div></div>',unsafe_allow_html=True)

    st.markdown('''
    <section class="landing-bottom-cta">
      <div class="bottom-cta-inner">
        <div class="bottom-cta-main">
          <div class="landing-eyebrow">AGENT MEMORY CONSOLE</div>
          <h2>Give your agent a memory it can trust.</h2>
        </div>
        <div class="landing-cta-copy">
          Run retrieval, inspect context, analyze drift and execute recovery from the navigation above.
        </div>
      </div>
    </section>
    ''', unsafe_allow_html=True)

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
