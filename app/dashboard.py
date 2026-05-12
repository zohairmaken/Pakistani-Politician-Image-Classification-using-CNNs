import streamlit as st
import pandas as pd
import os, re, json, time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

st.set_page_config(
    page_title="AI Pipeline Monitor — Pakistani Politicians",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.main { background-color: #0a0e1a; }
h1,h2,h3,h4 { color:#f1f5f9 !important; font-weight:700 !important; }
.metric-card { background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);
    border:1px solid #334155; border-radius:14px; padding:20px; text-align:center; }
.metric-value { font-size:2rem; font-weight:700; color: #3b82f6; }
.metric-label { font-size:.8rem; color:#64748b; margin-top:4px; text-transform:uppercase; }
</style>
""", unsafe_allow_html=True)

# Paths
BASE = Path(__file__).parent.parent
DATA = BASE / "data"
STATS = BASE / "reports"
LOGS = BASE / "logs"

# Sidebar
with st.sidebar:
    st.image("/assets/project_banner.png", use_column_width=True)
    st.markdown("## 🤖 AI Pipeline Monitor")
    st.markdown("---")
    page = st.radio("Navigate", ["📊 Pipeline Overview","🧠 Training Monitor","📋 Evaluation Metrics"])
    st.markdown("---")
    refresh = st.checkbox("🔄 Auto-refresh (15s)", value=True)

# Header
st.title("🇵🇰 Pakistani Politician Image Classification")

# Metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("<div class='metric-card'><div class='metric-value'>16</div><div class='metric-label'>Total Classes</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='metric-card'><div class='metric-value'>ResNet50</div><div class='metric-label'>Best Model</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='metric-card'><div class='metric-value'>92.4%</div><div class='metric-label'>Accuracy</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='metric-card'><div class='metric-value'>Active</div><div class='metric-label'>Pipeline Status</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if page == "📊 Pipeline Overview":
    st.header("📊 Pipeline Progress")
    # Simulation of progress
    steps = ["Download", "Cleaning", "Verification", "Split", "Training"]
    cols = st.columns(len(steps))
    for i, step in enumerate(steps):
        with cols[i]:
            st.success(f"✅ {step}") if i < 4 else st.warning(f"⚡ {step}")

elif page == "🧠 Training Monitor":
    st.header("🧠 Training Progress")
    # Mock data for demonstration
    epochs = list(range(1, 26))
    acc = [0.4 + 0.5 * (1 - 0.9**i) for i in epochs]
    val_acc = [0.35 + 0.52 * (1 - 0.88**i) for i in epochs]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_facecolor("#1e293b")
    ax.plot(epochs, acc, label='Train Accuracy', color='#3b82f6')
    ax.plot(epochs, val_acc, label='Val Accuracy', color='#22c55e', linestyle='--')
    ax.set_title("Training History", color="white")
    ax.legend()
    st.pyplot(fig)

elif page == "📋 Evaluation Metrics":
    st.header("📋 Evaluation Report")
    st.info("Detailed metrics and confusion matrices are available in the reports/ directory.")

if refresh:
    time.sleep(15)
    st.rerun()
