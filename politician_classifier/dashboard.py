"""
dashboard.py — Premium Real-Time Pipeline + Training Monitor
Pakistani Politician Image Classification System
"""

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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.main                        { background-color: #0a0e1a; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d1226 0%,#111827 100%);
    border-right: 1px solid #1e293b;
}
h1,h2,h3,h4 { color:#f1f5f9 !important; font-weight:700 !important; letter-spacing:-0.5px; }

/* metric */
.metric-card { background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);
    border:1px solid #334155; border-radius:14px; padding:22px; text-align:center; }
.metric-value { font-size:2.2rem; font-weight:700; }
.metric-label { font-size:.78rem; color:#64748b; margin-top:4px;
    text-transform:uppercase; letter-spacing:1px; }

/* step cards */
.step-card { background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);
    border:1px solid #334155; border-radius:14px; padding:16px 20px;
    margin-bottom:9px; display:flex; align-items:center; gap:14px; }
.step-card.done    { border-left:4px solid #22c55e; }
.step-card.active  { border-left:4px solid #f59e0b;
    background:linear-gradient(135deg,#1c1a0f 0%,#0f172a 100%); }
.step-card.pending { border-left:4px solid #475569; opacity:.6; }
.step-icon  { font-size:1.6rem; }
.step-name  { font-size:.95rem; font-weight:600; color:#e2e8f0; margin:0; }
.step-detail{ font-size:.75rem; color:#64748b; margin:2px 0 0 0; }
.badge-done   { background:#14532d; color:#4ade80; padding:3px 12px;
    border-radius:999px; font-size:.72rem; font-weight:600; }
.badge-active { background:#451a03; color:#fbbf24; padding:3px 12px;
    border-radius:999px; font-size:.72rem; font-weight:600;
    animation:pulse 1.5s infinite; }
.badge-pending{ background:#1e293b; color:#475569; padding:3px 12px;
    border-radius:999px; font-size:.72rem; font-weight:600; }
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}

/* log box */
.log-box { background:#020617; border:1px solid #1e293b; border-radius:10px;
    padding:12px; font-family:'Courier New',monospace; font-size:.72rem;
    color:#94a3b8; max-height:160px; overflow-y:auto; white-space:pre-wrap; }

/* section header */
.sec-hdr { background:linear-gradient(90deg,#1e3a5f 0%,transparent 100%);
    border-left:3px solid #3b82f6; padding:7px 16px;
    border-radius:0 8px 8px 0; margin:22px 0 14px 0; }
.sec-hdr p { margin:0; color:#93c5fd; font-size:.9rem; font-weight:600; }

/* epoch table */
.ep-row-best { color:#4ade80 !important; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── PATHS ────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
RAW    = BASE / "dataset" / "raw"
CLEAN  = BASE / "dataset" / "cleaned"
STATS  = BASE / "stats"
MODELS = BASE / "models"
LOGS   = BASE / "logs"

POLITICIANS = [
    "imran_khan","nawaz_sharif","shahbaz_sharif","maryam_nawaz",
    "bilawal_bhutto","asif_ali_zardari","fazlur_rehman","sheikh_rasheed",
    "mohsin_naqvi","hina_rabbani_khar","murad_ali_shah","ali_amin_gandapur",
    "khawaja_asif","attaullah_tarar","chaudhry_pervaiz_elahi","ahmed_sharif_chaudhry"
]

PIPELINE_STEPS = [
    (10,"Image Crawling",          "01_download.log",               "Multi-source harvesting (Google, Bing, news)"),
    (20,"Initial Cleaning",        "02_clean.log",                  "Blur / duplicate / corrupt removal"),
    (30,"Face Extraction (MTCNN)", "03_intelligent_cleaner.log",    "Single-face isolation — group photos purged"),
    (40,"Anchor Generation",       "03b_auto_ref.log",              "First-image anchor heuristic per class"),
    (50,"Identity Verification",   "03a_identity_verification.log", "Biometric purification via Facenet512"),
    (60,"Dataset Split",           "04_split.log",                  "75 % Train · 15 % Val · 10 % Test"),
    (70,"Visualisation",           "05_visualize.log",              "Sample grids & class distribution"),
    (80,"CNN Training",            "06_training.log",               "ResNet50 + EfficientNetB0 — 25 epochs"),
]

ICONS = {10:"📥",20:"🧹",30:"👁️",40:"⚓",50:"🛡️",60:"✂️",70:"📊",80:"🧠"}

# ── HELPERS ──────────────────────────────────────────────────────────────────
def count_clean(slug):
    p = CLEAN / slug
    if not p.exists(): return 0
    return len([f for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in {'.jpg','.jpeg','.png'}
                and not f.name.startswith('_')])

def count_raw(slug):
    p = RAW / slug
    return len(list(p.iterdir())) if p.exists() else 0

def read_pipeline_log():
    log = LOGS / "pipeline.log"
    completed, active = set(), None
    if not log.exists(): return completed, active
    for line in log.read_text(errors="ignore").splitlines():
        m = re.search(r">> STEP (\d+)", line)
        if m: active = int(m.group(1))
        if "Step completed" in line and active:
            completed.add(active); active = None
    return completed, active

def parse_verification_log():
    log = LOGS / "03a_identity_verification.log"
    results = {}
    if not log.exists(): return results
    for line in log.read_text(errors="ignore").splitlines():
        m = re.search(r"\[(\w+)\] Verified: (\d+) \| Rejected: (\d+)", line)
        if m: results[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return results

def tail_log(filename, n=10):
    log = LOGS / filename
    if not log.exists(): return "No log yet."
    lines = log.read_text(errors="ignore").splitlines()
    return "\n".join(lines[-n:]) if lines else "Empty."

def load_training_progress(model_name):
    """Load epoch-by-epoch history from the JSON written by ProgressCallback."""
    p = STATS / f"{model_name}_progress.json"
    if not p.exists(): return None
    try:
        with open(p) as f: return json.load(f)
    except: return None

def load_report(model_name):
    p = STATS / f"{model_name}_report.json"
    if not p.exists(): return None
    try:
        with open(p) as f: return json.load(f)
    except: return None

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI Pipeline Monitor")
    st.markdown("**Pakistani Politician**  \nImage Classification")
    st.markdown("---")
    st.markdown("**Architectures:** ResNet50 · EfficientNetB0")
    st.markdown("**Verification:** Facenet512 · MTCNN")
    st.markdown("**Split:** 75 / 15 / 10 %")
    st.markdown("**Augmentation:** Flip · Rotate · Brightness · Zoom · Crop")
    st.markdown("---")
    page = st.radio("Navigate", ["📊 Pipeline Overview","🧠 Training Monitor","📋 Evaluation Metrics"])
    st.markdown("---")
    auto = st.checkbox("🔄 Auto-refresh (15 s)", value=True)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#1e3a5f,#0a0e1a);
     padding:26px 30px;border-radius:16px;border:1px solid #1e3a8a;margin-bottom:20px;'>
  <h1 style='margin:0;font-size:1.7rem;color:#f1f5f9;'>
    🇵🇰 Pakistani Politician Image Classification
  </h1>
  <p style='margin:6px 0 0;color:#64748b;font-size:.88rem;'>
    CNN-Based Multi-Class Facial Recognition · Transfer Learning Pipeline · Real-Time Monitor
  </p>
</div>""", unsafe_allow_html=True)

# ── TOP METRICS ───────────────────────────────────────────────────────────────
total_raw     = sum(count_raw(p)   for p in POLITICIANS)
total_clean   = sum(count_clean(p) for p in POLITICIANS)
classes_ok    = sum(1 for p in POLITICIANS if count_clean(p) >= 80)
models_done   = sum(1 for m in ["ResNet50","EfficientNetB0"]
                    if (MODELS/f"{m}_best.h5").exists())
resnet_report = load_report("ResNet50")
best_acc      = f"{resnet_report.get('accuracy','—'):.1%}" if resnet_report else "—"

cols = st.columns(5)
for col,(val,lbl,color) in zip(cols,[
    (f"{total_raw:,}",  "Raw Images Harvested",    "#8b5cf6"),
    (f"{total_clean:,}","Cleaned Faces (Verified)", "#06b6d4"),
    (f"{classes_ok}/16","Classes ≥ 80 Images",      "#22c55e"),
    (f"{models_done}/2","Models Trained",            "#f59e0b"),
    (best_acc,          "Best Test Accuracy",        "#ef4444"),
]):
    with col:
        st.markdown(f"""<div class='metric-card'>
          <div class='metric-value' style='color:{color}'>{val}</div>
          <div class='metric-label'>{lbl}</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Pipeline Overview":
# ══════════════════════════════════════════════════════════════════════════════

    st.markdown("<div class='sec-hdr'><p>⚙️  Pipeline Step Progress</p></div>", unsafe_allow_html=True)
    completed, active = read_pipeline_log()
    cl, cr = st.columns(2)
    for i,(num,name,logf,desc) in enumerate(PIPELINE_STEPS):
        icon = ICONS.get(num,"🔵")
        if num in completed:
            cls,badge = "done",  "<span class='badge-done'>✅ Done</span>"
        elif num == active:
            cls,badge = "active","<span class='badge-active'>⚡ Running</span>"
        else:
            cls,badge = "pending","<span class='badge-pending'>⏳ Pending</span>"
        html = f"""<div class='step-card {cls}'>
          <div class='step-icon'>{icon}</div>
          <div style='flex:1'>
            <p class='step-name'>Step {num} — {name}</p>
            <p class='step-detail'>{desc}</p>
          </div>{badge}</div>"""
        (cl if i%2==0 else cr).markdown(html, unsafe_allow_html=True)

    st.markdown("<div class='sec-hdr'><p>📁  Per-Class Dataset Status</p></div>", unsafe_allow_html=True)
    ver = parse_verification_log()
    rows = []
    for slug in POLITICIANS:
        raw   = count_raw(slug)
        clean = count_clean(slug)
        v,r   = ver.get(slug,("-","-"))
        req   = ("✅ Met" if clean>=80 else (f"🔄 {clean}/80" if clean>0 else "❌ Empty"))
        rows.append({
            "Politician": slug.replace("_"," ").title(),
            "Raw": raw, "Cleaned": clean,
            "Verified ✔ | Rejected ✖": f"{v} | {r}" if isinstance(v,int) else "—",
            "Requirement": req,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True,
        column_config={
            "Politician":              st.column_config.TextColumn(width="medium"),
            "Raw":                     st.column_config.NumberColumn(format="%d"),
            "Cleaned":                 st.column_config.NumberColumn(format="%d"),
            "Verified ✔ | Rejected ✖":st.column_config.TextColumn(),
            "Requirement":             st.column_config.TextColumn(width="small"),
        })
    pct = classes_ok/16
    st.markdown(f"**{classes_ok}/16 classes** meet the 80-image minimum — `{pct*100:.0f}%`")
    st.progress(pct)

    st.markdown("<div class='sec-hdr'><p>📡  Live Log Feed</p></div>", unsafe_allow_html=True)
    la,lb,lc = st.columns(3)
    with la:
        st.markdown("**🛡️ Identity Verification**")
        st.markdown(f"<div class='log-box'>{tail_log('03a_identity_verification.log')}</div>", unsafe_allow_html=True)
    with lb:
        st.markdown("**🧠 CNN Training**")
        st.markdown(f"<div class='log-box'>{tail_log('06_training.log')}</div>", unsafe_allow_html=True)
    with lc:
        st.markdown("**⚙️ Pipeline Master**")
        st.markdown(f"<div class='log-box'>{tail_log('pipeline.log')}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Training Monitor":
# ══════════════════════════════════════════════════════════════════════════════

    for model in ["ResNet50","EfficientNetB0"]:
        hist = load_training_progress(model)
        st.markdown(f"<div class='sec-hdr'><p>🧠  {model} — Epoch-by-Epoch Progress</p></div>", unsafe_allow_html=True)

        if hist is None:
            st.info(f"⏳ {model} training not started yet — waiting for dataset split to complete.")
            continue

        epochs_done = len(hist.get("accuracy",[]))
        train_acc   = hist.get("accuracy",[])
        val_acc     = hist.get("val_accuracy",[])
        train_loss  = hist.get("loss",[])
        val_loss    = hist.get("val_loss",[])
        best_epoch  = int(max(range(len(val_acc)), key=lambda i: val_acc[i])) if val_acc else 0

        # Summary metrics
        mc1,mc2,mc3,mc4 = st.columns(4)
        with mc1: st.metric("Epochs Done",    f"{epochs_done}/25")
        with mc2: st.metric("Best Val Acc",   f"{max(val_acc)*100:.2f}%" if val_acc else "—",
                            delta=f"Epoch {best_epoch+1}")
        with mc3: st.metric("Latest Train Acc",f"{train_acc[-1]*100:.2f}%" if train_acc else "—")
        with mc4: st.metric("Latest Val Loss", f"{val_loss[-1]:.4f}" if val_loss else "—")

        # Plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor="#0f172a")
        for ax in axes:
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#94a3b8")
            ax.spines[:].set_color("#334155")
            for label in ax.get_xticklabels()+ax.get_yticklabels():
                label.set_color("#94a3b8")

        ep = list(range(1, epochs_done+1))
        # Accuracy
        axes[0].plot(ep, [a*100 for a in train_acc], color="#3b82f6", lw=2, label="Train Acc")
        axes[0].plot(ep, [a*100 for a in val_acc],   color="#22c55e", lw=2, label="Val Acc", linestyle="--")
        axes[0].axvline(best_epoch+1, color="#f59e0b", linestyle=":", alpha=.8, label=f"Best (E{best_epoch+1})")
        axes[0].set_title("Accuracy (%)", color="#f1f5f9", fontweight="bold")
        axes[0].set_xlabel("Epoch", color="#94a3b8")
        axes[0].legend(facecolor="#1e293b", labelcolor="#e2e8f0", framealpha=.8)
        axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

        # Loss
        axes[1].plot(ep, train_loss, color="#ef4444", lw=2, label="Train Loss")
        axes[1].plot(ep, val_loss,   color="#f97316", lw=2, label="Val Loss", linestyle="--")
        axes[1].set_title("Loss", color="#f1f5f9", fontweight="bold")
        axes[1].set_xlabel("Epoch", color="#94a3b8")
        axes[1].legend(facecolor="#1e293b", labelcolor="#e2e8f0", framealpha=.8)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Epoch table
        ep_rows = []
        for i in range(epochs_done):
            ep_rows.append({
                "Epoch": i+1,
                "Train Acc %": f"{train_acc[i]*100:.2f}",
                "Val Acc %":   f"{val_acc[i]*100:.2f}",
                "Train Loss":  f"{train_loss[i]:.4f}",
                "Val Loss":    f"{val_loss[i]:.4f}",
                "Best": "⭐" if i == best_epoch else "",
            })
        ep_df = pd.DataFrame(ep_rows)
        st.dataframe(ep_df, use_container_width=True, hide_index=True, height=280)
        st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Evaluation Metrics":
# ══════════════════════════════════════════════════════════════════════════════

    for model in ["ResNet50","EfficientNetB0"]:
        report = load_report(model)
        st.markdown(f"<div class='sec-hdr'><p>📋  {model} — Evaluation Metrics</p></div>",
                    unsafe_allow_html=True)
        if report is None:
            st.info(f"⏳ {model} evaluation not complete yet.")
            continue

        overall_acc = report.get("accuracy", 0)
        m1,m2,m3 = st.columns(3)
        with m1: st.metric("Overall Accuracy",  f"{overall_acc:.1%}")
        with m2: st.metric("Macro F1-Score",    f"{report.get('macro avg',{}).get('f1-score',0):.1%}")
        with m3: st.metric("Weighted F1-Score", f"{report.get('weighted avg',{}).get('f1-score',0):.1%}")

        # Per-class report
        skip = {"accuracy","macro avg","weighted avg"}
        per_class = {k:v for k,v in report.items() if k not in skip and isinstance(v,dict)}
        per_rows = []
        for cls, metrics in per_class.items():
            per_rows.append({
                "Class":     cls.replace("_"," ").title(),
                "Precision": f"{metrics.get('precision',0):.1%}",
                "Recall":    f"{metrics.get('recall',0):.1%}",
                "F1-Score":  f"{metrics.get('f1-score',0):.1%}",
                "Support":   int(metrics.get("support",0)),
            })
        if per_rows:
            st.dataframe(pd.DataFrame(per_rows), use_container_width=True, hide_index=True)

        # Confusion matrix image
        cm_path = STATS / f"{model}_cm.png"
        if cm_path.exists():
            st.image(str(cm_path), caption=f"{model} Confusion Matrix",
                     use_column_width=True)

        # Misclassification
        mis_path = STATS / f"{model}_misclassified.png"
        if mis_path.exists():
            st.markdown("**🔬 Top-5 High-Confidence Misclassifications**")
            st.image(str(mis_path), use_column_width=True)
        st.markdown("---")

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(
    f"<center style='color:#334155;font-size:.78rem;'>"
    f"Pakistani Politician Classifier · CNN Transfer Learning · "
    f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}</center>",
    unsafe_allow_html=True
)

if auto:
    time.sleep(15)
    st.rerun()
