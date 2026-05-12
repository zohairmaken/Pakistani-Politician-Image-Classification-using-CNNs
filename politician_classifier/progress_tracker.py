"""
progress_tracker.py
===================
A live dashboard to track the progress of the image classification pipeline.
Shows which politicians are "Done", "In Progress", or "Pending".
"""

import streamlit as st
import os
import time
import pandas as pd
from pathlib import Path
import re

# --- Configuration & Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw")
CLEANED_DIR = os.path.join(DATASET_DIR, "cleaned")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

st.set_page_config(page_title="Pipeline Progress", page_icon="⏳", layout="wide")

st.markdown("""
    <style>
    .status-done { color: #28a745; font-weight: bold; }
    .status-progress { color: #ffc107; font-weight: bold; }
    .status-pending { color: #6c757d; font-style: italic; }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background: #1e1e1e;
        border: 1px solid #333;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

def get_politicians():
    # Dynamically get list from raw folder
    if os.path.exists(RAW_DIR):
        return sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))])
    return []

def get_latest_log_line(log_name):
    path = os.path.join(LOGS_DIR, log_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()
            if lines:
                return lines[-1]
    return ""

def get_class_status():
    pol_list = get_politicians()
    stats = []
    
    # Read cleaner log for detailed status
    log_path = os.path.join(LOGS_DIR, "03_intelligent_cleaner.log")
    log_mtime = os.path.getmtime(log_path) if os.path.exists(log_path) else 0
    
    cleaner_log = ""
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            cleaner_log = f.read()

    for pol in pol_list:
        raw_count = len(os.listdir(os.path.join(RAW_DIR, pol))) if os.path.exists(os.path.join(RAW_DIR, pol)) else 0
        cleaned_path = os.path.join(CLEANED_DIR, pol)
        cleaned_count = len(os.listdir(cleaned_path)) if os.path.exists(cleaned_path) else 0
        
        # Check if "Done" in logs from TODAY or current session
        # For simplicity, we check if the entry exists and cleaned_count is significant
        is_done = f"[{pol}]" in cleaner_log and cleaned_count >= 80
        
        # Check if currently being processed (last line of log)
        latest_line = get_latest_log_line("03_intelligent_cleaner.log")
        is_active = f"Cleaning: {pol}" in latest_line or (not is_done and cleaned_count > 0 and not f"[{pol}]" in cleaner_log)

        status = "⏳ Pending"
        if is_done:
            status = "✅ Done (>80 imgs)"
        elif is_active:
            status = "🔄 In Progress"
        elif cleaned_count > 0 and cleaned_count < 80:
             status = "⚠️ Low Data (Redownloading)"
            
        stats.append({
            "Politician": pol.replace("_", " ").title(),
            "Status": status,
            "Raw Images": raw_count,
            "Cleaned Faces": cleaned_count
        })
    return stats

st.title("🦅 Pipeline Progress Tracker")
st.write("Real-time monitoring of dataset cleaning and face extraction.")

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    refresh_rate = st.slider("Refresh Rate (seconds)", 2, 30, 5)
    st.info("The dashboard refreshes automatically.")

# Main Layout
stats = get_class_status()
df = pd.DataFrame(stats)

# Overall Progress
total = len(stats)
done = sum(1 for s in stats if "✅" in s["Status"])
prog = sum(1 for s in stats if "🔄" in s["Status"])

col1, col2, col3 = st.columns(3)
col1.metric("Total Classes", total)
col2.metric("Completed", done, f"{done/total:.0%}")
col3.metric("In Progress", prog)

st.progress(done / total if total > 0 else 0)

# Grid View
st.subheader("Class-wise Breakdown")
cols = st.columns(4)
for i, row in enumerate(stats):
    with cols[i % 4]:
        st.markdown(f"""
            <div class="card">
                <div style="font-size: 1.1rem; font-weight: 600;">{row['Politician']}</div>
                <div style="margin-top: 0.5rem;">{row['Status']}</div>
                <div style="font-size: 0.8rem; color: #888; margin-top: 0.5rem;">
                    Raw: {row['Raw Images']} | Cleaned: {row['Cleaned Faces']}
                </div>
            </div>
        """, unsafe_allow_html=True)

# Latest Activity
st.subheader("Latest Pipeline Activity")
latest_log = get_latest_log_line("pipeline.log")
st.code(latest_log if latest_log else "Initializing...")

# Auto-refresh
time.sleep(refresh_rate)
st.rerun()
