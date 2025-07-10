import os
import shutil
import zipfile
import random
import time

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import plotly.express as px

from collections import defaultdict
from io import StringIO
from sklearn.metrics import roc_curve

# ───── Upload Limit Reminder ─────
st.warning(
    "⚠️ To allow uploads up to 5 GB, run with:\n"
    "`streamlit run user_interface.py --server.maxUploadSize=5000`",
    icon="⚙️"
)

# ───── Logging ─────
import logging
log_stream = StringIO()
stream_handler = logging.StreamHandler(log_stream)
stream_handler.setLevel(logging.INFO)
root_logger = logging.getLogger()
root_logger.handlers = []
root_logger.addHandler(stream_handler)
root_logger.setLevel(logging.INFO)

# ───── Core Imports ─────
from main import load_dataset, make_pairs, init_models, embed_pytorch, logger
from morph import build_caches, run_morph_attack

logger.handlers = []
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

# ───── Streamlit Setup ─────
st.set_page_config(page_title="Face Biometrics Evaluation Dashboard", layout="wide")
st.title("🧠 Face Biometrics Evaluation")

# ───── Sidebar: Upload & Config ─────
st.sidebar.header("1️⃣ Dataset Upload & Config")
uploaded_zip = st.sidebar.file_uploader("Upload face dataset (.zip)", type=["zip"])
dataset_dir = "Data/Face"
NUM_THREADS = st.sidebar.slider("Embedding Threads", 1, 8, 4)
ALPHA = st.sidebar.slider("Morph Blend (α)", 0.0, 1.0, 0.5)
THRESHOLD = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.7)
NUM_MORPHS = st.sidebar.slider("Number of Morph Tests", 10, 500, 100)

# ───── Upload Extract ─────
if uploaded_zip:
    with st.spinner("Extracting dataset..."):
        tmp_dir = os.path.join("Data", "uploaded")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
        with zipfile.ZipFile(uploaded_zip) as z:
            z.extractall(tmp_dir)
        dataset_dir = tmp_dir
    st.success("✅ Dataset uploaded and extracted.")

# ───── Pairing ─────
@st.cache_data(show_spinner=False)
def load_and_pair(dataset_dir):
    paths, labels = load_dataset(dataset_dir)
    by_label = defaultdict(list)
    for p, l in zip(paths, labels):
        by_label[l].append(p)
    valid = {l: imgs for l, imgs in by_label.items() if len(imgs) >= 2}
    genuine, impostor = make_pairs(valid)
    pairs = genuine + impostor
    y_true = np.array([1] * len(genuine) + [0] * len(impostor))
    return paths, pairs, y_true, genuine

paths, pairs, y_true, genuine = load_and_pair(dataset_dir)

# ───── Helper: Logs ─────
def update_logs(box):
    box.text(log_stream.getvalue())

# ───── Caches ─────
st.subheader("2️⃣ Build Feature Caches")
if st.button("🔨 Pre-build all caches"):
    log_stream.truncate(0)
    log_stream.seek(0)
    log_box = st.empty()

    with st.spinner("Building caches..."):
        gray_cache = {
            p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
                     .flatten().astype(np.float32)
            for p in paths
        }
        st.session_state['gray_cache'] = gray_cache

        good_paths, landmark_cache, df_cache, pt_model, pt_cache = build_caches(paths)
        st.session_state.update({
            'good_paths': good_paths,
            'landmark_cache': landmark_cache,
            'df_cache': df_cache,
            'pt_model': pt_model,
            'pt_cache': pt_cache
        })

        update_logs(log_box)

    st.success("✅ Caches built successfully.")
    st.text_area("Cache Build Logs", log_stream.getvalue(), height=300)

# ───── ROC ─────
st.subheader("3️⃣ ROC Evaluation")
if st.button("📈 Run ROC Analysis"):
    if 'good_paths' not in st.session_state:
        st.error("❌ You must run 'Pre-build caches' first.")
    else:
        def plot_roc(pairs, y_true, df_cache, pt_cache, gray_cache):
            df_plot = []
            for name, cache in [
                ("DeepFace", df_cache),
                ("FaceNet", pt_cache),
                ("OpenCV", gray_cache)
            ]:
                scores = []
                filtered_y = []
                for (p1, p2), y in zip(pairs, y_true):
                    if p1 not in cache or p2 not in cache:
                        continue
                    e1, e2 = cache[p1], cache[p2]
                    s = -np.linalg.norm(e1 - e2) if name == "OpenCV" else np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
                    scores.append(s)
                    filtered_y.append(y)
                fpr, tpr, _ = roc_curve(filtered_y, scores, pos_label=1)
                df_plot.append(pd.DataFrame({"Tool": name, "FMR": fpr, "TPR": tpr}))
            df = pd.concat(df_plot, ignore_index=True)
            fig = px.line(df, x="FMR", y="TPR", color="Tool", title="ROC Curve")
            fig.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%")
            return fig

        fig = plot_roc(pairs, y_true,
                       st.session_state['df_cache'],
                       st.session_state['pt_cache'],
                       st.session_state['gray_cache'])
        st.plotly_chart(fig, use_container_width=True)

# ───── Morph Attack ─────
st.subheader("4️⃣ Morphing Attack Demo")
if st.button("🌀 Run Morph Attack"):
    if 'good_paths' not in st.session_state:
        st.error("❌ You must run 'Pre-build caches' first.")
    else:
        log_stream.truncate(0)
        log_stream.seek(0)
        log_box = st.empty()

        st.info("Running morph attacks on genuine pairs...")
        filtered = [(a, b) for (a, b) in genuine
                    if a in st.session_state['good_paths'] and b in st.session_state['good_paths']]
        test_pairs = random.sample(filtered, k=min(NUM_MORPHS, len(filtered)))

        rate = run_morph_attack(
            test_pairs,
            (
                st.session_state['good_paths'],
                st.session_state['landmark_cache'],
                st.session_state['df_cache'],
                st.session_state['pt_model'],
                st.session_state['pt_cache']
            ),
            alpha=ALPHA,
            threshold=THRESHOLD
        )

        update_logs(log_box)
        st.success(f"✅ Morph Attack Success Rate: {rate*100:.2f}%")
        st.text_area("Morph Logs", log_stream.getvalue(), height=300)
