import os
import glob
import random
import sys
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import roc_curve, auc
from deepface import DeepFace
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import cv2
import torch
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ───── SETUP LOGGING ─────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
# ─────────────────────────

# ───── CONFIG ─────
DATASET_DIR = os.path.join('Data', 'Face')
OUTPUT_DIR   = 'results'
FMR_TARGETS  = [0.001, 0.01, 0.05]
IMAGE_EXTS   = ('.bmp', '.jpg', '.jpeg', '.png') #I know all our images are bmp but this makes the code more versatile
LOG_EVERY    = 100   # log progress every N items
NUM_THREADS  = 4     # parallel threads for embedding
# ─────────────────────

def load_dataset(folder):
    logger.info(f"Loading dataset from {folder!r}...")
    paths, labels = [], []
    for person in sorted(os.listdir(folder)):
        pdir = os.path.join(folder, person)
        if not os.path.isdir(pdir):
            continue
        for f in glob.glob(os.path.join(pdir, '*')):
            if f.lower().endswith(IMAGE_EXTS):
                paths.append(f)
                labels.append(person)
    logger.info(f"Found {len(paths)} images, {len(set(labels))} identities.")
    return paths, labels

def make_pairs(label_map):
    logger.info("Generating genuine/impostor pairs...")
    genuine, impostor = [], []
    for imgs in label_map.values():
        for i in range(len(imgs)):
            for j in range(i+1, len(imgs)):
                genuine.append((imgs[i], imgs[j]))
    keys = list(label_map.keys())
    while len(impostor) < len(genuine):
        a, b = random.sample(keys, 2)
        impostor.append((random.choice(label_map[a]), random.choice(label_map[b])))
    logger.info(f"{len(genuine)} genuine, {len(impostor)} impostor pairs.")
    return genuine, impostor

def init_models():
    logger.info("Initializing models...")
    pt_fn = InceptionResnetV1(pretrained='vggface2').eval()
    logger.info("PyTorch FaceNet ready.")
    return pt_fn

def embed_pytorch(path, model):
    img = cv2.imread(path)
    r = cv2.resize(img, (160,160))
    rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
    t = torch.tensor(rgb).permute(2,0,1).unsqueeze(0).float()
    t = fixed_image_standardization(t)
    with torch.no_grad():
        return model(t).squeeze().numpy()

def score_opencv_gray(g1, g2):
    return -np.linalg.norm(g1 - g2)

def main():
    start = time.time()

    # 1) Load dataset
    try:
        paths, labels = load_dataset(DATASET_DIR)
    except Exception:
        logger.exception("Failed to load dataset")
        sys.exit(1)

    # 2) Build map label -> images
    label_map = defaultdict(list)
    for p, l in zip(paths, labels):
        label_map[l].append(p)
    valid = {l:imgs for l,imgs in label_map.items() if len(imgs) >= 2}
    if len(valid) < 2:
        logger.error("Need at least two identities with 2+ images each.")
        sys.exit(1)

    # 3) Make pairs
    try:
        genuine, impostor = make_pairs(valid)
    except Exception:
        logger.exception("Failed to generate pairs")
        sys.exit(1)
    pairs = genuine + impostor
    y_true = np.array([1]*len(genuine) + [0]*len(impostor))
    total_pairs = len(pairs)

    # 4) Init models
    try:
        pt_model = init_models()
    except Exception:
        logger.exception("Failed to initialize models")
        sys.exit(1)

    # 5) Cache grayscale
    logger.info("Caching grayscale images…")
    gray_cache = {}
    try:
        for i, p in enumerate(paths, 1):
            img = cv2.imread(p)
            gray_cache[p] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten().astype(np.float32)
            if i % LOG_EVERY == 0 or i == len(paths):
                logger.info(f"{i}/{len(paths)} grays cached")
    except Exception:
        logger.exception("Failed during grayscale caching")
        sys.exit(1)

    # 6) Build DeepFace Facenet model ONCE
    logger.info("Building DeepFace Facenet model…")
    try:
        df_model = DeepFace.build_model('Facenet')
        logger.info("DeepFace Facenet model ready.")
    except Exception:
        logger.exception("Failed to build DeepFace model")
        sys.exit(1)

    # 7) Cache DeepFace embeddings SEQUENTIALLY
    logger.info("Caching DeepFace embeddings…")
    df_cache = {}
    try:
        for i, p in enumerate(paths, 1):
            emb = DeepFace.represent(
                img_path=p,
                model_name='Facenet',
                enforce_detection=False
            )[0]['embedding']
            df_cache[p] = np.array(emb)
            if i % LOG_EVERY == 0 or i == len(paths):
                logger.info(f"DF {i}/{len(paths)} embeddings cached")
    except Exception:
        logger.exception("Failed during DeepFace embedding cache")
        sys.exit(1)

    # 8) Cache PyTorch embeddings in parallel
    logger.info("Caching PyTorch embeddings…")
    pt_cache = {}
    try:
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as ex:
            futures = {ex.submit(embed_pytorch, p, pt_model): p for p in paths}
            for i, fut in enumerate(as_completed(futures), 1):
                pt_cache[futures[fut]] = fut.result()
                if i % LOG_EVERY == 0 or i == len(paths):
                    logger.info(f"PT {i}/{len(paths)} embeddings cached")
    except Exception:
        logger.exception("Failed during PyTorch embedding cache")
        sys.exit(1)

    # 9) Score all pairs
    logger.info(f"Scoring {total_pairs} pairs…")
    scores = {'DeepFaceFN':[], 'FaceNetPT':[], 'OpenCV':[]}
    try:
        for i, (p1, p2) in enumerate(pairs, 1):
            e1, e2 = df_cache[p1], df_cache[p2]
            scores['DeepFaceFN'].append(e1.dot(e2)/(np.linalg.norm(e1)*np.linalg.norm(e2)))
            f1, f2 = pt_cache[p1], pt_cache[p2]
            scores['FaceNetPT'].append(f1.dot(f2)/(np.linalg.norm(f1)*np.linalg.norm(f2)))
            g1, g2 = gray_cache[p1], gray_cache[p2]
            scores['OpenCV'].append(score_opencv_gray(g1, g2))
            if i % LOG_EVERY == 0 or i == total_pairs:
                elapsed = time.time() - start
                logger.info(f"{i}/{total_pairs} scored | {elapsed:.1f}s")
    except Exception:
        logger.exception("Failed during scoring")
        sys.exit(1)

    # 10) Static ROC
    logger.info("Plotting static ROC…")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    records = []
    try:
        plt.figure(figsize=(8,6))
        for t, scr in scores.items():
            fpr, tpr, thr = roc_curve(y_true, scr, pos_label=1)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{t} (AUC={roc_auc:.2f})')
            for tgt in FMR_TARGETS:
                idx = np.argmin(np.abs(fpr - tgt))
                records.append({
                    'Tool': t,
                    'FMR_target': tgt,
                    'Threshold': thr[idx],
                    'FMR': float(fpr[idx]),
                    'FNMR': float(1 - tpr[idx])
                })
        plt.xlabel('False Match Rate (FMR) [%]')
        plt.ylabel('True Match Rate (TPR) [%]')
        plt.title('ROC Comparison')
        plt.legend(loc='lower right')
        plt.grid(True)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=2))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=2))

        plt.savefig(os.path.join(OUTPUT_DIR, 'roc_comparison.png'))
        pd.DataFrame(records).to_csv(os.path.join(OUTPUT_DIR, 'fmr_fnmr_table.csv'), index=False)
    except Exception:
        logger.exception("Failed while plotting or saving static ROC")
        sys.exit(1)

    # # 11) Interactive ROC
    # logger.info("Saving interactive ROC…")
    # try:
    #     import plotly.express as px
    #     df_plot = []
    #     for t, scr in scores.items():
    #         fpr, tpr, _ = roc_curve(y_true, scr, pos_label=1)
    #         df_plot.append(pd.DataFrame({
    #             'Method':    t,
    #             'FMR (%)':   fpr,
    #             'TPR (%)':   tpr,
    #         }))
    #     df_plot = pd.concat(df_plot, ignore_index=True)
    #     fig = px.line(
    #         df_plot,
    #         x='FMR (%)', y='TPR (%)', color='Method',
    #         title='Interactive ROC Comparison'
    #     )
    #     fig.update_layout(
    #         xaxis_title='False Match Rate (FMR) [%]',
    #         yaxis_title='True Match Rate (TPR) [%]',
    #         legend_title_text='Tool',
    #         xaxis_tickformat='.2%', yaxis_tickformat='.2%'
    #     )
    #     html_path = os.path.join(OUTPUT_DIR, 'roc_comparison.html')
    #     fig.write_html(html_path, include_plotlyjs='cdn')
    #     logger.info(f"Saved interactive ROC -> {html_path}")
    # except ImportError:
    #     logger.warning("Plotly not installed; skipping interactive ROC.")
    # except Exception:
    #     logger.exception("Failed while generating interactive ROC")

    # 12) Wrap up
    total_min = (time.time() - start) / 60
    logger.info(f"Done in {total_min:.1f} min")
    logger.info(f"Static ROC  -> {OUTPUT_DIR}/roc_comparison.png")
    logger.info(f"Table       -> {OUTPUT_DIR}/fmr_fnmr_table.csv")

if __name__ == '__main__':
    main()
