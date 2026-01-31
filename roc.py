import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import roc_curve, auc
from deepface import DeepFace
from collections import defaultdict

# Import from our modules
from logger import logger
from utils import (
    DATASET_DIR, OUTPUT_DIR, FMR_TARGETS, LOG_EVERY,
    load_dataset, make_pairs, init_models, embed_pytorch, score_opencv_gray,
    cache_grayscale_images, cache_pytorch_embeddings, cosine_similarity
)

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
    valid = {l: imgs for l, imgs in label_map.items() if len(imgs) >= 2}
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
    y_true = np.array([1] * len(genuine) + [0] * len(impostor))
    total_pairs = len(pairs)

    # 4) Init models
    try:
        pt_model = init_models()
    except Exception:
        logger.exception("Failed to initialize models")
        sys.exit(1)

    # 5) Cache grayscale images
    try:
        gray_cache = cache_grayscale_images(paths)
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
    try:
        pt_cache = cache_pytorch_embeddings(paths, pt_model)
    except Exception:
        logger.exception("Failed during PyTorch embedding cache")
        sys.exit(1)

    # 9) Score all pairs
    logger.info(f"Scoring {total_pairs} pairs…")
    scores = {'DeepFace FaceNet': [], 'PyTorch FaceNet': [], 'OpenCV Grayscale': []}
    try:
        for i, (p1, p2) in enumerate(pairs, 1):
            # DeepFace scoring
            e1, e2 = df_cache[p1], df_cache[p2]
            scores['DeepFace FaceNet'].append(cosine_similarity(e1, e2))

            # PyTorch FaceNet scoring
            f1, f2 = pt_cache[p1], pt_cache[p2]
            scores['PyTorch FaceNet'].append(cosine_similarity(f1, f2))

            # OpenCV grayscale scoring
            g1, g2 = gray_cache[p1], gray_cache[p2]
            scores['OpenCV Grayscale'].append(score_opencv_gray(g1, g2))

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
        plt.figure(figsize=(8, 6))
        for t, scr in scores.items():
            fpr, tpr, thr = roc_curve(y_true, scr, pos_label=1)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{t} (AUC={roc_auc:.2f})')
            for tgt in FMR_TARGETS:
                idx = np.argmin(np.abs(fpr - tgt))
                records.append({
                    'Method': t,
                    'Target FMR': f'{tgt*100:.1f}%',
                    'Actual FMR': f'{fpr[idx]*100:.2f}%',
                    'FNMR': f'{(1-tpr[idx])*100:.2f}%',
                    'Threshold': f'{thr[idx]:.3f}'
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

    # 11) Wrap up
    total_min = (time.time() - start) / 60
    logger.info(f"Done in {total_min:.1f} min")
    logger.info(f"Static ROC  -> {OUTPUT_DIR}/roc_comparison.png")
    logger.info(f"Table       -> {OUTPUT_DIR}/fmr_fnmr_table.csv")

if __name__ == '__main__':
    main()