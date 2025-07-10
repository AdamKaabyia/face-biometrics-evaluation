import os
import sys
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

def generate_interactive_roc():
    """Generate interactive HTML ROC curves"""
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

    # 10) Generate Interactive ROC Plot
    logger.info("Generating interactive ROC curves…")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('ROC Curves Comparison', 'Performance Metrics'),
            specs=[[{"secondary_y": False}, {"type": "table"}]],
            column_widths=[0.7, 0.3]
        )

        # Color scheme
        colors = {
            'DeepFace FaceNet': '#1f77b4',
            'PyTorch FaceNet': '#ff7f0e',
            'OpenCV Grayscale': '#2ca02c'
        }

        records = []

        # Add ROC curves
        for method_name, scr in scores.items():
            fpr, tpr, thr = roc_curve(y_true, scr, pos_label=1)
            roc_auc = auc(fpr, tpr)

            # Add ROC curve trace
            fig.add_trace(
                go.Scatter(
                    x=fpr * 100,  # Convert to percentage
                    y=tpr * 100,  # Convert to percentage
                    mode='lines',
                    name=f'{method_name} (AUC={roc_auc:.3f})',
                    line=dict(color=colors[method_name], width=3),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'FMR: %{x:.2f}%<br>' +
                                'TMR: %{y:.2f}%<br>' +
                                '<extra></extra>'
                ),
                row=1, col=1
            )

            # Calculate performance metrics at target FMRs
            for tgt in FMR_TARGETS:
                idx = np.argmin(np.abs(fpr - tgt))
                records.append({
                    'Method': method_name,
                    'Target FMR': f'{tgt*100:.1f}%',
                    'Actual FMR': f'{fpr[idx]*100:.2f}%',
                    'FNMR': f'{(1-tpr[idx])*100:.2f}%',
                    'Threshold': f'{thr[idx]:.3f}'
                })

        # Add diagonal reference line
        fig.add_trace(
            go.Scatter(
                x=[0, 100],
                y=[0, 100],
                mode='lines',
                name='Random Classifier',
                line=dict(color='gray', width=1, dash='dash'),
                hovertemplate='Random Classifier<br>' +
                            'FMR: %{x:.2f}%<br>' +
                            'TMR: %{y:.2f}%<br>' +
                            '<extra></extra>'
            ),
            row=1, col=1
        )

        # Create performance table
        df_metrics = pd.DataFrame(records)

        # Add performance table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(df_metrics.columns),
                    fill_color='lightblue',
                    align='center',
                    font=dict(size=12, color='black')
                ),
                cells=dict(
                    values=[df_metrics[col] for col in df_metrics.columns],
                    fill_color='white',
                    align='center',
                    font=dict(size=11)
                )
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            title={
                'text': 'Face Recognition System Performance Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=0.7
            ),
            width=1400,
            height=700,
            template='plotly_white'
        )

        # Update ROC plot axes
        fig.update_xaxes(
            title_text="False Match Rate (FMR) [%]",
            range=[0, 100],
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="True Match Rate (TMR) [%]",
            range=[0, 100],
            row=1, col=1
        )

        # Add annotations
        total_min = (time.time() - start) / 60
        fig.add_annotation(
            text=f"Dataset: {len(genuine)} genuine pairs, {len(impostor)} impostor pairs<br>" +
                 f"Processing time: {total_min:.1f} minutes",
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            showarrow=False,
            font=dict(size=10, color="gray"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )

        # Save as HTML
        html_path = os.path.join(OUTPUT_DIR, 'index.html')
        fig.write_html(
            html_path,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
            }
        )

        logger.info(f"Interactive ROC -> {html_path}")

        # Also save the static PNG for compatibility
        png_path = os.path.join(OUTPUT_DIR, 'roc_comparison.png')
        fig.write_image(png_path, width=1400, height=700, scale=2)
        logger.info(f"Static ROC -> {png_path}")

        # Save metrics table as CSV
        csv_path = os.path.join(OUTPUT_DIR, 'fmr_fnmr_table.csv')
        df_metrics.to_csv(csv_path, index=False)
        logger.info(f"Metrics table -> {csv_path}")

    except Exception:
        logger.exception("Failed while generating interactive ROC")
        sys.exit(1)

    # 11) Wrap up
    total_min = (time.time() - start) / 60
    logger.info(f"Interactive ROC generation completed in {total_min:.1f} minutes")

def main():
    """Main function"""
    generate_interactive_roc()

if __name__ == '__main__':
    main()