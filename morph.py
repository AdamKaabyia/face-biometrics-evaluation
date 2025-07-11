from __future__ import annotations
import os, random, time, json, csv, tempfile, pathlib, logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace

# Import from our modules
from logger import logger
from utils import (
    OUTPUT_DIR, load_dataset, make_pairs, init_models, embed_pytorch,
    score_opencv_gray, cosine_similarity
)

# Configuration
DATASET_DIR   = os.path.join("Data", "Face")
CANVAS_SIZE   = (512, 512)          # warp target size (W, H)
MAX_PAIRS     = 100                 # morphs to evaluate
N_THREADS     = 4                   # for embedding / evaluation
ALPHA         = 0.5                 # blend factor for morphing
PROGRESS_STEP = 100                 # log every N items

THRESHOLDS: Dict[str, float] = {
    "pytorch"     : 0.70,   # cosine ≥ threshold then accept
    "deepface"    : 0.70,
    "opencv_gray" : -0.50   # distance ≥ –0.50 then accept (note sign)
}

# Landmark extractor
_mesh  = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
IDX_68 = np.linspace(0, 467, 68, dtype=int)          # down-sample to 68 pts
W, H   = CANVAS_SIZE


def extract_landmarks(img_bgr: np.ndarray) -> np.ndarray:
    """Return 68 normalized (x, y) landmarks in [0, 1]."""
    res = _mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        raise RuntimeError("no face detected")
    lm0 = res.multi_face_landmarks[0]
    return np.array([(lm0.landmark[i].x, lm0.landmark[i].y) for i in IDX_68],
                    dtype=np.float32)

# Triangle-warp helpers
def _warp_triangle(src, tri_src, tri_dst):
    M = cv2.getAffineTransform(tri_src.astype(np.float32),
                               tri_dst.astype(np.float32))
    warped = cv2.warpAffine(src, M, (W, H),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)

    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, np.int32(tri_dst), 255)
    return warped, mask


def build_morph(img_a, img_b, lm_a, lm_b, alpha: float = ALPHA) -> np.ndarray:
    """Return morph of img_a → img_b using Delaunay triangulation."""
    pts_a = lm_a * [W, H]
    pts_b = lm_b * [W, H]
    pts_m = (1 - alpha) * pts_a + alpha * pts_b

    subdiv = cv2.Subdiv2D((0, 0, W, H))
    for pt in pts_m:
        subdiv.insert(tuple(pt))
    triangles = subdiv.getTriangleList().reshape(-1, 3, 2)

    morph = np.zeros_like(img_a)
    for tri in triangles:
        idx = []
        for v in tri:                           # map triangle vertices to idx
            j = np.argmin(np.linalg.norm(pts_m - v, axis=1))
            if np.linalg.norm(pts_m[j] - v) < 1.0:
                idx.append(j)
        if len(idx) != 3:
            continue
        warp_a, mask_a = _warp_triangle(img_a, pts_a[idx], pts_m[idx])
        warp_b, mask_b = _warp_triangle(img_b, pts_b[idx], pts_m[idx])
        blend_mask = ((mask_a / 255.) * (1 - alpha) +
                      (mask_b / 255.) * alpha)[..., None]
        morph = (morph * (1 - blend_mask) +
                 (warp_a * (1 - alpha) + warp_b * alpha) * blend_mask
                 ).astype(np.uint8)
    return morph

# Cache builder
def build_caches(img_paths: List[str]):
    """Pre-compute landmarks, embeddings and grayscale vectors."""
    # Landmarks
    landmarks, skipped = {}, []
    for i, p in enumerate(img_paths, 1):
        try:
            img_rs = cv2.resize(cv2.imread(p), CANVAS_SIZE)
            landmarks[p] = extract_landmarks(img_rs)
        except RuntimeError:
            skipped.append(p)
        if i % PROGRESS_STEP == 0 or i == len(img_paths):
            logger.info("  Landmarks %d/%d", i, len(img_paths))

    usable = [p for p in img_paths if p not in skipped]
    logger.info("Landmarks OK for %d images  |  %d skipped",
                len(usable), len(skipped))

    # PyTorch FaceNet
    pytorch_model = init_models()
    pytorch_cache = {}
    logger.info("Caching PyTorch embeddings …")
    with ThreadPoolExecutor(N_THREADS) as pool:
        futures = {pool.submit(embed_pytorch, p, pytorch_model): p
                   for p in usable}
        for done, fut in enumerate(as_completed(futures), 1):
            pytorch_cache[futures[fut]] = fut.result()
            if done % PROGRESS_STEP == 0 or done == len(usable):
                logger.info("  PyTorch %d/%d", done, len(usable))

    # DeepFace FaceNet
    deepface_cache = {}
    logger.info("Caching DeepFace embeddings …")
    for i, p in enumerate(usable, 1):
        deepface_cache[p] = np.asarray(
            DeepFace.represent(img_path=p,
                               model_name="Facenet",
                               enforce_detection=False)[0]["embedding"],
            dtype=np.float32
        )
        if i % PROGRESS_STEP == 0 or i == len(usable):
            logger.info("  DeepFace %d/%d", i, len(usable))

    # Grayscale baseline
    gray_cache = {}
    logger.info("Caching grayscale vectors …")
    for i, p in enumerate(usable, 1):
        img_rs = cv2.resize(cv2.imread(p), CANVAS_SIZE)
        gray_cache[p] = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY).flatten().astype(np.float32)
        if i % PROGRESS_STEP == 0 or i == len(usable):
            logger.info("  Gray %d/%d", i, len(usable))

    return dict(
        usable=usable,
        landmarks=landmarks,
        pytorch_model=pytorch_model,
        pytorch_cache=pytorch_cache,
        deepface_cache=deepface_cache,
        gray_cache=gray_cache
    )

# Pair evaluation
def evaluate_pair(pair: Tuple[str, str], caches) -> Tuple[bool, bool, bool]:
    a, b = pair
    lm = caches["landmarks"]

    img_a = cv2.resize(cv2.imread(a), CANVAS_SIZE)
    img_b = cv2.resize(cv2.imread(b), CANVAS_SIZE)
    morph_img = build_morph(img_a, img_b, lm[a], lm[b])

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, morph_img)
        morph_path = tmp.name

    # PyTorch
    emb_morph = embed_pytorch(morph_path, caches["pytorch_model"])
    ok_pt = (
        cosine_similarity(caches["pytorch_cache"][a], emb_morph) >= THRESHOLDS["pytorch"] and
        cosine_similarity(caches["pytorch_cache"][b], emb_morph) >= THRESHOLDS["pytorch"]
    )

    # DeepFace (now wrapped in try/except)
    try:
        emb_morph = np.asarray(
            DeepFace.represent(img_path=morph_path,
                               model_name="Facenet",
                               enforce_detection=False)[0]["embedding"],
            dtype=np.float32
        )
        ok_df = (
            cosine_similarity(caches["deepface_cache"][a], emb_morph) >= THRESHOLDS["deepface"] and
            cosine_similarity(caches["deepface_cache"][b], emb_morph) >= THRESHOLDS["deepface"]
        )
    except Exception as e:
        logging.debug("DeepFace failed on morph (%s , %s): %s", a, b, e)
        ok_df = False

    # Grayscale baseline
    gray_morph = cv2.cvtColor(morph_img, cv2.COLOR_BGR2GRAY).flatten().astype(np.float32)
    ok_gray = (
        score_opencv_gray(caches["gray_cache"][a], gray_morph) >= THRESHOLDS["opencv_gray"] and
        score_opencv_gray(caches["gray_cache"][b], gray_morph) >= THRESHOLDS["opencv_gray"]
    )

    # cleanup
    os.unlink(morph_path)
    return ok_pt, ok_df, ok_gray

# Attack runner
def run_attack(pairs: List[Tuple[str, str]], caches) -> None:
    """Run morph attack evaluation on the given pairs."""
    results = {"pytorch": [], "deepface": [], "opencv_gray": []}
    pair_records = []

    for i, pair in enumerate(pairs, 1):
        ok_pt, ok_df, ok_gray = evaluate_pair(pair, caches)

        results["pytorch"].append(ok_pt)
        results["deepface"].append(ok_df)
        results["opencv_gray"].append(ok_gray)

        pair_records.append({
            "pair_id": i,
            "image_a": pair[0],
            "image_b": pair[1],
            "pytorch_success": ok_pt,
            "deepface_success": ok_df,
            "opencv_success": ok_gray
        })

    # Summary statistics
    stats = {}
    for method, successes in results.items():
        count = sum(successes)
        rate = count / len(pairs) * 100
        stats[method] = {"count": int(count), "rate": float(rate)}

    # Output results
    print("\nMorph-Attack Summary")
    print(f"PyTorch-FaceNet   : {stats['pytorch']['count']:3d}/{len(pairs):3d}   ({stats['pytorch']['rate']:4.1f}%)")
    print(f"DeepFace-FaceNet  : {stats['deepface']['count']:3d}/{len(pairs):3d}   ({stats['deepface']['rate']:4.1f}%)")
    print(f"OpenCV grayscale  : {stats['opencv_gray']['count']:3d}/{len(pairs):3d}   ({stats['opencv_gray']['rate']:4.1f}%)")
    elapsed = time.time() - start_time if 'start_time' in globals() else 0
    print(f"Elapsed: {elapsed:.1f} s")


    # Save detailed results
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pair-level CSV
    import pandas as pd
    df_pairs = pd.DataFrame(pair_records)
    pairs_csv = os.path.join(OUTPUT_DIR, "morph_attack_pairs.csv")
    df_pairs.to_csv(pairs_csv, index=False)
    logger.info(f"Pair-level CSV  -> {pairs_csv}")

    # Summary JSON
    summary_data = {
        "total_pairs": int(len(pairs)),
        "results": stats,
        "elapsed_seconds": float(elapsed),
        "thresholds": THRESHOLDS
    }
    summary_json = os.path.join(OUTPUT_DIR, "morph_attack_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Summary  JSON   -> {summary_json}")

if __name__ == "__main__":
    start_time = time.time()

    # Load data and generate pairs
    paths, labels = load_dataset(DATASET_DIR)
    label_map = defaultdict(list)
    for p, l in zip(paths, labels):
        label_map[l].append(p)

    # Filter to identities with at least 2 images
    valid_map = {k: v for k, v in label_map.items() if len(v) >= 2}
    genuine_pairs, _ = make_pairs(valid_map)

    # Sample pairs for evaluation
    if len(genuine_pairs) > MAX_PAIRS:
        attack_pairs = random.sample(genuine_pairs, MAX_PAIRS)
    else:
        attack_pairs = genuine_pairs

    logger.info(f"Running morph attack on {len(attack_pairs)} pairs …")

    # Get all unique image paths
    unique_paths = list(set([p for pair in attack_pairs for p in pair]))

    # Build caches
    caches = build_caches(unique_paths)

    # Filter attack pairs to only include usable images (ones with landmarks)
    usable_set = set(caches["usable"])
    filtered_pairs = [pair for pair in attack_pairs
                     if pair[0] in usable_set and pair[1] in usable_set]

    logger.info(f"Filtered {len(attack_pairs)} pairs to {len(filtered_pairs)} usable pairs")

    # Run attack
    run_attack(filtered_pairs, caches)
