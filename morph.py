#!/usr/bin/env python3
"""
morph.py — Demonstrates a face-morphing attack and evaluates its success
           with three matchers: PyTorch-FaceNet, DeepFace-FaceNet and
           a naive OpenCV-grayscale baseline.
"""

from __future__ import annotations
import os, random, time, json, csv, tempfile, pathlib, logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace

from main import (
    load_dataset, make_pairs, init_models, embed_pytorch, score_opencv_gray,
    logger, OUTPUT_DIR
)

# ───────────────────────────── configuration ────────────────────────────────
DATASET_DIR   = os.path.join("Data", "Face")
CANVAS_SIZE   = (512, 512)          # warp target size (W, H)
MAX_PAIRS     = 100                 # morphs to evaluate
N_THREADS     = 4                   # for embedding / evaluation
ALPHA         = 0.5                 # blend factor for morphing
PROGRESS_STEP = 100                 # log every N items

THRESHOLDS: Dict[str, float] = {
    "pytorch"     : 0.70,   # cosine ≥ threshold → accept
    "deepface"    : 0.70,
    "opencv_gray" : -0.50   # distance ≥ –0.50 → accept (note sign)
}

# ───────────────────────────── landmark extractor ───────────────────────────
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

# ───────────────────────────── triangle-warp helpers ────────────────────────
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

# ───────────────────────────── cache builder ────────────────────────────────
def build_caches(img_paths: List[str]):
    """Pre-compute landmarks, embeddings and grayscale vectors."""
    # ---- landmarks
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

    # ---- PyTorch FaceNet
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

    # ---- DeepFace FaceNet
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

    # ---- grayscale baseline
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

# ───────────────────────────── pair evaluation ──────────────────────────────
def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Return classic cosine similarity."""
    return float(u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def evaluate_pair(pair: Tuple[str, str], caches) -> Tuple[bool, bool, bool]:
    a, b = pair
    lm = caches["landmarks"]

    img_a = cv2.resize(cv2.imread(a), CANVAS_SIZE)
    img_b = cv2.resize(cv2.imread(b), CANVAS_SIZE)
    morph_img = build_morph(img_a, img_b, lm[a], lm[b])

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, morph_img)
        morph_path = tmp.name

    # ---- PyTorch
    emb_morph = embed_pytorch(morph_path, caches["pytorch_model"])
    ok_pt = (
        cosine(caches["pytorch_cache"][a], emb_morph) >= THRESHOLDS["pytorch"] and
        cosine(caches["pytorch_cache"][b], emb_morph) >= THRESHOLDS["pytorch"]
    )

    # ── DeepFace  (now wrapped in try/except)
    try:
        emb_morph = np.asarray(
            DeepFace.represent(img_path=morph_path,
                               model_name="Facenet",
                               enforce_detection=False)[0]["embedding"],
            dtype=np.float32
        )
        ok_df = (
            cosine(caches["deepface_cache"][a], emb_morph) >= THRESHOLDS["deepface"] and
            cosine(caches["deepface_cache"][b], emb_morph) >= THRESHOLDS["deepface"]
        )
    except Exception as e:
        logging.debug("DeepFace failed on morph (%s , %s): %s", a, b, e)
        ok_df = False

    # ---- grayscale baseline
    gray_morph = cv2.cvtColor(morph_img, cv2.COLOR_BGR2GRAY).flatten().astype(np.float32)
    ok_gray = (
        score_opencv_gray(caches["gray_cache"][a], gray_morph) >= THRESHOLDS["opencv_gray"] and
        score_opencv_gray(caches["gray_cache"][b], gray_morph) >= THRESHOLDS["opencv_gray"]
    )

    os.remove(morph_path)
    return ok_pt, ok_df, ok_gray

# ───────────────────────────── attack driver ────────────────────────────────
def run_attack(pairs: List[Tuple[str, str]], caches) -> None:
    pathlib.Path(OUTPUT_DIR).mkdir(exist_ok=True)

    tallies = Counter(pytorch=0, deepface=0, opencv_gray=0)
    records = []
    t0 = time.time()

    with ThreadPoolExecutor(N_THREADS) as pool:
        for (ok_pt, ok_df, ok_gray), (a, b) in zip(
                pool.map(lambda pr: evaluate_pair(pr, caches), pairs), pairs):
            tallies["pytorch"]     += int(ok_pt)
            tallies["deepface"]    += int(ok_df)
            tallies["opencv_gray"] += int(ok_gray)
            records.append({
                "img_A": a, "img_B": b,
                "pass_pytorch": int(ok_pt),
                "pass_deepface": int(ok_df),
                "pass_gray":    int(ok_gray)
            })

    total = len(pairs)

    # -------- console summary --------
    print("\n────────  Morph-Attack Summary  ────────")
    for k, lbl in [("pytorch", "PyTorch-FaceNet"),
                   ("deepface", "DeepFace-FaceNet"),
                   ("opencv_gray", "OpenCV grayscale")]:
        print(f"{lbl:<18}: {tallies[k]:>3}/{total}   ({tallies[k] / total:.1%})")
    print(f"Elapsed: {time.time() - t0:.1f} s")
    print("────────────────────────────────────────\n")

    # -------- persist summary & per-pair log --------
    csv_path = os.path.join(OUTPUT_DIR, "morph_attack_pairs.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    def py(v):
        if isinstance(v, (np.integer, np.bool_)):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v

    json_path = os.path.join(OUTPUT_DIR, "morph_attack_summary.json")
    with open(json_path, "w") as fh:
        json.dump({
            "total_pairs": int(total),
            "success_counts": {k: py(v) for k, v in tallies.items()},
            "success_rates": {k: tallies[k] / total for k in tallies}
        }, fh, indent=2)

    logger.info("Pair-level CSV  -> %s", csv_path)
    logger.info("Summary  JSON   -> %s", json_path)

# ──────────────────────────────── entry-point ───────────────────────────────
if __name__ == "__main__":
    img_paths, labels = load_dataset(DATASET_DIR)

    by_id = defaultdict(list)
    for p, lab in zip(img_paths, labels):
        by_id[lab].append(p)
    genuine_pairs, _ = make_pairs({k: v for k, v in by_id.items() if len(v) >= 2})

    caches      = build_caches(img_paths)
    usable_set  = set(caches["usable"])
    valid_pairs = [(a, b) for a, b in genuine_pairs if a in usable_set and b in usable_set]
    random.shuffle(valid_pairs)
    test_pairs  = valid_pairs[:MAX_PAIRS]

    logger.info("Running morph attack on %d pairs …", len(test_pairs))
    run_attack(test_pairs, caches)
