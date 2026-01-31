import os
import glob
import random
import numpy as np
import cv2
import torch
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from typing import List, Tuple, Dict, Any

from logger import logger

# Configuration constants
DATASET_DIR = os.path.join('Data')
OUTPUT_DIR = 'results'
FMR_TARGETS = [0.001, 0.01, 0.05]
IMAGE_EXTS = ('.bmp', '.jpg', '.jpeg', '.png')  # Image extensions to process
LOG_EVERY = 100  # Log progress every N items
NUM_THREADS = 4  # Parallel threads for embedding
GRAY_IMAGE_SIZE = (512, 512)  # Standard size for grayscale image processing


def load_dataset(folder: str) -> Tuple[List[str], List[str]]:
    """Load dataset paths and labels from a folder structure.

    Args:
        folder: Root folder containing person subdirectories

    Returns:
        Tuple of (image_paths, labels)
    """
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

def make_pairs(label_map: Dict[str, List[str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Generate genuine and impostor pairs from label mapping.

    Args:
        label_map: Dictionary mapping person IDs to list of image paths

    Returns:
        Tuple of (genuine_pairs, impostor_pairs)
    """
    logger.info("Generating genuine/impostor pairs...")
    genuine, impostor = [], []

    # Generate genuine pairs (same person)
    for imgs in label_map.values():
        for i in range(len(imgs)):
            for j in range(i+1, len(imgs)):
                genuine.append((imgs[i], imgs[j]))

    # Generate impostor pairs (different persons)
    keys = list(label_map.keys())
    while len(impostor) < len(genuine):
        a, b = random.sample(keys, 2)
        impostor.append((random.choice(label_map[a]), random.choice(label_map[b])))

    logger.info(f"{len(genuine)} genuine, {len(impostor)} impostor pairs.")
    return genuine, impostor

def init_models():
    """Initialize face recognition models.

    Returns:
        PyTorch FaceNet model
    """
    logger.info("Initializing models...")
    pt_fn = InceptionResnetV1(pretrained='vggface2').eval()
    logger.info("PyTorch FaceNet ready.")
    return pt_fn

def embed_pytorch(path: str, model) -> np.ndarray:
    """Extract face embedding using PyTorch FaceNet.

    Args:
        path: Path to image file
        model: PyTorch FaceNet model

    Returns:
        Face embedding as numpy array
    """
    img = cv2.imread(path)
    r = cv2.resize(img, (160, 160))
    rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
    t = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).float()
    t = fixed_image_standardization(t)

    with torch.no_grad():
        return model(t).squeeze().numpy()

def score_opencv_gray(g1: np.ndarray, g2: np.ndarray) -> float:
    """Calculate similarity score using grayscale image comparison.

    Args:
        g1: First grayscale image vector
        g2: Second grayscale image vector

    Returns:
        Similarity score (negative L2 distance)
    """
    return -np.linalg.norm(g1 - g2)

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        u: First vector
        v: Second vector

    Returns:
        Cosine similarity score
    """
    return float(u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v)))

def cache_grayscale_images(paths: List[str], log_every: int = LOG_EVERY,
                           image_size: tuple = GRAY_IMAGE_SIZE) -> Dict[str, np.ndarray]:
    """Cache grayscale versions of images.

    Args:
        paths: List of image paths
        log_every: Log progress every N images
        image_size: Target size (width, height) for resizing images

    Returns:
        Dictionary mapping paths to flattened grayscale arrays
    """
    logger.info("Caching grayscale images…")
    gray_cache = {}

    for i, p in enumerate(paths, 1):
        img = cv2.imread(p)
        img_resized = cv2.resize(img, image_size)
        gray_cache[p] = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY).flatten().astype(np.float32)

        if i % log_every == 0 or i == len(paths):
            logger.info(f"{i}/{len(paths)} grays cached")

    return gray_cache

def cache_pytorch_embeddings(paths: List[str], model, num_threads: int = NUM_THREADS,
                           log_every: int = LOG_EVERY) -> Dict[str, np.ndarray]:
    """Cache PyTorch face embeddings in parallel.

    Args:
        paths: List of image paths
        model: PyTorch FaceNet model
        num_threads: Number of parallel threads
        log_every: Log progress every N images

    Returns:
        Dictionary mapping paths to face embeddings
    """
    logger.info("Caching PyTorch embeddings…")
    pt_cache = {}

    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futures = {ex.submit(embed_pytorch, p, model): p for p in paths}

        for i, fut in enumerate(as_completed(futures), 1):
            pt_cache[futures[fut]] = fut.result()

            if i % log_every == 0 or i == len(paths):
                logger.info(f"PT {i}/{len(paths)} embeddings cached")

    return pt_cache