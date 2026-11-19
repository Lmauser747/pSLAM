#!/usr/bin/env python3
"""Offline binary Bag-of-Words vocabulary generator for PoolSLAM.

Generates a 4-level, 10-branch (10,000-word) binary vocabulary tree compatible
with PoolSLAM::descriptor::BinaryBowTree::readVocabFile without using OpenCV.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Tree constants matching PoolSLAM::descriptor::BinaryBowTree (BinaryBow.hpp)
NUM_BRANCHES = 10
TREE_DEPTH = 4
TREE_SIZE = NUM_BRANCHES * (NUM_BRANCHES ** (TREE_DEPTH - 1) - 1) // (NUM_BRANCHES - 1)  # 1,110
VOCAB_SIZE = NUM_BRANCHES**TREE_DEPTH  # 10,000
TOTAL_NODES = TREE_SIZE + VOCAB_SIZE  # 11,110
EXPECTED_FILE_BYTES = TREE_SIZE * 32 + VOCAB_SIZE * (32 + 4)  # 395,520

# QOV descriptor constants matching PoolSLAM::descriptor::Qov.hpp
QOV_BITS = 256
ROTATIONS = 32
ANGLE_STEP_DEG = 360.0 / ROTATIONS
ANGLE_STEP_RAD = 2.0 * np.pi / ROTATIONS
CENTROID_RADIUS = 15

# 16-pixel Bresenham circle (radius 3) for FAST-9 detection
FAST_CIRCLE = [
    (0, -3), (1, -3), (2, -2), (3, -1),
    (3, 0), (3, 1), (2, 2), (1, 3),
    (0, 3), (-1, 3), (-2, 2), (-3, 1),
    (-3, 0), (-3, -1), (-2, -2), (-1, -3),
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


def c_lround(x: np.ndarray) -> np.ndarray:
    """Round half away from zero to match C++ std::lround."""
    return np.where(x >= 0.0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int32)


def find_default_offsets_header() -> Path | None:
    """Attempt to locate QovBaseOffsets.hpp relative to this script."""
    candidate = Path(__file__).resolve().parents[2] / "src" / "descriptor" / "QovBaseOffsets.hpp"
    return candidate if candidate.is_file() else None


def load_qov_base_offsets(header_path: Path | None) -> np.ndarray:
    """Parse QOV_BASE_OFFSETS (1024 ints -> 256x4) from QovBaseOffsets.hpp."""
    if header_path is None:
        header_path = find_default_offsets_header()

    if header_path is None or not header_path.is_file():
        raise FileNotFoundError(
            "Could not locate QovBaseOffsets.hpp. Please specify --offsets-header pointing to "
            "src/descriptor/QovBaseOffsets.hpp so vocabulary bits match runtime Qov descriptors."
        )

    print(f"Loading QOV base offsets from: {header_path}")
    text = header_path.read_text(encoding="utf-8")
    # Strip C/C++ comments
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    match = re.search(r"QOV_BASE_OFFSETS[^=]*=\s*\{(.*?)\}\s*;", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find QOV_BASE_OFFSETS initializer in {header_path}")

    values = [int(tok) for tok in re.findall(r"-?\d+", match.group(1))]
    if len(values) != QOV_BITS * 4:
        raise ValueError(
            f"Expected {QOV_BITS * 4} integers in QOV_BASE_OFFSETS, found {len(values)}"
        )
    return np.array(values, dtype=np.int32).reshape(QOV_BITS, 4)


class QovExtractor:
    """Pure NumPy/SciPy implementation of FAST-9 + Shi-Tomasi bucketing + QOV descriptors."""

    def __init__(
        self,
        base_offsets: np.ndarray,
        fast_threshold: int = 20,
        bucket_size: int = 32,
        per_bucket: int = 2,
    ) -> None:
        self.fast_threshold = fast_threshold
        self.bucket_size = bucket_size
        self.per_bucket = per_bucket

        # Precompute 32 rotated offset lookup tables matching RotatedQovPairs
        x1 = base_offsets[:, 0].astype(np.float64)
        y1 = base_offsets[:, 1].astype(np.float64)
        x2 = base_offsets[:, 2].astype(np.float64)
        y2 = base_offsets[:, 3].astype(np.float64)

        self.rot_dx1 = np.empty((ROTATIONS, QOV_BITS), dtype=np.int32)
        self.rot_dy1 = np.empty((ROTATIONS, QOV_BITS), dtype=np.int32)
        self.rot_dx2 = np.empty((ROTATIONS, QOV_BITS), dtype=np.int32)
        self.rot_dy2 = np.empty((ROTATIONS, QOV_BITS), dtype=np.int32)

        for idx in range(ROTATIONS):
            rad = np.deg2rad(idx * ANGLE_STEP_DEG)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)
            self.rot_dx1[idx] = c_lround(x1 * cos_a - y1 * sin_a)
            self.rot_dy1[idx] = c_lround(x1 * sin_a + y1 * cos_a)
            self.rot_dx2[idx] = c_lround(x2 * cos_a - y2 * sin_a)
            self.rot_dy2[idx] = c_lround(x2 * sin_a + y2 * cos_a)

        max_rot = int(
            max(
                np.max(np.abs(self.rot_dx1)),
                np.max(np.abs(self.rot_dy1)),
                np.max(np.abs(self.rot_dx2)),
                np.max(np.abs(self.rot_dy2)),
            )
        )
        self.margin = max(CENTROID_RADIUS, max_rot, 3) + 1

        # Precompute square patch coordinates ([-15, 15] x [-15, 15]) matching LfTracker::calculateDescriptors
        grid = np.arange(-CENTROID_RADIUS, CENTROID_RADIUS + 1, dtype=np.int32)
        gx, gy = np.meshgrid(grid, grid)
        self.cx = gx.ravel()
        self.cy = gy.ravel()
        self.cx_f = self.cx.astype(np.float64)
        self.cy_f = self.cy.astype(np.float64)

    def detect_keypoints(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run vectorized FAST-9 + Shi-Tomasi scoring + 3x3 NMS + grid bucketing."""
        h, w = gray.shape
        m = self.margin
        if h <= 2 * m or w <= 2 * m:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        img_i16 = gray.astype(np.int16)
        center = img_i16[m : h - m, m : w - m]
        ring = np.stack(
            [img_i16[m + dy : h - m + dy, m + dx : w - m + dx] for dx, dy in FAST_CIRCLE],
            axis=0,
        )

        # Check for 9 contiguous brighter or darker pixels around the 16-pixel ring
        thresh = self.fast_threshold
        brighter = (ring > (center + thresh)).astype(np.uint8)
        darker = (ring < (center - thresh)).astype(np.uint8)

        def has_9_contiguous(flags: np.ndarray) -> np.ndarray:
            wrapped = np.concatenate([flags, flags[:8]], axis=0)
            cs = np.cumsum(wrapped, axis=0)
            zeros = np.zeros((1, wrapped.shape[1], wrapped.shape[2]), dtype=cs.dtype)
            cs = np.concatenate([zeros, cs], axis=0)
            return np.any((cs[9:] - cs[:-9]) == 9, axis=0)

        is_fast = has_9_contiguous(brighter) | has_9_contiguous(darker)
        if not np.any(is_fast):
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        # Shi-Tomasi minimum eigenvalue (lambda_min) across a 5x5 window
        img_f = gray.astype(np.float32)
        ix = ndimage.sobel(img_f, axis=1) * 0.125
        iy = ndimage.sobel(img_f, axis=0) * 0.125
        ixx = ndimage.uniform_filter(ix * ix, size=5)[m : h - m, m : w - m]
        iyy = ndimage.uniform_filter(iy * iy, size=5)[m : h - m, m : w - m]
        ixy = ndimage.uniform_filter(ix * iy, size=5)[m : h - m, m : w - m]

        trace = ixx + iyy
        diff = ixx - iyy
        disc = np.sqrt(np.maximum(0.0, diff * diff + 4.0 * ixy * ixy))
        lambda_min = 0.5 * (trace - disc)

        score = np.where(is_fast, lambda_min, 0.0)
        local_max = ndimage.maximum_filter(score, size=3)
        peaks = (score > 0.0) & (score == local_max)

        ys, xs = np.nonzero(peaks)
        if ys.size == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        peak_scores = score[ys, xs]

        # Spatial grid bucketing (keep top `per_bucket` Shi-Tomasi corners per cell)
        order = np.argsort(peak_scores)[::-1]
        ys = ys[order]
        xs = xs[order]

        b_rows = ys // self.bucket_size
        b_cols = xs // self.bucket_size
        n_cols = ((w - 2 * m) + self.bucket_size - 1) // self.bucket_size
        bucket_ids = b_rows * n_cols + b_cols

        counts: dict[int, int] = {}
        keep_idx = []
        for idx, bid in enumerate(bucket_ids.tolist()):
            c = counts.get(bid, 0)
            if c < self.per_bucket:
                counts[bid] = c + 1
                keep_idx.append(idx)

        keep = np.asarray(keep_idx, dtype=np.int64)
        return (xs[keep] + m).astype(np.int32), (ys[keep] + m).astype(np.int32)

    @staticmethod
    def blur_image(gray: np.ndarray) -> np.ndarray:
        """Apply 5x5 separable binomial filter [1, 4, 6, 4, 1] / 16 matching LfTracker::gaussianBlur."""
        k = np.array([1, 4, 6, 4, 1], dtype=np.int32)
        tmp = (ndimage.correlate1d(gray.astype(np.int32), k, axis=1, mode="nearest") >> 4).astype(np.int32)
        out = (ndimage.correlate1d(tmp, k, axis=0, mode="nearest") >> 4).astype(np.uint8)
        return out

    def extract(self, gray: np.ndarray) -> np.ndarray:
        """Extract (K, 256) uint8 {0, 1} QOV descriptors from a grayscale image."""
        xs, ys = self.detect_keypoints(gray)
        if xs.size == 0:
            return np.empty((0, QOV_BITS), dtype=np.uint8)

        # Smooth image matching LfTracker::gaussianBlur prior to descriptor calculation
        blurred = self.blur_image(gray)

        # Intensity centroid orientation over 31x31 square patch matching LfTracker
        patch = blurred[ys[:, None] + self.cy[None, :], xs[:, None] + self.cx[None, :]].astype(
            np.float64
        )
        m10 = patch @ self.cx_f
        m01 = patch @ self.cy_f
        angles = np.arctan2(m01, m10)

        norm_angles = np.mod(angles, 2.0 * np.pi)
        bins = c_lround(norm_angles / ANGLE_STEP_RAD) % ROTATIONS

        # Evaluate 256 steered binary intensity comparisons: I(pt + p1) < I(pt + p2) on blurred image
        p1 = blurred[ys[:, None] + self.rot_dy1[bins], xs[:, None] + self.rot_dx1[bins]]
        p2 = blurred[ys[:, None] + self.rot_dy2[bins], xs[:, None] + self.rot_dx2[bins]]
        return (p1 < p2).astype(np.uint8)


def hamming_assign(
    x_bits: np.ndarray,
    centroids: np.ndarray,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Compute exact nearest centroid index under Hamming distance via BLAS GEMM.

    For binary vectors x, c in {0, 1}^256:
        Hamming(x, c) = sum(x) + sum(c) - 2 * (x . c)
    """
    n = x_bits.shape[0]
    c_f = centroids.astype(np.float32)
    c_sum = c_f.sum(axis=1)  # (k,)
    labels = np.empty(n, dtype=np.int32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x_f = x_bits[start:end].astype(np.float32)
        x_sum = x_f.sum(axis=1, keepdims=True)
        dists = x_sum + c_sum[None, :] - 2.0 * (x_f @ c_f.T)
        labels[start:end] = np.argmin(dists, axis=1)
    return labels


def kmeans_plus_plus_binary(
    x_bits: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pure NumPy k-means++ initialization directly using Hamming distance."""
    n = x_bits.shape[0]
    centroids = np.empty((k, x_bits.shape[1]), dtype=np.uint8)
    first_idx = int(rng.integers(0, n))
    centroids[0] = x_bits[first_idx]

    min_dists = np.count_nonzero(x_bits != centroids[0], axis=1).astype(np.float64)
    for i in range(1, k):
        probs = min_dists * min_dists
        total = probs.sum()
        if total <= 0.0:
            centroids[i:] = centroids[0]
            break
        probs /= total
        chosen_idx = int(rng.choice(n, p=probs))
        centroids[i] = x_bits[chosen_idx]
        new_dists = np.count_nonzero(x_bits != centroids[i], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
    return centroids


def cluster_binary_node(
    x_bits: np.ndarray,
    k: int,
    parent_bits: np.ndarray,
    rng: np.random.Generator,
    refine_iters: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition binary descriptors into `k` clusters with binary centroids."""
    n = x_bits.shape[0]
    if n == 0:
        return np.repeat(parent_bits[None, :], k, axis=0), np.empty(0, dtype=np.int32)

    # Handle small or low-uniqueness subsets cleanly so all `k` branches exist
    if n <= k:
        centroids = np.empty((k, QOV_BITS), dtype=np.uint8)
        centroids[:n] = x_bits
        centroids[n:] = parent_bits
        labels = np.arange(n, dtype=np.int32)
        return centroids, labels

    # Guard against fewer than `k` distinct descriptors in small nodes
    if n < 200:
        unique_rows = np.unique(x_bits, axis=0)
        if unique_rows.shape[0] < k:
            centroids = np.empty((k, QOV_BITS), dtype=np.uint8)
            u = unique_rows.shape[0]
            centroids[:u] = unique_rows
            centroids[u:] = parent_bits
            labels = hamming_assign(x_bits, centroids)
            return centroids, labels

    seed = int(rng.integers(0, 2**31 - 1))
    if HAS_SKLEARN:
        x_f = x_bits.astype(np.float32)
        # Seed with scikit-learn k-means++ (MiniBatchKMeans for large upper-level nodes)
        if n > 20_000:
            model = MiniBatchKMeans(
                n_clusters=k,
                init="k-means++",
                batch_size=4096,
                max_iter=50,
                n_init=3,
                random_state=seed,
            )
        else:
            model = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init="auto",
                max_iter=30,
                random_state=seed,
            )
        model.fit(x_f)
        centroids = (model.cluster_centers_ >= 0.5).astype(np.uint8)
    else:
        centroids = kmeans_plus_plus_binary(x_bits, k, rng)

    # Refine centroids using exact Hamming distance + bitwise majority vote
    labels = hamming_assign(x_bits, centroids)
    for _ in range(refine_iters):
        new_centroids = centroids.copy()
        for c_idx in range(k):
            mask = labels == c_idx
            count = int(np.count_nonzero(mask))
            if count > 0:
                new_centroids[c_idx] = (x_bits[mask].sum(axis=0) * 2 >= count).astype(np.uint8)
            else:
                # Reseed empty cluster with the sample furthest from its assigned centroid
                assigned = centroids[labels] if len(labels) > 0 else None
                if assigned is not None:
                    cur_dists = np.count_nonzero(x_bits != assigned, axis=1)
                    if np.max(cur_dists) > 0:
                        furthest_idx = int(np.argmax(cur_dists))
                        new_centroids[c_idx] = x_bits[furthest_idx]
                    else:
                        new_centroids[c_idx] = parent_bits
                else:
                    new_centroids[c_idx] = parent_bits
        new_labels = hamming_assign(x_bits, new_centroids)
        centroids = new_centroids
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

    # Guard against any empty clusters remaining after refinement
    for c_idx in range(k):
        if not np.any(labels == c_idx):
            assigned = centroids[labels] if len(labels) > 0 else None
            if assigned is not None:
                cur_dists = np.count_nonzero(x_bits != assigned, axis=1)
                if np.max(cur_dists) > 0:
                    furthest_idx = int(np.argmax(cur_dists))
                    centroids[c_idx] = x_bits[furthest_idx]
                    labels = hamming_assign(x_bits, centroids)

    return centroids, labels


def build_bow_tree(
    all_descriptors: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Build the 4-level, 10-branch binary vocabulary tree in breadth-first order."""
    rng = np.random.default_rng(seed)
    tree_bits = np.zeros((TOTAL_NODES, QOV_BITS), dtype=np.uint8)

    # Level 0 (root -> nodes 0..9)
    root_default = (all_descriptors.sum(axis=0) * 2 >= len(all_descriptors)).astype(np.uint8)
    print(f"[Level 1/{TREE_DEPTH}] Clustering {len(all_descriptors):,} descriptors into {NUM_BRANCHES} branches...")
    centroids, labels = cluster_binary_node(all_descriptors, NUM_BRANCHES, root_default, rng)
    tree_bits[0:NUM_BRANCHES] = centroids

    current_level_nodes: list[tuple[int, np.ndarray]] = []
    all_idx = np.arange(len(all_descriptors), dtype=np.int64)
    for b in range(NUM_BRANCHES):
        current_level_nodes.append((b, all_idx[labels == b]))

    # Levels 1..3 (nodes 10..109, 110..1109, 1110..11109)
    for level in range(1, TREE_DEPTH):
        next_level_nodes: list[tuple[int, np.ndarray]] = []
        total_parents = len(current_level_nodes)
        print(
            f"[Level {level + 1}/{TREE_DEPTH}] Expanding {total_parents:,} parent nodes -> "
            f"{total_parents * NUM_BRANCHES:,} children..."
        )
        for p_i, (parent_idx, member_indices) in enumerate(current_level_nodes):
            first_child = NUM_BRANCHES * parent_idx + NUM_BRANCHES
            parent_bits = tree_bits[parent_idx]
            subset = all_descriptors[member_indices]

            child_centroids, child_labels = cluster_binary_node(
                subset, NUM_BRANCHES, parent_bits, rng
            )
            tree_bits[first_child : first_child + NUM_BRANCHES] = child_centroids

            if level < TREE_DEPTH - 1:
                for b in range(NUM_BRANCHES):
                    next_level_nodes.append((first_child + b, member_indices[child_labels == b]))

            if total_parents >= 100 and (p_i + 1) % max(1, total_parents // 10) == 0:
                print(f"  Progress: {p_i + 1}/{total_parents} parent nodes clustered")

        current_level_nodes = next_level_nodes

    return tree_bits


def route_descriptors(
    tree_bits: np.ndarray,
    descriptors: np.ndarray,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Route descriptors down the tree identically to BinaryBowTree::associatedWord."""
    n = descriptors.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int32)

    if n > chunk_size:
        words = np.empty(n, dtype=np.int32)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            words[start:end] = route_descriptors(tree_bits, descriptors[start:end], chunk_size)
        return words

    # Level 1 (indices 0..9)
    best = hamming_assign(descriptors, tree_bits[0:NUM_BRANCHES])
    branch_offsets = np.arange(NUM_BRANCHES, dtype=np.int32)

    # Levels 2..4
    for _ in range(1, TREE_DEPTH):
        first_children = NUM_BRANCHES * best + NUM_BRANCHES
        cand_indices = first_children[:, None] + branch_offsets[None, :]  # (K, 10)
        cand_bits = tree_bits[cand_indices]  # (K, 10, 256)
        dists = np.count_nonzero(cand_bits != descriptors[:, None, :], axis=2)
        best = first_children + np.argmin(dists, axis=1).astype(np.int32)

    return best - TREE_SIZE


def compute_idf_weights(
    tree_bits: np.ndarray,
    all_descriptors: np.ndarray,
    image_slices: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute IDF(w) = ln(N_images / doc_freq[w]) for all 10,000 leaf words."""
    doc_freq = np.zeros(VOCAB_SIZE, dtype=np.int32)
    num_images = len(image_slices)

    for start, end in image_slices:
        if end <= start:
            continue
        word_ids = route_descriptors(tree_bits, all_descriptors[start:end])
        doc_freq[np.unique(word_ids)] += 1

    idf = np.zeros(VOCAB_SIZE, dtype=np.float32)
    seen = doc_freq > 0
    idf[seen] = np.log(num_images / doc_freq[seen]).astype(np.float32)
    return idf, doc_freq


def write_vocab_binary(output_path: Path, tree_bits: np.ndarray, idf: np.ndarray) -> None:
    """Serialize tree centroids and IDF weights to match BinaryBowTree::readVocabFile."""
    if tree_bits.shape != (TOTAL_NODES, QOV_BITS):
        raise ValueError(f"Expected tree_bits shape {(TOTAL_NODES, QOV_BITS)}, got {tree_bits.shape}")
    if idf.shape != (VOCAB_SIZE,):
        raise ValueError(f"Expected idf shape {(VOCAB_SIZE,)}, got {idf.shape}")

    # Little-endian bit packing matches std::array<uint64_t, 4> layout in QovDescriptor
    packed = np.packbits(tree_bits, axis=1, bitorder="little")  # (11110, 32) uint8
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        # 1. Internal branch nodes (levels 1..3): 1,110 x 32 bytes
        f.write(packed[:TREE_SIZE].tobytes())
        # 2. Leaf words (level 4): 10,000 x (32 bytes bits + 4 bytes float32 idf)
        for word_id in range(VOCAB_SIZE):
            f.write(packed[TREE_SIZE + word_id].tobytes())
            f.write(struct.pack("<f", float(idf[word_id])))

    actual_bytes = output_path.stat().st_size
    if actual_bytes != EXPECTED_FILE_BYTES:
        raise RuntimeError(
            f"Binary size mismatch: wrote {actual_bytes} bytes, expected {EXPECTED_FILE_BYTES}"
        )


def collect_image_paths(images_dir: Path, max_images: int | None) -> list[Path]:
    """Recursively find supported image files in sorted order."""
    paths = sorted(
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_images is not None:
        paths = paths[:max_images]
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 10,000-word binary DBoW vocabulary for PoolSLAM."
    )
    parser.add_argument(
        "images_dir",
        type=Path,
        help="Path to folder containing training images (searched recursively).",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Destination path for the binary vocabulary file (e.g., pool_vocab_10k.bin).",
    )
    parser.add_argument(
        "--offsets-header",
        type=Path,
        default=None,
        help="Path to C++ QovBaseOffsets.hpp (defaults to src/descriptor/QovBaseOffsets.hpp).",
    )
    parser.add_argument(
        "--fast-threshold",
        type=int,
        default=20,
        help="FAST-9 intensity threshold (default: 20).",
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=32,
        help="Spatial grid bucket size in pixels (default: 32).",
    )
    parser.add_argument(
        "--per-bucket",
        type=int,
        default=2,
        help="Max Shi-Tomasi keypoints retained per grid bucket (default: 2).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on number of training images to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic clustering (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {args.images_dir}")

    image_paths = collect_image_paths(args.images_dir, args.max_images)
    if not image_paths:
        raise RuntimeError(f"No supported images found in {args.images_dir}")

    base_offsets = load_qov_base_offsets(args.offsets_header)
    extractor = QovExtractor(
        base_offsets=base_offsets,
        fast_threshold=args.fast_threshold,
        bucket_size=args.bucket_size,
        per_bucket=args.per_bucket,
    )

    print(f"Extracting QOV descriptors from {len(image_paths):,} images...")
    descriptor_blocks: list[np.ndarray] = []
    image_slices: list[tuple[int, int]] = []
    offset = 0

    for i, img_path in enumerate(image_paths):
        with Image.open(img_path) as im:
            gray = np.asarray(im.convert("L"), dtype=np.uint8)
        descs = extractor.extract(gray)
        count = descs.shape[0]
        if count > 0:
            descriptor_blocks.append(descs)
        image_slices.append((offset, offset + count))
        offset += count

        if (i + 1) % max(1, len(image_paths) // 10) == 0 or (i + 1) == len(image_paths):
            print(f"  Processed {i + 1}/{len(image_paths)} images ({offset:,} descriptors)")

    if offset == 0:
        raise RuntimeError("Zero descriptors extracted. Try lowering --fast-threshold.")

    all_descriptors = np.vstack(descriptor_blocks)
    del descriptor_blocks

    if all_descriptors.shape[0] < VOCAB_SIZE * 5:
        print(
            f"[WARN] Only {all_descriptors.shape[0]:,} descriptors extracted for {VOCAB_SIZE:,} words. "
            "For well-populated leaves, aim for >= 200,000 descriptors (e.g. 500–2,000+ diverse frames).",
            file=sys.stderr,
        )

    tree_bits = build_bow_tree(all_descriptors, seed=args.seed)

    print("Computing TF-IDF document frequencies via top-down tree routing...")
    idf, doc_freq = compute_idf_weights(tree_bits, all_descriptors, image_slices)

    write_vocab_binary(args.output_file, tree_bits, idf)

    populated = int(np.count_nonzero(doc_freq > 0))
    nonzero_idf = idf[doc_freq > 0]
    print(f"\nVocabulary written to: {args.output_file}")
    print(f"  File size       : {EXPECTED_FILE_BYTES:,} bytes (verified)")
    print(f"  Populated words : {populated:,} / {VOCAB_SIZE:,}")
    if nonzero_idf.size > 0:
        print(
            f"  IDF range       : min={nonzero_idf.min():.3f}, "
            f"median={np.median(nonzero_idf):.3f}, max={nonzero_idf.max():.3f}"
        )


if __name__ == "__main__":
    main()