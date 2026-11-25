#!/usr/bin/env python3
"""
Pool-SLAM Accuracy Benchmark Harness
====================================
Evaluates Pool-SLAM trajectory estimation accuracy against a ground-truth dataset
(such as the EuRoC MAV V1_01_easy sequence).

Calibration & Camera Matrix (K):
--------------------------------
The camera intrinsic calibration matrix K:
      [ fx   0  cx ]
  K = [  0  fy  cy ]
      [  0   0   1 ]
is resolved in the following priority order:
  1. User-specified configuration: If --config <path.yaml> is passed, intrinsics
     and distortion are read directly from that YAML file.
  2. Dataset sensor.yaml: If <dataset>/cam0/sensor.yaml exists, the script automatically
     extracts:
       - 'resolution': [width, height] -> (e.g. 752 x 480)
       - 'intrinsics': [fu, fv, cu, cv] -> fx, fy, cx, cy
       - 'distortion_coefficients': [k1, k2, p1, p2] -> radtan distortion
       - 'T_BS': 4x4 matrix -> Camera-to-body extrinsics (T_BC)
  3. Default EuRoC calibration: If sensor.yaml is absent, canonical MT9V034
     calibration is used:
       fx=458.654, fy=457.296, cx=367.215, cy=248.375,
       distortion=[-0.283408, 0.073959, 0.000194, 0.000018].

CLI Documentation & Help:
-------------------------
Run with -h or --help to display full usage details:
  python3 benchmarks/accuracy/benchmark_accuracy.py -h

Usage Examples:
  # 1. Run on extracted EuRoC dataset for the first 60 seconds:
  python3 benchmarks/accuracy/benchmark_accuracy.py --dataset /path/to/V1_01_easy --time-cutoff 60.0

  # 2. Run with custom evaluation interval (every 3 seconds) and depth noise (0.01m):
  python3 benchmarks/accuracy/benchmark_accuracy.py \\
      --dataset /path/to/V1_01_easy \\
      --eval-interval 3.0 \\
      --depth-noise-sigma 0.01

  # 3. Use custom SystemConfig YAML:
  python3 benchmarks/accuracy/benchmark_accuracy.py \\
      --dataset /path/to/V1_01_easy \\
      --config /path/to/custom_config.yaml

  # 4. Run self-test using synthetic circular trajectory (no external files needed):
  python3 benchmarks/accuracy/benchmark_accuracy.py --synthetic --time-cutoff 10.0
"""

import argparse
import csv
import datetime
import math
import os
import struct
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image

# -----------------------------------------------------------------------------
# Dynamic Path Setup for pool_slam module
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

BUILD_DIRS = [
    os.path.join(REPO_ROOT, "build", "python", "bindings"),
    os.path.join(REPO_ROOT, "build", "lib"),
    os.path.join(REPO_ROOT, "build"),
]
for p in BUILD_DIRS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    import pool_slam
except ImportError as err:
    sys.stderr.write(
        f"[ERROR] Could not import 'pool_slam' module: {err}\n"
        "Ensure the bindings are built via:\n"
        "  cmake -B build -DBUILD_PYTHON_BINDINGS=ON && cmake --build build --target pool_slam\n"
    )
    sys.exit(1)

DEFAULT_VOCAB_PATH = os.path.join(REPO_ROOT, "benchmarks", "accuracy", "vocab.bin")


# -----------------------------------------------------------------------------
# Vocabulary File Generator
# -----------------------------------------------------------------------------
def ensure_vocab_file(path: str) -> str:
    """Ensures a valid binary BoW vocabulary exists on disk."""
    if os.path.exists(path) and os.path.getsize(path) >= 395520:
        return os.path.abspath(path)

    # If default vocab exists and caller did not specify a distinct custom path, reuse default
    if (not path or os.path.abspath(path) == os.path.abspath(DEFAULT_VOCAB_PATH)) and \
       os.path.exists(DEFAULT_VOCAB_PATH) and os.path.getsize(DEFAULT_VOCAB_PATH) >= 395520:
        return os.path.abspath(DEFAULT_VOCAB_PATH)

    target_path = os.path.abspath(path if path else DEFAULT_VOCAB_PATH)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "wb") as f:
        # 1110 internal tree nodes (32 bytes each)
        for i in range(1110):
            w0 = (i * 0x0101010101010101) & 0xFFFFFFFFFFFFFFFF
            w1 = (~i) & 0xFFFFFFFFFFFFFFFF
            w2 = (i ^ 0x5555555555555555) & 0xFFFFFFFFFFFFFFFF
            w3 = (i + 17) & 0xFFFFFFFFFFFFFFFF
            f.write(struct.pack("<4Q", w0, w1, w2, w3))

        # 10000 leaf visual words (32 bytes bits + 4 bytes float32 idf)
        default_idf = 1.0
        for w in range(10000):
            w0 = (w * 0xABCDEF0123456789) & 0xFFFFFFFFFFFFFFFF
            w1 = (~(w * 3)) & 0xFFFFFFFFFFFFFFFF
            w2 = (w ^ 0xF0F0F0F0F0F0F0F0) & 0xFFFFFFFFFFFFFFFF
            w3 = (w + 99) & 0xFFFFFFFFFFFFFFFF
            f.write(struct.pack("<4Qf", w0, w1, w2, w3, default_idf))

    return target_path


# -----------------------------------------------------------------------------
# Math and Geometry Helpers
# -----------------------------------------------------------------------------
def interpolate_pose(
    gt_times: np.ndarray,
    gt_positions: np.ndarray,
    gt_quaternions: np.ndarray,
    query_time: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolates position and SLERPs quaternion at query_time."""
    if len(gt_times) == 0 or len(gt_positions) == 0:
        return np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    idx = int(np.searchsorted(gt_times, query_time))
    if idx == 0:
        return gt_positions[0].copy(), gt_quaternions[0].copy()
    if idx >= len(gt_times):
        return gt_positions[-1].copy(), gt_quaternions[-1].copy()

    t0, t1 = gt_times[idx - 1], gt_times[idx]
    if t1 <= t0:
        return gt_positions[idx].copy(), gt_quaternions[idx].copy()

    alpha = float((query_time - t0) / (t1 - t0))
    pos = (1.0 - alpha) * gt_positions[idx - 1] + alpha * gt_positions[idx]

    # Quaternion SLERP (quaternions stored as [w, x, y, z])
    q0 = gt_quaternions[idx - 1]
    q1 = gt_quaternions[idx]
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        q_interp = (1.0 - alpha) * q0 + alpha * q1
    else:
        theta = math.acos(max(-1.0, min(1.0, dot)))
        sin_theta = math.sin(theta)
        q_interp = (math.sin((1.0 - alpha) * theta) * q0 + math.sin(alpha * theta) * q1) / sin_theta

    norm = float(np.linalg.norm(q_interp))
    if norm > 1e-9:
        q_interp /= norm

    return pos, q_interp


def compute_rotation_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Computes angular distance in degrees between two quaternions [w, x, y, z]."""
    dot = abs(float(np.dot(q1, q2)))
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot) * (180.0 / math.pi)


def umeyama_alignment(model: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes optimal SE(3) rigid transform [R, t] mapping data onto model:
    model ≈ R @ data + t. Returns (R, t, rmse).
    """
    assert model.shape == data.shape
    n = model.shape[0]
    if n < 3:
        return np.eye(3), np.zeros(3), 0.0

    mu_m = np.mean(model, axis=0)
    mu_d = np.mean(data, axis=0)

    mc = model - mu_m
    dc = data - mu_d

    H = dc.T @ mc / n
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_m - R @ mu_d
    aligned = (R @ data.T).T + t
    residuals = np.linalg.norm(model - aligned, axis=1)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return R, t, rmse


# -----------------------------------------------------------------------------
# EuRoC Dataset Loader & Calibrator
# -----------------------------------------------------------------------------
def load_cam0_calibration(cam_dir: str) -> Optional[Dict[str, Any]]:
    """Loads camera resolution, intrinsics (K), and extrinsics (T_BC) from sensor.yaml."""
    yaml_path = os.path.join(cam_dir, "sensor.yaml")
    if not os.path.isfile(yaml_path):
        return None
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        res = data.get("resolution", [752, 480])
        intr = data.get("intrinsics", [458.654, 457.296, 367.215, 248.375])
        dist = data.get("distortion_coefficients", [-0.283408, 0.073959, 0.000194, 0.000018])
        t_bs_raw = data.get("T_BS", {}).get("data", None)
        t_bc = None
        if t_bs_raw and len(t_bs_raw) == 16:
            t_bc = np.array(t_bs_raw, dtype=np.float64).reshape((4, 4)).tolist()

        return {
            "width": int(res[0]),
            "height": int(res[1]),
            "fx": float(intr[0]),
            "fy": float(intr[1]),
            "cx": float(intr[2]),
            "cy": float(intr[3]),
            "dist": (float(dist[0]), float(dist[1]), float(dist[2]), float(dist[3])),
            "t_bc": t_bc,
        }
    except Exception:
        return None


class EurocDataset:
    def __init__(self, root_dir: str, time_cutoff: Optional[float] = None):
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        if os.path.isdir(os.path.join(self.root_dir, "mav0")):
            self.root_dir = os.path.join(self.root_dir, "mav0")

        self.cam_dir = os.path.join(self.root_dir, "cam0")
        self.imu_dir = os.path.join(self.root_dir, "imu0")

        if not os.path.isdir(self.cam_dir) or not os.path.isdir(self.imu_dir):
            raise FileNotFoundError(
                f"Dataset at '{self.root_dir}' must contain 'cam0' and 'imu0' subdirectories."
            )

        # Ground truth search
        self.gt_csv = None
        for candidate in [
            os.path.join(self.root_dir, "state_groundtruth_estimate0", "data.csv"),
            os.path.join(self.root_dir, "vicon0", "data.csv"),
        ]:
            if os.path.isfile(candidate):
                self.gt_csv = candidate
                break

        if not self.gt_csv:
            raise FileNotFoundError(
                f"No ground truth CSV found in '{self.root_dir}' (checked state_groundtruth_estimate0 and vicon0)."
            )

        self.time_cutoff = time_cutoff

        # Parse ground truth
        self.gt_times: List[float] = []
        self.gt_positions: List[np.ndarray] = []
        self.gt_quaternions: List[np.ndarray] = []
        self._load_ground_truth()

        # Parse IMU
        self.imu_samples: List[Tuple[float, np.ndarray, np.ndarray]] = []
        self._load_imu()

        # Parse Camera frames
        self.frame_records: List[Tuple[float, str]] = []
        self._load_frames()

        # Calibration parameters (K, distortion, extrinsics)
        calib = load_cam0_calibration(self.cam_dir)
        if calib:
            self.width = calib["width"]
            self.height = calib["height"]
            self.fx = calib["fx"]
            self.fy = calib["fy"]
            self.cx = calib["cx"]
            self.cy = calib["cy"]
            self.dist = calib["dist"]
            self.t_bc = calib["t_bc"]
            self.calib_source = f"{os.path.join(self.cam_dir, 'sensor.yaml')}"
        else:
            with Image.open(self.frame_records[0][1]) as first_img:
                self.width, self.height = first_img.size
            self.fx = 458.654
            self.fy = 457.296
            self.cx = 367.215
            self.cy = 248.375
            self.dist = (-0.283408, 0.073959, 0.000194, 0.000018)
            self.t_bc = None
            self.calib_source = "Default EuRoC MT9V034 Factory Calibration"

        if not self.frame_records:
            raise FileNotFoundError(f"No valid image frames found in '{self.cam_dir}'.")
        if not self.imu_samples:
            raise FileNotFoundError(f"No valid IMU samples found in '{self.imu_dir}'.")

        # Reference timestamps relative to sequence start (t0)
        start_candidates = [self.frame_records[0][0], self.imu_samples[0][0]]
        if self.gt_times:
            start_candidates.append(self.gt_times[0])
        self.t0 = min(start_candidates)

        if self.gt_times:
            self.gt_times = np.array(self.gt_times) - self.t0
            self.gt_positions = np.array(self.gt_positions)
            self.gt_quaternions = np.array(self.gt_quaternions)
        else:
            self.gt_times = np.empty((0,), dtype=np.float64)
            self.gt_positions = np.empty((0, 3), dtype=np.float64)
            self.gt_quaternions = np.empty((0, 4), dtype=np.float64)

        self.imu_samples = [
            (t - self.t0, a, w) for (t, a, w) in self.imu_samples
        ]
        self.frame_records = [
            (t - self.t0, path) for (t, path) in self.frame_records
        ]

        # Align frame start to ground truth start so initial_pose and synthetic depth match frame 0
        if len(self.gt_times) > 0:
            gt_start = float(self.gt_times[0])
            aligned_frames = [rec for rec in self.frame_records if rec[0] >= gt_start]
            if aligned_frames:
                self.frame_records = aligned_frames

        # Apply time cutoff if specified
        if self.time_cutoff is not None and self.time_cutoff > 0:
            filtered_frames = [
                rec for rec in self.frame_records if rec[0] <= self.time_cutoff
            ]
            if filtered_frames:
                self.frame_records = filtered_frames
            filtered_imu = [
                rec for rec in self.imu_samples if rec[0] <= self.time_cutoff + 0.1
            ]
            if filtered_imu:
                self.imu_samples = filtered_imu
            if len(self.gt_times) > 0:
                gt_mask = self.gt_times <= self.time_cutoff + 0.1
                if np.any(gt_mask):
                    self.gt_times = self.gt_times[gt_mask]
                    self.gt_positions = self.gt_positions[gt_mask]
                    self.gt_quaternions = self.gt_quaternions[gt_mask]

        self.duration = float(self.frame_records[-1][0] - self.frame_records[0][0])

    def _load_ground_truth(self):
        with open(self.gt_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue
                t_sec = float(row[0].strip()) * 1e-9
                px, py, pz = float(row[1]), float(row[2]), float(row[3])
                qw, qx, qy, qz = float(row[4]), float(row[5]), float(row[6]), float(row[7])
                self.gt_times.append(t_sec)
                self.gt_positions.append(np.array([px, py, pz], dtype=np.float64))
                self.gt_quaternions.append(np.array([qw, qx, qy, qz], dtype=np.float64))

    def _load_imu(self):
        imu_csv = os.path.join(self.imu_dir, "data.csv")
        with open(imu_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue
                t_sec = float(row[0].strip()) * 1e-9
                wx, wy, wz = float(row[1]), float(row[2]), float(row[3])
                ax, ay, az = float(row[4]), float(row[5]), float(row[6])
                self.imu_samples.append((
                    t_sec,
                    np.array([ax, ay, az], dtype=np.float64),
                    np.array([wx, wy, wz], dtype=np.float64),
                ))

    def _load_frames(self):
        cam_csv = os.path.join(self.cam_dir, "data.csv")
        data_subfolder = os.path.join(self.cam_dir, "data")
        with open(cam_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue
                t_sec = float(row[0].strip()) * 1e-9
                filename = row[1].strip()
                img_path = (
                    os.path.join(self.cam_dir, filename)
                    if os.path.isfile(os.path.join(self.cam_dir, filename))
                    else os.path.join(data_subfolder, filename)
                )
                self.frame_records.append((t_sec, img_path))


# -----------------------------------------------------------------------------
# Synthetic Dataset Generator for Self-Testing
# -----------------------------------------------------------------------------
class SyntheticDataset:
    """Generates synthetic visual, inertial, and ground truth data for self-testing."""
    def __init__(self, duration: float = 12.0, width: int = 752, height: int = 480):
        self.duration = duration
        self.width = width
        self.height = height
        self.root_dir = "[SYNTHETIC_SELF_TEST]"
        self.calib_source = "Synthetic Ideal Calibration"

        self.fx = 458.654
        self.fy = 457.296
        self.cx = 367.215
        self.cy = 248.375
        self.dist = (0.0, 0.0, 0.0, 0.0)
        self.t_bc = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        # Create GT trajectory: smooth circular yaw + gentle vertical bob
        num_gt = int(duration * 100) + 1
        self.gt_times = np.linspace(0.0, duration, num_gt)
        self.gt_positions = np.zeros((num_gt, 3))
        self.gt_quaternions = np.zeros((num_gt, 4))

        radius = 1.5
        omega = 0.2 * math.pi
        for i, t in enumerate(self.gt_times):
            yaw = omega * t
            self.gt_positions[i] = [
                radius * math.cos(yaw) - radius,
                radius * math.sin(yaw),
                1.0 + 0.1 * math.sin(0.5 * yaw),
            ]
            self.gt_quaternions[i] = [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]

        # IMU samples at 200 Hz
        num_imu = int(duration * 200) + 1
        self.imu_samples = []
        for i in range(num_imu):
            t = float(i) * 0.005
            a = np.array([0.0, 0.0, 9.81], dtype=np.float64)
            w = np.array([0.0, 0.0, omega], dtype=np.float64)
            self.imu_samples.append((t, a, w))

        # Frames at 20 Hz
        num_frames = int(duration * 20) + 1
        self.frame_records = []
        self._temp_dir = tempfile.TemporaryDirectory()
        for i in range(num_frames):
            t = float(i) * 0.05
            frame_path = os.path.join(self._temp_dir.name, f"synth_{i:05d}.png")
            arr = np.full((height, width), 128, dtype=np.uint8)
            for r in range(50, height - 50, 35):
                for c in range(50, width - 50, 35):
                    arr[r - 3 : r + 4, c - 3 : c + 4] = 230
            Image.fromarray(arr).save(frame_path)
            self.frame_records.append((t, frame_path))


# -----------------------------------------------------------------------------
# Configuration File Generation
# -----------------------------------------------------------------------------
def create_system_config_yaml(
    vocab_path: str,
    output_yaml: str,
    width: int = 752,
    height: int = 480,
    fx: float = 458.654,
    fy: float = 457.296,
    cx: float = 367.215,
    cy: float = 248.375,
    dist: Tuple[float, float, float, float] = (-0.283408, 0.073959, 0.000194, 0.000018),
    t_bc: Optional[List[List[float]]] = None,
    t_depth: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    initial_pose: Optional[List[List[float]]] = None,
) -> str:
    if t_bc is None:
        t_bc = [
            [0.01486554, -0.99988093, 0.00414030, -0.02164015],
            [0.99955725, 0.01496721, 0.02571553, -0.06467699],
            [-0.02577443, 0.00375619, 0.99966073, 0.00981073],
            [0.00000000, 0.00000000, 0.00000000, 1.00000000],
        ]

    initial_pose_block = ""
    if initial_pose is not None:
        initial_pose_block = (
            f"\ninitial_pose:\n"
            f"  - [{initial_pose[0][0]}, {initial_pose[0][1]}, {initial_pose[0][2]}, {initial_pose[0][3]}]\n"
            f"  - [{initial_pose[1][0]}, {initial_pose[1][1]}, {initial_pose[1][2]}, {initial_pose[1][3]}]\n"
            f"  - [{initial_pose[2][0]}, {initial_pose[2][1]}, {initial_pose[2][2]}, {initial_pose[2][3]}]\n"
            f"  - [{initial_pose[3][0]}, {initial_pose[3][1]}, {initial_pose[3][2]}, {initial_pose[3][3]}]\n"
        )

    content = f"""pipeline:
  bow_vocabulary_path: "{vocab_path}"

camera:
  width: {width}
  height: {height}
  intrinsics:
    fx: {fx}
    fy: {fy}
    cx: {cx}
    cy: {cy}
  distortion_coeffs: [{dist[0]}, {dist[1]}, {dist[2]}, {dist[3]}]

extrinsics:
  T_body_camera:
    - [{t_bc[0][0]}, {t_bc[0][1]}, {t_bc[0][2]}, {t_bc[0][3]}]
    - [{t_bc[1][0]}, {t_bc[1][1]}, {t_bc[1][2]}, {t_bc[1][3]}]
    - [{t_bc[2][0]}, {t_bc[2][1]}, {t_bc[2][2]}, {t_bc[2][3]}]
    - [{t_bc[3][0]}, {t_bc[3][1]}, {t_bc[3][2]}, {t_bc[3][3]}]
  t_body_depth: [{t_depth[0]}, {t_depth[1]}, {t_depth[2]}]{initial_pose_block}
"""
    with open(output_yaml, "w", encoding="utf-8") as f:
        f.write(content)
    return output_yaml


# -----------------------------------------------------------------------------
# Main Benchmark Execution Function
# -----------------------------------------------------------------------------
def run_accuracy_benchmark(
    dataset,
    vocab_path: str,
    custom_config_path: Optional[str] = None,
    eval_interval_sec: float = 5.0,
    depth_noise_sigma: float = 0.02,
    output_dir: str = "benchmarks/accuracy/runs",
    realtime: bool = False,
    playback_speed: float = 1.0,
) -> str:
    """Executes the accuracy benchmark and writes the timestamped report."""
    os.makedirs(output_dir, exist_ok=True)
    run_timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = os.path.join(output_dir, f"run_{run_timestamp_str}.txt")

    # Determine SystemConfig YAML
    temp_dir = tempfile.TemporaryDirectory()
    if custom_config_path and os.path.isfile(custom_config_path):
        cfg_yaml_path = custom_config_path
        calib_origin = f"Custom YAML ({custom_config_path})"
    else:
        cfg_yaml_path = os.path.join(temp_dir.name, "euroc_system_config.yaml")
        initial_pose = None
        if len(dataset.gt_times) > 0 and len(dataset.gt_positions) > 0:
            p0 = dataset.gt_positions[0]
            q0 = dataset.gt_quaternions[0]  # [w, x, y, z]
            qw, qx, qy, qz = float(q0[0]), float(q0[1]), float(q0[2]), float(q0[3])
            R0 = [
                [1.0 - 2.0 * (qy*qy + qz*qz), 2.0 * (qx*qy - qz*qw), 2.0 * (qx*qz + qy*qw)],
                [2.0 * (qx*qy + qz*qw), 1.0 - 2.0 * (qx*qx + qz*qz), 2.0 * (qy*qz - qx*qw)],
                [2.0 * (qx*qz - qy*qw), 2.0 * (qy*qz + qx*qw), 1.0 - 2.0 * (qx*qx + qy*qy)],
            ]
            initial_pose = [
                [R0[0][0], R0[0][1], R0[0][2], float(p0[0])],
                [R0[1][0], R0[1][1], R0[1][2], float(p0[1])],
                [R0[2][0], R0[2][1], R0[2][2], float(p0[2])],
                [0.0, 0.0, 0.0, 1.0],
            ]
        create_system_config_yaml(
            vocab_path=vocab_path,
            output_yaml=cfg_yaml_path,
            width=dataset.width,
            height=dataset.height,
            fx=dataset.fx,
            fy=dataset.fy,
            cx=dataset.cx,
            cy=dataset.cy,
            dist=dataset.dist,
            t_bc=dataset.t_bc,
            initial_pose=initial_pose,
        )
        calib_origin = getattr(dataset, "calib_source", "Auto-generated from sensor.yaml")

    # Initialize SLAM System
    sys_config = pool_slam.SystemConfig(cfg_yaml_path)
    system = pool_slam.System(sys_config)

    # Console banner
    print("\n" + "=" * 68)
    print("           Pool-SLAM State Estimation Accuracy Benchmark")
    print("=" * 68)
    print(f"Dataset             : {dataset.root_dir}")
    print(f"Calibration Origin  : {calib_origin}")
    print(f"Camera Matrix (K)   : fx={sys_config.fx:.3f}, fy={sys_config.fy:.3f}, cx={sys_config.cx:.3f}, cy={sys_config.cy:.3f}")
    print(f"Resolution          : {sys_config.w} x {sys_config.h}")
    print(f"Distortion (radtan) : [{sys_config.k1:.5f}, {sys_config.k2:.5f}, {sys_config.p1:.6f}, {sys_config.p2:.6f}]")
    print(f"Total Sequence Time : {dataset.duration:.2f} s ({len(dataset.frame_records)} frames)")
    print(f"Evaluation Interval : {eval_interval_sec:.2f} s")
    print(f"Depth Noise Sigma   : {depth_noise_sigma:.4f} m")
    print(f"Run Output File     : {report_filename}")
    print("-" * 68)

    system.start()

    # Pre-generate synthetic depth samples aligned with ground truth
    depth_samples = []
    np.random.seed(42)  # Deterministic depth noise across benchmark runs
    if len(dataset.gt_times) > 0 and len(dataset.gt_positions) > 0:
        for t_gt, pos in zip(dataset.gt_times, dataset.gt_positions):
            z_noisy = float(pos[2] + np.random.normal(0.0, depth_noise_sigma))
            depth_samples.append((float(t_gt), z_noisy))
    else:
        for t_frame, _ in dataset.frame_records:
            depth_samples.append((float(t_frame), 1.0))

    imu_idx = 0
    depth_idx = 0
    total_frames = len(dataset.frame_records)
    total_imu = len(dataset.imu_samples)

    checkpoints: List[Dict[str, Any]] = []
    next_eval_time = eval_interval_sec

    start_wall_time = time.time()
    last_frame_wall_time = start_wall_time

    for frame_idx, (t_frame, frame_path) in enumerate(dataset.frame_records):
        # 1. Push all IMU samples up to t_frame + 0.500 so Synchronizer has sufficient margin
        while imu_idx < total_imu and dataset.imu_samples[imu_idx][0] <= t_frame + 0.500:
            t_imu, a, w = dataset.imu_samples[imu_idx]
            system.push_imu(pool_slam.Imu(timestamp=t_imu, a=a, w=w, t=20.0))
            imu_idx += 1

        # 2. Push all Depth samples up to t_frame + 0.500 (100 samples ahead, well within buffer size 500)
        while depth_idx < len(depth_samples) and depth_samples[depth_idx][0] <= t_frame + 0.500:
            t_d, z_d = depth_samples[depth_idx]
            system.push_depth(pool_slam.Depth(timestamp=t_d, z=z_d))
            depth_idx += 1

        # 3. Ingestion pacing & throttling before pushing frame (Section 4.3)
        if realtime and frame_idx > 0:
            dt_dataset = t_frame - dataset.frame_records[frame_idx - 1][0]
            target_sleep = dt_dataset / max(0.01, playback_speed)
            elapsed_frame_time = time.time() - last_frame_wall_time
            if target_sleep > elapsed_frame_time:
                time.sleep(target_sleep - elapsed_frame_time)
            last_frame_wall_time = time.time()
        elif not realtime and frame_idx > 0:
            # Fast-forward mode:
            # 1. Ensure smooth frame spacing floor to avoid flooding buffer pool
            # 2. Throttle ingestion if Backend lags (> 0.15s) to avoid un-synchronized frame drops
            # Invariant: Total gap between frame pushes must NEVER exceed 250ms
            # to strictly prevent starving Hal ImageStream (400ms timeout) into FATALCAM.
            elapsed_since_push = time.time() - last_frame_wall_time
            max_safe_pause = max(0.0, 0.25 - elapsed_since_push)

            if elapsed_since_push < 0.06:
                sleep_needed = min(0.06 - elapsed_since_push, max_safe_pause)
                time.sleep(sleep_needed)

            t_throttle_start = time.time()
            rem_avail = max(0.0, 0.25 - (time.time() - last_frame_wall_time))
            while (t_frame - system.get_pose().timestamp) > 0.15 and (time.time() - t_throttle_start) < min(0.10, rem_avail):
                while imu_idx < total_imu and dataset.imu_samples[imu_idx][0] <= t_frame + 0.500:
                    t_imu, a, w = dataset.imu_samples[imu_idx]
                    system.push_imu(pool_slam.Imu(timestamp=t_imu, a=a, w=w, t=20.0))
                    imu_idx += 1
                while depth_idx < len(depth_samples) and depth_samples[depth_idx][0] <= t_frame + 0.500:
                    t_d, z_d = depth_samples[depth_idx]
                    system.push_depth(pool_slam.Depth(timestamp=t_d, z=z_d))
                    depth_idx += 1
                time.sleep(0.002)

        # 4. Read image and push frame with backpressure wait (Section 4.2)
        with Image.open(frame_path) as img:
            frame_arr = np.array(img.convert("L"), dtype=np.uint8)
        pushed = False
        for _ in range(50):
            if system.push_frame(frame_arr, t_frame):
                pushed = True
                break
            time.sleep(0.005)
        last_frame_wall_time = time.time()

        # 5. Check if current dataset time hits or passes next evaluation interval (Section 4.4)
        if t_frame >= next_eval_time:
            # Synchronize with backend worker thread so pose matches evaluation time
            # Limit blocking wait strictly < Hal 400ms timeout so ImageStream is never starved into FATALCAM
            rem_eval = max(0.0, 0.15 - (time.time() - last_frame_wall_time))
            t_wait_start = time.time()
            pose = system.get_pose()
            while pose.timestamp < next_eval_time - 1e-4 and (time.time() - t_wait_start) < min(0.10, rem_eval):
                while imu_idx < total_imu and dataset.imu_samples[imu_idx][0] <= t_frame + 0.500:
                    t_imu, a, w = dataset.imu_samples[imu_idx]
                    system.push_imu(pool_slam.Imu(timestamp=t_imu, a=a, w=w, t=20.0))
                    imu_idx += 1
                while depth_idx < len(depth_samples) and depth_samples[depth_idx][0] <= t_frame + 0.500:
                    t_d, z_d = depth_samples[depth_idx]
                    system.push_depth(pool_slam.Depth(timestamp=t_d, z=z_d))
                    depth_idx += 1
                time.sleep(0.002)
                pose = system.get_pose()

            if pose.timestamp < next_eval_time - 0.5 and pose.timestamp > 0.0:
                print(f"[Benchmark WARNING] Estimator lagging: checkpoint {next_eval_time:.1f}s, current pose {pose.timestamp:.2f}s", file=sys.stderr)

            eval_t = pose.timestamp if pose.timestamp > 0.0 else t_frame
            gt_pos, gt_quat = interpolate_pose(
                dataset.gt_times, dataset.gt_positions, dataset.gt_quaternions, eval_t
            )

            est_pos = np.asarray(pose.t, dtype=np.float64)
            est_quat = np.asarray(pose.r, dtype=np.float64)
            est_vel = np.asarray(pose.v, dtype=np.float64)

            pos_err = float(np.linalg.norm(est_pos - gt_pos))
            horiz_err = float(np.linalg.norm(est_pos[:2] - gt_pos[:2]))
            depth_err = float(abs(est_pos[2] - gt_pos[2]))
            rot_err = compute_rotation_angle_deg(est_quat, gt_quat)

            now_wall = time.time()
            elapsed_wall = now_wall - start_wall_time
            rtf = eval_t / max(0.001, elapsed_wall)

            checkpoint_data = {
                "checkpoint": len(checkpoints) + 1,
                "dataset_time": eval_t,
                "wall_time": elapsed_wall,
                "est_pos": est_pos,
                "gt_pos": gt_pos,
                "est_quat": est_quat,
                "gt_quat": gt_quat,
                "est_vel": est_vel,
                "pos_err": pos_err,
                "horiz_err": horiz_err,
                "depth_err": depth_err,
                "rot_err": rot_err,
            }
            checkpoints.append(checkpoint_data)

            # Concise single-line console output
            print(
                f"[{eval_t:6.1f}s / {dataset.duration:5.1f}s] "
                f"Pos Err: {pos_err:6.3f}m (Horiz: {horiz_err:5.3f}m, Depth: {depth_err:5.3f}m) | "
                f"Rot Err: {rot_err:5.2f}° | RTF: {rtf:4.2f}x"
            )

            next_eval_time += eval_interval_sec

    # Drain remaining leading IMU/depth processing and cleanly stop pipeline
    while imu_idx < total_imu:
        t_imu, a, w = dataset.imu_samples[imu_idx]
        system.push_imu(pool_slam.Imu(timestamp=t_imu, a=a, w=w, t=20.0))
        imu_idx += 1
    while depth_idx < len(depth_samples):
        t_d, z_d = depth_samples[depth_idx]
        system.push_depth(pool_slam.Depth(timestamp=t_d, z=z_d))
        depth_idx += 1
    t_drain_start = time.time()
    while system.get_pose().timestamp < dataset.duration - 0.10 and (time.time() - t_drain_start) < 0.15:
        time.sleep(0.005)
    system.stop()
    total_wall_time = time.time() - start_wall_time

    # Compute trajectory error statistics
    if checkpoints:
        pos_errors = [c["pos_err"] for c in checkpoints]
        depth_errors = [c["depth_err"] for c in checkpoints]
        rot_errors = [c["rot_err"] for c in checkpoints]

        mean_pos_err = float(np.mean(pos_errors))
        rmse_pos_err = float(np.sqrt(np.mean(np.array(pos_errors) ** 2)))
        max_pos_err = float(np.max(pos_errors))
        med_pos_err = float(np.median(pos_errors))

        mean_depth_err = float(np.mean(depth_errors))
        rmse_depth_err = float(np.sqrt(np.mean(np.array(depth_errors) ** 2)))
        max_depth_err = float(np.max(depth_errors))

        mean_rot_err = float(np.mean(rot_errors))
        rmse_rot_err = float(np.sqrt(np.mean(np.array(rot_errors) ** 2)))
        max_rot_err = float(np.max(rot_errors))

        # Umeyama SE(3) Rigid Alignment on evaluated checkpoints
        est_mat = np.array([c["est_pos"] for c in checkpoints])
        gt_mat = np.array([c["gt_pos"] for c in checkpoints])
        _, _, aligned_rmse = umeyama_alignment(gt_mat, est_mat)
    else:
        mean_pos_err = rmse_pos_err = max_pos_err = med_pos_err = 0.0
        mean_depth_err = rmse_depth_err = max_depth_err = 0.0
        mean_rot_err = rmse_rot_err = max_rot_err = 0.0
        aligned_rmse = 0.0

    rtf_overall = dataset.duration / max(0.001, total_wall_time)

    # Console summary
    print("-" * 68)
    print("Benchmark Run Complete!")
    print(f"  Processed Frames      : {total_frames}")
    print(f"  Total IMU Samples     : {imu_idx}")
    print(f"  Checkpoints Evaluated : {len(checkpoints)}")
    print(f"  Position Error (ATE)  : RMSE={rmse_pos_err:.4f}m, Mean={mean_pos_err:.4f}m, Max={max_pos_err:.4f}m")
    print(f"  Depth (Z) Error       : RMSE={rmse_depth_err:.4f}m, Mean={mean_depth_err:.4f}m")
    print(f"  Rotation Error        : RMSE={rmse_rot_err:.2f}°, Mean={mean_rot_err:.2f}°")
    print(f"  SE(3) Aligned ATE     : RMSE={aligned_rmse:.4f}m")
    print(f"  Real-Time Factor      : {rtf_overall:.2f}x ({dataset.duration:.1f}s in {total_wall_time:.1f}s wall-time)")
    print(f"  Full Report Written   : {report_filename}")
    print("=" * 68 + "\n")

    # Write formatted report file
    with open(report_filename, "w", encoding="utf-8") as out:
        out.write("=" * 80 + "\n")
        out.write("                    Pool-SLAM Accuracy Benchmark Report\n")
        out.write("=" * 80 + "\n")
        out.write(f"Run Timestamp          : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write(f"Dataset Path           : {dataset.root_dir}\n")
        out.write(f"Calibration Source     : {calib_origin}\n")
        out.write(f"Camera Matrix (K)      : fx={sys_config.fx:.4f}, fy={sys_config.fy:.4f}, cx={sys_config.cx:.4f}, cy={sys_config.cy:.4f}\n")
        out.write(f"Camera Resolution      : {sys_config.w} x {sys_config.h}\n")
        out.write(f"Distortion (radtan)    : k1={sys_config.k1:.6f}, k2={sys_config.k2:.6f}, p1={sys_config.p1:.6f}, p2={sys_config.p2:.6f}\n")
        out.write(f"Sequence Duration      : {dataset.duration:.2f} s\n")
        out.write(f"Evaluation Interval    : {eval_interval_sec:.2f} s\n")
        out.write(f"Depth Noise Sigma      : {depth_noise_sigma:.4f} m\n")
        out.write(f"Total Frames Processed : {total_frames}\n")
        out.write(f"Total IMU Samples      : {imu_idx}\n")
        out.write(f"Wall-Clock Duration    : {total_wall_time:.2f} s\n")
        out.write(f"Real-Time Factor (RTF) : {rtf_overall:.2f}x\n\n")

        out.write("-" * 80 + "\n")
        out.write("Periodic Checkpoint Evaluations\n")
        out.write("-" * 80 + "\n")
        for cp in checkpoints:
            out.write(
                f"[Checkpoint {cp['checkpoint']:02d}] "
                f"Dataset Time: {cp['dataset_time']:6.2f}s | Wall Time: {cp['wall_time']:6.2f}s\n"
            )
            out.write(
                f"  Estimated Position (m) : X={cp['est_pos'][0]:+8.4f}, "
                f"Y={cp['est_pos'][1]:+8.4f}, Z={cp['est_pos'][2]:+8.4f}\n"
            )
            out.write(
                f"  Ground Truth Pos   (m) : X={cp['gt_pos'][0]:+8.4f}, "
                f"Y={cp['gt_pos'][1]:+8.4f}, Z={cp['gt_pos'][2]:+8.4f}\n"
            )
            out.write(
                f"  Estimated Quat  (wxyz) : [{cp['est_quat'][0]:.4f}, {cp['est_quat'][1]:.4f}, "
                f"{cp['est_quat'][2]:.4f}, {cp['est_quat'][3]:.4f}]\n"
            )
            out.write(
                f"  Ground Truth Quat (wxyz): [{cp['gt_quat'][0]:.4f}, {cp['gt_quat'][1]:.4f}, "
                f"{cp['gt_quat'][2]:.4f}, {cp['gt_quat'][3]:.4f}]\n"
            )
            out.write(
                f"  Estimated Velocity(m/s): [{cp['est_vel'][0]:+.3f}, {cp['est_vel'][1]:+.3f}, {cp['est_vel'][2]:+.3f}]\n"
            )
            out.write(
                f"  Position Error         : {cp['pos_err']:.4f} m "
                f"(Horizontal: {cp['horiz_err']:.4f} m, Depth: {cp['depth_err']:.4f} m)\n"
            )
            out.write(f"  Rotation Error         : {cp['rot_err']:.2f} deg\n\n")

        out.write("-" * 80 + "\n")
        out.write("Summary Statistics & Error Metrics\n")
        out.write("-" * 80 + "\n")
        out.write("Position Error (Absolute Trajectory Error):\n")
        out.write(f"  RMSE Error  : {rmse_pos_err:.4f} m\n")
        out.write(f"  Mean Error  : {mean_pos_err:.4f} m\n")
        out.write(f"  Median Error: {med_pos_err:.4f} m\n")
        out.write(f"  Max Error   : {max_pos_err:.4f} m\n\n")

        out.write("Vertical / Depth Sensor Error:\n")
        out.write(f"  RMSE Error  : {rmse_depth_err:.4f} m\n")
        out.write(f"  Mean Error  : {mean_depth_err:.4f} m\n")
        out.write(f"  Max Error   : {max_depth_err:.4f} m\n\n")

        out.write("Orientation Error:\n")
        out.write(f"  RMSE Error  : {rmse_rot_err:.2f} deg\n")
        out.write(f"  Mean Error  : {mean_rot_err:.2f} deg\n")
        out.write(f"  Max Error   : {max_rot_err:.2f} deg\n\n")

        out.write("SE(3) Umeyama Rigid Trajectory Alignment:\n")
        out.write(f"  Aligned Position RMSE : {aligned_rmse:.4f} m\n")
        out.write("=" * 80 + "\n")

    return report_filename


# -----------------------------------------------------------------------------
# Entry Point & CLI Parser
# -----------------------------------------------------------------------------
def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_accuracy.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
================================================================================
                    Pool-SLAM State Estimation Accuracy Benchmark
================================================================================
Evaluates Pool-SLAM trajectory estimation accuracy against a ground-truth dataset
(such as the EuRoC MAV V1_01_easy sequence).

Features:
  - Ingests synchronized monocular camera frames (cam0) and 6-axis IMU (imu0).
  - Automatically loads camera intrinsics (K), radtan distortion, and extrinsics
    from the dataset's 'cam0/sensor.yaml' (or uses --config if provided).
  - Generates synthetic 1D depth readings from ground-truth vertical position + noise.
  - Checks state estimation at a fixed dataset time interval (default: every 5.0s).
  - Writes detailed checkpoint telemetry and ATE RMSE summary to a timestamped file
    in 'benchmarks/accuracy/runs/run_YYYYMMDD_HHMMSS.txt'.
""",
        epilog="""
--------------------------------------------------------------------------------
CLI Usage Examples:
--------------------------------------------------------------------------------
  # 1. Standard benchmark on extracted EuRoC V1_01_easy (first 60s):
  python3 benchmarks/accuracy/benchmark_accuracy.py \\
      --dataset /path/to/V1_01_easy \\
      --time-cutoff 60.0

  # 2. Benchmark every 3 seconds with depth noise sigma = 0.01m:
  python3 benchmarks/accuracy/benchmark_accuracy.py \\
      --dataset /path/to/V1_01_easy \\
      --eval-interval 3.0 \\
      --depth-noise-sigma 0.01

  # 3. Supply a custom SystemConfig YAML file (overriding auto-calibration):
  python3 benchmarks/accuracy/benchmark_accuracy.py \\
      --dataset /path/to/V1_01_easy \\
      --config /path/to/my_config.yaml

  # 4. Run internal self-test using synthetic circular trajectory:
  python3 benchmarks/accuracy/benchmark_accuracy.py \\
      --synthetic \\
      --time-cutoff 12.0
================================================================================
""",
    )

    # Core Dataset Arguments
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the dataset directory containing 'cam0' and 'imu0' (e.g. /data/V1_01_easy or .../mav0)",
    )
    parser.add_argument(
        "--time-cutoff",
        "-t",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Maximum dataset elapsed time in seconds to process (e.g. 60.0). Defaults to processing full sequence.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to custom SystemConfig YAML file. If omitted, K, distortion, and T_BC are auto-loaded from cam0/sensor.yaml.",
    )

    # Evaluation Arguments
    parser.add_argument(
        "--eval-interval",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Interval in seconds of dataset time between state check evaluations (default: 5.0).",
    )
    parser.add_argument(
        "--depth-noise-sigma",
        type=float,
        default=0.02,
        metavar="METERS",
        help="Gaussian noise standard deviation added to ground-truth depth readings in meters (default: 0.02).",
    )

    # Resource & Output Paths
    parser.add_argument(
        "--vocab",
        type=str,
        default=DEFAULT_VOCAB_PATH,
        metavar="PATH",
        help="Path to BoW binary vocabulary file (default: benchmarks/accuracy/vocab.bin).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/accuracy/runs",
        metavar="DIR",
        help="Directory to write timestamped benchmark run reports (default: benchmarks/accuracy/runs).",
    )

    # Playback & Testing Modes
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Throttle image feeding to match dataset timestamp pacing.",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        metavar="SPEED",
        help="Playback speed multiplier when --realtime is enabled (default: 1.0).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run self-test using internally generated synthetic visual/inertial data (no dataset required).",
    )

    return parser


def main():
    parser = build_cli_parser()
    args = parser.parse_args()

    if not args.dataset and not args.synthetic:
        parser.error(
            "Please provide a dataset path via --dataset <path> (or -d <path>), "
            "or use --synthetic for self-testing."
        )

    # Ensure vocabulary file exists
    args.vocab = ensure_vocab_file(args.vocab)

    # Load dataset
    if args.synthetic:
        duration = args.time_cutoff if (args.time_cutoff and args.time_cutoff > 0) else 15.0
        dataset = SyntheticDataset(duration=duration)
    else:
        dataset = EurocDataset(args.dataset, time_cutoff=args.time_cutoff)

    run_accuracy_benchmark(
        dataset=dataset,
        vocab_path=args.vocab,
        custom_config_path=args.config,
        eval_interval_sec=args.eval_interval,
        depth_noise_sigma=args.depth_noise_sigma,
        output_dir=args.output_dir,
        realtime=args.realtime,
        playback_speed=args.playback_speed,
    )


if __name__ == "__main__":
    main()
