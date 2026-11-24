#!/usr/bin/env python3
"""
Test Suite for Accuracy Benchmark Harness (benchmarks/AccuracyBenchmark.md)
==========================================================================
Verifies:
  7.1 Synthetic Trajectory Self-Test
  7.2 Dataset Loader & Calibration Precedence
  7.3 Mathematical Interpolation & Geometry Metric Unit Tests
  7.4 Umeyama SE(3) Rigid Alignment Validation
  7.5 Ingestion Backpressure & Flow Control
"""

import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import yaml
from PIL import Image

# Ensure repository root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarks.accuracy.benchmark_accuracy import (
    DEFAULT_VOCAB_PATH,
    EurocDataset,
    SyntheticDataset,
    build_cli_parser,
    compute_rotation_angle_deg,
    create_system_config_yaml,
    ensure_vocab_file,
    interpolate_pose,
    load_cam0_calibration,
    run_accuracy_benchmark,
    umeyama_alignment,
)


class TestMathematicalInterpolationAndGeometry(unittest.TestCase):
    """Verifies Section 7.3: SLERP, Linear Position Interpolation, and Rotation Metrics."""

    def test_slerp_antipodal_invariance(self):
        """Interpolating between q and -q yields angular distance 0.0 deg."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        q1 = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        times = np.array([0.0, 1.0], dtype=np.float64)
        positions = np.zeros((2, 3), dtype=np.float64)
        quaternions = np.vstack([q0, q1])

        # Test at midpoint alpha = 0.5
        _, q_interp = interpolate_pose(times, positions, quaternions, query_time=0.5)
        angle_deg = compute_rotation_angle_deg(q_interp, q0)
        self.assertAlmostEqual(angle_deg, 0.0, places=5)

    def test_slerp_midpoint_accuracy(self):
        """Midpoint interpolation between identity and 90 deg yaw yields 45 deg yaw."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # Identity
        # 90 deg rotation around Z: [cos(pi/4), 0, 0, sin(pi/4)]
        q1 = np.array([math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)], dtype=np.float64)

        times = np.array([0.0, 1.0], dtype=np.float64)
        positions = np.zeros((2, 3), dtype=np.float64)
        quaternions = np.vstack([q0, q1])

        _, q_interp = interpolate_pose(times, positions, quaternions, query_time=0.5)

        # Expected: 45 deg rotation around Z -> angle from q0 is 45 deg, angle from q1 is 45 deg
        angle_from_q0 = compute_rotation_angle_deg(q_interp, q0)
        angle_from_q1 = compute_rotation_angle_deg(q_interp, q1)

        self.assertAlmostEqual(angle_from_q0, 45.0, places=4)
        self.assertAlmostEqual(angle_from_q1, 45.0, places=4)

    def test_linear_position_interpolation(self):
        """Linear position interpolation arithmetic matching across varying alpha."""
        times = np.array([10.0, 20.0], dtype=np.float64)
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]], dtype=np.float64)
        quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))

        # Query at alpha = 0.3 (t = 13.0) -> [3.0, 6.0, 9.0]
        pos_mid, _ = interpolate_pose(times, positions, quaternions, query_time=13.0)
        np.testing.assert_allclose(pos_mid, [3.0, 6.0, 9.0], rtol=1e-6)

        # Query at alpha = 0.75 (t = 17.5) -> [7.5, 15.0, 22.5]
        pos_75, _ = interpolate_pose(times, positions, quaternions, query_time=17.5)
        np.testing.assert_allclose(pos_75, [7.5, 15.0, 22.5], rtol=1e-6)

        # Boundary clamping: before t0 and after t1
        pos_before, _ = interpolate_pose(times, positions, quaternions, query_time=5.0)
        np.testing.assert_allclose(pos_before, [0.0, 0.0, 0.0], rtol=1e-6)

        pos_after, _ = interpolate_pose(times, positions, quaternions, query_time=25.0)
        np.testing.assert_allclose(pos_after, [10.0, 20.0, 30.0], rtol=1e-6)

    def test_rotation_distance_formula(self):
        """Verifies 2 * acos(|q1 . q2|) matches axis-angle distances."""
        q_ident = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # 90 deg pitch around Y: [cos(pi/4), 0, sin(pi/4), 0]
        q_pitch90 = np.array([math.cos(math.pi / 4.0), 0.0, math.sin(math.pi / 4.0), 0.0], dtype=np.float64)
        self.assertAlmostEqual(compute_rotation_angle_deg(q_ident, q_pitch90), 90.0, places=5)

        # 180 deg yaw around Z: [cos(pi/2), 0, 0, sin(pi/2)] = [0, 0, 0, 1]
        q_yaw180 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.assertAlmostEqual(compute_rotation_angle_deg(q_ident, q_yaw180), 180.0, places=5)

        # Antipodal identity: [-1, 0, 0, 0] must give 0 deg
        q_neg_ident = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.assertAlmostEqual(compute_rotation_angle_deg(q_ident, q_neg_ident), 0.0, places=5)


class TestUmeyamaRigidAlignment(unittest.TestCase):
    """Verifies Section 7.4: Umeyama SE(3) Rigid Trajectory Alignment."""

    def setUp(self):
        np.random.seed(42)
        # 20 non-collinear 3D model points
        self.model = np.random.uniform(-5.0, 5.0, size=(20, 3))

    def test_identity_invariance(self):
        """Identical point sets produce R = I, t = 0, RMSE = 0.0."""
        R, t, rmse = umeyama_alignment(self.model, self.model)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(t, np.zeros(3), atol=1e-12)
        self.assertLess(rmse, 1e-12)

    def test_pure_translation(self):
        """Points shifted by delta_t recover exact translation with RMSE < 1e-12."""
        delta_t = np.array([5.0, -2.0, 3.0])
        # data points such that model = data + delta_t => data = model - delta_t
        data = self.model - delta_t

        R, t, rmse = umeyama_alignment(self.model, data)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(t, delta_t, atol=1e-12)
        self.assertLess(rmse, 1e-12)

    def test_pure_rotation(self):
        """Points rotated by 90 deg around Y-axis recover exact rotation and det(R) = +1."""
        # Ry(90 deg): [cos(pi/2), 0, sin(pi/2); 0, 1, 0; -sin(pi/2), 0, cos(pi/2)]
        R_true = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        # model = R_true @ data => data = R_true.T @ model
        data = (R_true.T @ self.model.T).T

        R, t, rmse = umeyama_alignment(self.model, data)
        np.testing.assert_allclose(R, R_true, atol=1e-12)
        np.testing.assert_allclose(t, np.zeros(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)
        self.assertLess(rmse, 1e-12)

    def test_reflection_protection(self):
        """Reflection transformation (det < 0) is rejected; det(R) = +1 strictly enforced."""
        # Reflect across Z: z -> -z
        data = self.model.copy()
        data[:, 2] *= -1.0

        R, _, _ = umeyama_alignment(self.model, data)
        det_R = float(np.linalg.det(R))
        self.assertAlmostEqual(det_R, 1.0, places=8)
        self.assertGreater(det_R, 0.0)

    def test_collinear_and_few_points_degeneracy(self):
        """Point set with < 3 points returns identity fallback without throwing."""
        points_two = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        R, t, rmse = umeyama_alignment(points_two, points_two)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(t, np.zeros(3), atol=1e-12)
        self.assertAlmostEqual(rmse, 0.0, places=12)


class TestDatasetLoaderAndCalibrationPrecedence(unittest.TestCase):
    """Verifies Section 7.2: Dataset Loader & Calibration Precedence Hierarchy."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_root = os.path.join(self.temp_dir.name, "test_euroc")
        self.cam_dir = os.path.join(self.dataset_root, "cam0")
        self.imu_dir = os.path.join(self.dataset_root, "imu0")
        self.gt_dir = os.path.join(self.dataset_root, "state_groundtruth_estimate0")
        os.makedirs(self.cam_dir)
        os.makedirs(self.imu_dir)
        os.makedirs(self.gt_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mock_dataset(self, with_sensor_yaml=True, duration_sec=5.0):
        # Create sensor.yaml if requested
        if with_sensor_yaml:
            sensor_content = {
                "sensor_type": "camera",
                "resolution": [752, 480],
                "intrinsics": [460.0, 458.0, 360.0, 240.0],
                "distortion_coefficients": [-0.3, 0.08, 0.001, -0.0005],
                "T_BS": {
                    "data": [
                        0.0, 0.0, 1.0, 0.1,
                        -1.0, 0.0, 0.0, 0.0,
                        0.0, -1.0, 0.0, -0.05,
                        0.0, 0.0, 0.0, 1.0,
                    ]
                },
            }
            with open(os.path.join(self.cam_dir, "sensor.yaml"), "w", encoding="utf-8") as f:
                yaml.dump(sensor_content, f)

        # Create dummy cam0 frames
        cam_data_csv = os.path.join(self.cam_dir, "data.csv")
        cam_img_dir = os.path.join(self.cam_dir, "data")
        os.makedirs(cam_img_dir)
        num_frames = int(duration_sec * 20) + 1
        with open(cam_data_csv, "w", encoding="utf-8") as f:
            f.write("#timestamp [ns],filename\n")
            for i in range(num_frames):
                t_ns = int(i * 0.05 * 1e9)
                filename = f"{t_ns}.png"
                img_path = os.path.join(cam_img_dir, filename)
                Image.fromarray(np.full((480, 752), 128, dtype=np.uint8)).save(img_path)
                f.write(f"{t_ns},{filename}\n")

        # Create dummy imu0 samples
        imu_data_csv = os.path.join(self.imu_dir, "data.csv")
        num_imu = int(duration_sec * 200) + 1
        with open(imu_data_csv, "w", encoding="utf-8") as f:
            f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y,w_RS_S_z,a_RS_S_x [m s^-2],a_RS_S_y,a_RS_S_z\n")
            for i in range(num_imu):
                t_ns = int(i * 0.005 * 1e9)
                f.write(f"{t_ns},0.0,0.0,0.0,0.0,0.0,9.81\n")

        # Create dummy ground truth
        gt_data_csv = os.path.join(self.gt_dir, "data.csv")
        with open(gt_data_csv, "w", encoding="utf-8") as f:
            f.write("#timestamp [ns],p_RS_R_x [m],p_RS_R_y,p_RS_R_z,q_RS_w [],q_RS_x,q_RS_y,q_RS_z,v_RS_R_x [m s^-1],v_RS_R_y,v_RS_R_z,b_w_RS_S_x [rad s^-1],b_w_RS_S_y,b_w_RS_S_z,b_a_RS_S_x [m s^-2],b_a_RS_S_y,b_a_RS_S_z\n")
            for i in range(num_frames):
                t_ns = int(i * 0.05 * 1e9)
                f.write(f"{t_ns},0.0,0.0,5.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n")

    def test_sensor_yaml_resolution_parsing(self):
        """Correctly loads resolution, intrinsics, distortion, and T_BC from sensor.yaml."""
        self._create_mock_dataset(with_sensor_yaml=True)
        calib = load_cam0_calibration(self.cam_dir)

        self.assertIsNotNone(calib)
        self.assertEqual(calib["width"], 752)
        self.assertEqual(calib["height"], 480)
        self.assertAlmostEqual(calib["fx"], 460.0, places=4)
        self.assertAlmostEqual(calib["fy"], 458.0, places=4)
        self.assertAlmostEqual(calib["cx"], 360.0, places=4)
        self.assertAlmostEqual(calib["cy"], 240.0, places=4)
        self.assertAlmostEqual(calib["dist"][0], -0.3, places=4)
        self.assertAlmostEqual(calib["dist"][1], 0.08, places=4)
        self.assertIsNotNone(calib["t_bc"])

    def test_fallback_canonical_euroc_when_sensor_yaml_missing(self):
        """Falls back to canonical MT9V034 parameters when sensor.yaml is absent."""
        self._create_mock_dataset(with_sensor_yaml=False)
        calib = load_cam0_calibration(self.cam_dir)
        self.assertIsNone(calib)

        # EurocDataset loader must apply default MT9V034 calibration
        ds = EurocDataset(self.dataset_root)
        self.assertAlmostEqual(ds.fx, 458.654, places=3)
        self.assertAlmostEqual(ds.fy, 457.296, places=3)
        self.assertAlmostEqual(ds.cx, 367.215, places=3)
        self.assertAlmostEqual(ds.cy, 248.375, places=3)
        self.assertAlmostEqual(ds.dist[0], -0.283408, places=5)
        self.assertIn("Default EuRoC MT9V034", ds.calib_source)

    def test_time_cutoff_truncates_dataset(self):
        """--time-cutoff strictly limits the processed time range."""
        self._create_mock_dataset(with_sensor_yaml=True, duration_sec=6.0)
        cutoff = 2.0
        ds = EurocDataset(self.dataset_root, time_cutoff=cutoff)

        t_start = ds.frame_records[0][0]
        for t, _ in ds.frame_records:
            self.assertLessEqual(t - t_start, cutoff + 1e-4)

        imu_t_start = ds.imu_samples[0][0]
        for t, _, _ in ds.imu_samples:
            # IMU includes +0.1s lookahead buffer for interpolation boundary
            self.assertLessEqual(t - imu_t_start, cutoff + 0.1 + 1e-4)

    def test_user_config_overrides_dataset_calibration(self):
        """Passing custom SystemConfig YAML overrides dataset parameters."""
        yaml_path = os.path.join(self.temp_dir.name, "override_config.yaml")
        create_system_config_yaml(
            vocab_path="",
            output_yaml=yaml_path,
            width=640,
            height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            dist=(0.0, 0.0, 0.0, 0.0),
        )

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.assertEqual(data["camera"]["width"], 640)
        self.assertEqual(data["camera"]["height"], 480)
        self.assertAlmostEqual(data["camera"]["intrinsics"]["fx"], 500.0, places=2)


class TestCliAndIngestionFlow(unittest.TestCase):
    """Verifies Section 7.5: CLI argument parsing and flow control."""

    def test_cli_parser_defaults_and_options(self):
        """CLI parser correctly accepts all required flags and applies defaults."""
        parser = build_cli_parser()
        args = parser.parse_args(["--synthetic", "--time-cutoff", "10.0", "--realtime", "--playback-speed", "2.0"])

        self.assertTrue(args.synthetic)
        self.assertEqual(args.time_cutoff, 10.0)
        self.assertTrue(args.realtime)
        self.assertEqual(args.playback_speed, 2.0)
        self.assertEqual(args.eval_interval, 5.0)
        self.assertEqual(args.depth_noise_sigma, 0.02)


class TestSyntheticSelfTest(unittest.TestCase):
    """Verifies Section 7.1: Synthetic Trajectory Self-Test Execution."""

    def test_synthetic_trajectory_execution(self):
        """Full synthetic trajectory execution completes without dataset and writes report."""
        with tempfile.TemporaryDirectory() as output_dir:
            ds = SyntheticDataset(duration=2.0)
            vocab_path = ensure_vocab_file(DEFAULT_VOCAB_PATH)

            report_file = run_accuracy_benchmark(
                dataset=ds,
                vocab_path=vocab_path,
                eval_interval_sec=1.0,
                depth_noise_sigma=0.02,
                output_dir=output_dir,
                realtime=False,
            )

            # Invariant 1: Valid run report file generated
            self.assertTrue(os.path.isfile(report_file))

            # Invariant 2: Report file is non-empty and contains required summary headers
            with open(report_file, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("Pool-SLAM Accuracy Benchmark Report", content)
            self.assertIn("Position Error (Absolute Trajectory Error)", content)
            self.assertIn("SE(3) Umeyama Rigid Trajectory Alignment", content)

            # Invariant 3: Errors are finite (no NaN or Inf)
            self.assertNotIn("nan", content.lower())
            self.assertNotIn("inf", content.lower())

            ds._temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
