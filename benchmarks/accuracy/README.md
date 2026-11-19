# Pool-SLAM Accuracy Benchmark

The accuracy benchmark evaluates the state estimation performance of Pool-SLAM against reference 6-DoF ground truth trajectories from standard datasets (such as the EuRoC MAV `V1_01_easy` sequence).

---

## 1. Camera Calibration & Matrix $K$

The $3 \times 3$ camera intrinsic calibration matrix:

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

and the 4-parameter Brown-Conrady (`radtan`) distortion coefficients $[k_1, k_2, p_1, p_2]$ are resolved in the following priority:

1. **User-Specified Configuration (`--config <path.yaml>`):**
   If a custom `SystemConfig` YAML file is passed via `-c` / `--config`, all intrinsics, distortion coefficients, and camera-to-body extrinsics ($T_{BC}$) are loaded directly from that file.
2. **Auto-Detected from Dataset (`cam0/sensor.yaml`):**
   When running on an extracted EuRoC dataset, the harness reads `cam0/sensor.yaml` and extracts:
   - `resolution: [width, height]` $\rightarrow$ image dimensions (e.g., $752 \times 480$)
   - `intrinsics: [fu, fv, cu, cv]` $\rightarrow$ $f_x, f_y, c_x, c_y$
   - `distortion_coefficients: [k1, k2, p1, p2]` $\rightarrow$ Brown-Conrady coefficients
   - `T_BS`: $4 \times 4$ rigid transform $\rightarrow$ Camera-to-body extrinsics $T_{BC}$
3. **Canonical EuRoC Fallback:**
   If `sensor.yaml` is absent, the script falls back to canonical MT9V034 factory parameters:
   - $f_x = 458.654$, $f_y = 457.296$, $c_x = 367.215$, $c_y = 248.375$
   - $[k_1, k_2, p_1, p_2] = [-0.283408, 0.073959, 0.000194, 0.000018]$

---

## 2. CLI Arguments & `-h` Flag

Run `benchmark_accuracy.py -h` or `--help` to view built-in documentation:

```bash
python3 benchmarks/accuracy/benchmark_accuracy.py -h
```

### Options Reference

| Flag | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--dataset` | `-d` | `str` | `None` | Path to dataset directory containing `cam0` and `imu0` (e.g. `/data/V1_01_easy` or `/data/V1_01_easy/mav0`). |
| `--time-cutoff` | `-t` | `float` | `None` | Maximum dataset elapsed time in seconds to process (e.g. `60.0`). Defaults to full sequence. |
| `--config` | `-c` | `str` | `None` | Path to custom `SystemConfig` YAML file. Overrides automatic calibration parsing from `sensor.yaml`. |
| `--eval-interval` | | `float` | `5.0` | Interval in seconds of dataset time between state check evaluations. |
| `--depth-noise-sigma`| | `float` | `0.02` | Standard deviation in meters of zero-mean Gaussian noise added to ground-truth depth readings. |
| `--vocab` | | `str` | `benchmarks/accuracy/vocab.bin` | Path to binary Bag-of-Words vocabulary file. |
| `--output-dir` | | `str` | `benchmarks/accuracy/runs` | Directory where timestamped run reports (`.txt`) are stored. |
| `--realtime` | | `flag` | `False` | Throttles frame ingestion to match real-time dataset timestamp pacing. |
| `--playback-speed` | | `float` | `1.0` | Playback speed multiplier when `--realtime` is enabled (e.g. `2.0` for 2x speed). |
| `--synthetic` | | `flag` | `False` | Runs self-test using an internally generated synthetic circular trajectory (no dataset files required). |

---

## 3. Usage Examples

### Run on EuRoC `V1_01_easy`
```bash
# Evaluate first 60 seconds of dataset:
python3 benchmarks/accuracy/benchmark_accuracy.py \
    --dataset /path/to/V1_01_easy \
    --time-cutoff 60.0

# Evaluate with custom evaluation interval (every 3 seconds) and depth noise (0.01m):
python3 benchmarks/accuracy/benchmark_accuracy.py \
    --dataset /path/to/V1_01_easy \
    --eval-interval 3.0 \
    --depth-noise-sigma 0.01
```

### Self-Test Mode (No External Dataset)
```bash
python3 benchmarks/accuracy/benchmark_accuracy.py \
    --synthetic \
    --time-cutoff 10.0 \
    --eval-interval 2.5
```

---

## 4. Output Run Reports

Instead of flooding the terminal, the script prints single-line checkpoint updates to the console and generates a detailed report file at:

```
benchmarks/accuracy/runs/run_YYYYMMDD_HHMMSS.txt
```

Each run report includes:
- Run metadata, sequence duration, and calibration source
- Checkpoint telemetry: Estimated vs. Ground Truth positions $(x, y, z)$, orientations $(q_w, q_x, q_y, q_z)$, and velocities
- Absolute Trajectory Error (ATE) statistics (Mean, Median, Max, RMSE)
- Depth (vertical) error statistics
- Orientation error in degrees
- Umeyama SE(3) Rigidly Aligned Trajectory RMSE
