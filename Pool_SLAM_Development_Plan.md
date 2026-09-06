# Development and Testing Plan: Open Source ROV SLAM Framework (Pool-SLAM)

## 1. Introduction
This document provides a phased development roadmap and a progressive testing plan for the Open Source ROV SLAM Framework. It breaks down the architectural blueprint into actionable development milestones, ensuring that the hardware-agnostic, semi-autonomous Visual-Inertial-Depth Odometry (VIDO) system is built robustly and efficiently.

---

## 2. Phased Development Plan

### Phase 1: Core Infrastructure and API Foundation (Layer 4)
**Objective:** Establish the foundational build system, dependencies, and core data structures.
*   **Tasks:**
    *   Set up the C++17 environment and CMake build configurations.
    *   Integrate Eigen for high-performance matrix and vector operations.
    *   Define core data structures for asynchronous sensor ingestion (Images, 6-axis IMU, 1D Depth, 2D Feature Arrays).
    *   Define the output struct for the `getCurrentPose` method (6-DoF state: position, Roll/Pitch/Yaw, and confidence metrics).
    *   Implement initial Pybind11 wrappers to ensure Python and C++ environments are aligned from the start.

### Phase 2: Hardware Abstraction Layer (Layer 1 - HAL)
**Objective:** Implement standard interfaces for disparate hardware and ensure data synchronization.
*   **Tasks:**
    *   Develop `ImageStream`, `FeatureStream`, `IMUStream`, and `DepthStream` ingestion endpoints.
    *   Implement the `Synchronizer` module.
    *   Develop time-based interpolation algorithms to align asynchronous sensor frequencies into temporally coherent packets.
    *   Establish thread-safe queues for passing data from the HAL to the SLAM engine.

### Phase 3: Offline Calibration Engine (Layer 2)
**Objective:** Create the pre-mission calibration suite.
*   **Tasks:**
    *   Implement the calibration suite entirely in Python, utilizing NumPy and SciPy for all mathematical calibration routines. OpenCV is strictly prohibited across the entire project to minimize dependencies and overhead.
    *   **Intrinsics Module:** Implement algorithms to calculate focal length, optical center, and underwater distortion models using custom routines based on NumPy/SciPy.
    *   **IMU Parameter Module:** Develop routines to parse stationary data logs and compute bias instabilities (random walk) and noise densities.
    *   **Extrinsics Module:** Implement spatial translation and rotation matrix solvers to map camera and depth sensors to the IMU coordinate frame.
    *   Output results to a standardized YAML/JSON configuration file.

### Phase 4: Core SLAM Engine - Front-End Tracking (Layer 3)
**Objective:** Implement the real-time perception and tracking loop.
*   **Tasks:**
    *   Develop dynamic pipeline routing (bypassing internal extraction if `FeatureStream` is utilized via smart cameras).
    *   Implement hardware-accelerated, custom feature detection and tracking (using ARM NEON SIMD instructions). 
    *   Develop tracking logic across sequential frames (optical flow or descriptor matching).
    *   Implement IMU pre-integration to provide high-frequency, continuous 6-DoF pose predictions between visual frames.

### Phase 5: Core SLAM Engine - Back-End Optimization (Layer 3)
**Objective:** Refine trajectories and map points using a non-linear optimization backend.
*   **Tasks:**
    *   Integrate GTSAM as the sole non-linear optimization backend to handle the factor graph formulation and bundle adjustment.
    *   Construct the optimization factor graph incorporating:
        *   Visual reprojection errors.
        *   IMU pre-integration constraints.
        *   1D depth constraints (from pressure transducers).
    *   Execute bundle adjustment to simultaneously optimize the ROV trajectory and 3D map points.
    *   **Map Management:** Implement marginalization logic to drop older poses/points, maintaining a bounded computational footprint while preserving historical constraints.

### Phase 6: API Finalization and Python Bindings (Layer 4)
**Objective:** Finalize the user-facing integration endpoints.
*   **Tasks:**
    *   Complete `push` methods for dynamic, asynchronous data injection.
    *   Implement initialization routines that parse the Phase 3 YAML/JSON calibration files.
    *   Finalize and stabilize the Pybind11 Python wrappers.
    *   Audit the framework to ensure the 15-30 Hz performance requirement is met on constrained edge-compute platforms.

---

## 3. Progressive Testing Plan

To ensure mathematical correctness, system stability, and real-time performance, testing will be executed progressively through five distinct stages.

### Stage 1: Component-Level Unit Testing
*   **Focus:** Mathematical accuracy and core logic isolation.
*   **Methods:**
    *   Test Eigen operations for spatial transformations.
    *   Test the `Synchronizer` with mock asynchronous timestamps to verify correct temporal interpolation.
    *   Test the Offline Calibration engine using standard datasets with known ground-truth intrinsics/extrinsics.

### Stage 2: Subsystem Integration Testing
*   **Focus:** Inter-layer communication and concurrency.
*   **Methods:**
    *   Push simulated sensor streams into the HAL and verify thread-safe queue behavior into the Front-End.
    *   Verify dynamic configuration routing (switching between raw images and pre-computed smart camera features).
    *   Ensure Pybind11 bindings map perfectly to C++ methods without memory leaks.

### Stage 3: Dataset and Simulation Validation
*   **Focus:** Algorithm robustness on ideal and noisy data.
*   **Methods:**
    *   Run the framework against standard Visual-Inertial datasets (e.g., EuRoC, TartanAir).
    *   Evaluate bundle adjustment and factor graph integrity.
    *   Inject simulated 1D depth data into the datasets to verify depth constraints prevent Z-axis drift.

### Stage 4: Hardware-in-the-Loop (HIL) Profiling
*   **Focus:** Real-time operational viability and resource constraints.
*   **Methods:**
    *   Deploy the framework onto target hardware (Raspberry Pi 5, Jetson Nano).
    *   Profile CPU/Memory usage.
    *   Verify the core tracking and optimization loops maintain the strict 15-30 Hz requirement.
    *   Validate the ARM NEON SIMD optimizations for raw image feature extraction.

### Stage 5: In-Water Field Trials
*   **Focus:** Real-world underwater conditions.
*   **Methods:**
    *   Deploy on a fully assembled ROV in a controlled pool environment, followed by open-water testing.
    *   Evaluate tracking resilience against underwater visual artifacts (particulates, light scattering, low contrast).
    *   Compare the output of the `getCurrentPose` API against established baseline navigation systems or known physical waypoints.