# Open Source ROV SLAM Framework: Requirements and Blueprint

## 1. Overview
This document outlines the requirements and architectural blueprint for an open-source, semi-autonomous Simultaneous Localization and Mapping (SLAM) framework designed specifically for Remotely Operated Vehicles (ROVs). The framework abstracts complex perception and mathematical operations to allow teams to implement monocular Visual-Inertial-Depth Odometry (VIDO) without requiring deep domain expertise.

## 2. System Requirements

### 2.1. Functional Requirements
*   **Agnostic Sensor Ingestion:** The framework must expose standardized interfaces to accept data from disparate hardware configurations, specifically:
    *   Asynchronous monocular image frames.
    *   6-axis IMU readings (acceleration and angular velocity).
    *   1D depth readings (e.g., from a pressure transducer).
    *   Pre-computed 2D feature arrays (for smart cameras like OAK-D or RealSense).
*   **Visual-Inertial-Depth Odometry (VIDO):** The system must continuously estimate the ROV's 6-Degree-of-Freedom (6-DoF) pose relative to an initialization point.
*   **Dynamic Pipeline Configuration:** The system must allow users to configure the perception pipeline at runtime. This includes the ability to bypass internal feature extraction if a smart camera is utilized, or select varying feature extraction parameters based on available compute.
*   **Offline Calibration Suite:** A guided, mathematical abstraction layer must be provided for calculating camera intrinsics (pinhole model and underwater distortion), IMU noise parameters (random walk and noise density), and spatial extrinsics (rigid body transformations between sensors).
*   **Simplified Output Delivery:** SLAM outputs (pose, velocity, map points) must be exposed via simplified, standardized data structures.

### 2.2. Non-Functional Requirements
*   **Performance and Real-Time Operation:** The core tracking and optimization loop must maintain a minimum of 15-30 Hz on constrained compute platforms (e.g., Raspberry Pi 5, Jetson Nano).
*   **Hardware Agnosticism:** The architecture must not assume a specific brand or model of camera, IMU, or depth sensor. 
*   **Cross-Language Accessibility:** The framework must be accessible to teams developing in either C++ or Python.
*   **Concurrency:** Front-end tracking and back-end optimization must run asynchronously via thread-safe queues to prevent processing bottlenecks.

## 3. Languages and Dependencies

To balance high-performance execution with minimal dependency overhead, the framework will utilize the following stack:

*   **Core Language:** C++17 (compiled via CMake) for deterministic memory management, multithreading, and bare-metal execution speed.
*   **Bindings:** Pybind11 to expose the C++ API to Python environments.
*   **Linear Algebra:** Eigen for high-performance matrix and vector operations.
*   **Optimization Backend:** GTSAM to handle the factor graph formulation and bundle adjustment.
*   **Computer Vision (Real-Time):** Custom, hardware-accelerated implementations (e.g., utilizing ARM NEON SIMD instructions) for feature extraction and tracking. OpenCV is strictly prohibited anywhere in this project to reduce overhead and dependencies. 
*   **Computer Vision (Offline Calibration):** The Offline Calibration Engine will be implemented entirely in Python, utilizing mathematical calibration routines written with NumPy and SciPy. 

### 3.1. Top-Level Dependencies
*   **C++17 Compiler:** Core execution.
*   **CMake:** Build system configuration.
*   **Pybind11:** C++ to Python API binding generation.
*   **Eigen:** High-performance linear algebra and matrix operations.
*   **GTSAM:** Non-linear optimization backend.
*   **Python 3.x:** Execution environment for the offline calibration engine.
*   **NumPy:** Array and matrix processing for offline calibration.
*   **SciPy:** Advanced optimization and numerical routines for offline calibration.

## 4. Architectural Blueprint

The framework follows a modular, multi-layered architecture designed to isolate hardware ingestion from core mathematical operations.

### 4.1. Layer 1: Hardware Abstraction Layer (HAL)
This layer normalizes incoming sensor data into framework-compatible structures and handles synchronization.
*   **ImageStream:** Ingests raw image matrices for host-side feature extraction.
*   **FeatureStream:** Ingests external arrays of 2D keypoints/descriptors from edge-compute cameras.
*   **IMUStream / DepthStream:** Ingests timestamped inertial and depth data.
*   **Synchronizer:** Aligns asynchronous sensor frequencies using time-based interpolation, ensuring temporally coherent data packets are passed to the tracking engine.

### 4.2. Layer 2: Offline Calibration Engine
A standalone suite executed prior to deployment. Outputs a standardized configuration file (YAML/JSON). Operations are handled in Python via NumPy/SciPy.
*   **Intrinsics Module:** Calculates focal length, optical center, and distortion using standard target patterns.
*   **IMU Parameter Module:** Computes bias instabilities and noise densities from stationary data logs.
*   **Extrinsics Module:** Solves for the spatial translation and rotation matrices mapping the camera and depth sensor to the IMU coordinate frame.

### 4.3. Layer 3: Core SLAM Engine
The real-time processing core, operating via concurrent threads.
*   **Front-End (Tracking Module):**
    *   Routes data based on configuration (ingesting raw images vs. pre-computed features).
    *   If raw images are ingested, executes SIMD-optimized feature detection.
    *   Tracks features across sequential frames to establish optical flow or descriptor matches.
    *   Integrates IMU data between frames to provide a high-frequency pose prediction.
*   **Back-End (Optimization Module):**
    *   Constructs a factor graph.
    *   Incorporates visual reprojection errors, IMU pre-integration constraints, and 1D depth constraints.
    *   Executes bundle adjustment to simultaneously refine the trajectory and the 3D map points.
*   **Map Management Module:**
    *   Implements marginalization to drop old poses and map points, bounding computational complexity while preserving historical trajectory constraints.

### 4.4. Layer 4: API and User Interface
The integration endpoint for user code, exposing simplified access methods.
*   Provides initialization routines that parse the calibration/configuration file.
*   Exposes `push` methods for injecting asynchronous sensor data into the HAL.
*   Exposes a `getCurrentPose` method returning a clean, easily interpretable 6-DoF state struct containing position, orientation (Roll/Pitch/Yaw), and confidence metrics.
