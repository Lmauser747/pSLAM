# Vision Algorithms for Mobile Robotics: Integrated Learning Plan

## Course Objective
To develop a comprehensive understanding of monocular SLAM and visual odometry by pairing the practical curriculum of the University of Zurich (UZH) with the rigorous geometric foundations of Hartley and Zisserman (HZ).

---

## Module 1: Foundations and Camera Models
**Goal:** Understand the projective pipeline from 3D world coordinates to 2D pixel coordinates.

### Topics
*   **UZH Lecture 01 & 02:** Introduction to Visual Odometry, Image Formation, Perspective Projection.
*   **HZ Chapter 1:** Introduction to ubiquitous projective geometry.
*   **HZ Chapter 6:** Finite cameras and the projective camera model.

### Study Tasks
*   Define the intrinsic and extrinsic parameters of a pinhole camera.
*   Derive the projection matrix $P = K[R|t]$.

---

## Module 2: Camera Calibration and Pose Estimation
**Goal:** Learn how to determine the camera matrix from known 3D-2D correspondences.

### Topics
*   **UZH Lecture 03:** Camera Calibration & the Perspective-n-Point (PnP) problem.
*   **HZ Chapter 7 (7.1, 7.2):** Computation of the Camera Matrix P, basic equations, and geometric error.
*   **HZ Section 22.1:** Camera resectioning.

### Study Tasks
*   Implement the Direct Linear Transformation (DLT) for camera calibration.
*   Minimize geometric reprojection error using non-linear optimization.

---

## Module 3: Front-End Vision (Features and Tracking)
**Goal:** Extract repeatable landmarks and track them across frames.
*Note: This module relies entirely on UZH materials and external computer vision resources, as HZ focuses on geometric relations rather than image processing.*

### Topics
*   **UZH Lecture 04:** Filtering & Edge Detection.
*   **UZH Lecture 05:** Point Feature Detectors (Harris).
*   **UZH Lecture 06:** Point Feature Detectors and Descriptors (SIFT, SURF, BRIEF, BRISK).
*   **UZH Lecture 11:** Optical Flow and Tracking (Lucas-Kanade).

### Study Tasks
*   Implement a Harris corner detector from scratch.
*   Compare descriptor matching (Brute Force vs. FLANN) against Lucas-Kanade feature tracking.

---

## Module 4: Two-View Geometry and Initialization
**Goal:** Estimate camera motion between two frames and triangulate 3D landmarks without prior knowledge of the scene.

### Topics
*   **UZH Lecture 07:** Stereo vision, rectification, and triangulation.
*   **UZH Lecture 08:** The Eight-Point Algorithm.
*   **UZH Lecture 09:** P3P algorithm and RANSAC.
*   **HZ Chapter 9:** Epipolar Geometry and the Fundamental Matrix.
*   **HZ Chapter 11 (11.1, 11.2, 11.12):** Normalized 8-point algorithm, image rectification.
*   **HZ Chapter 12 (12.1, 12.2):** Structure Computation and linear triangulation.
*   **HZ Section 4.7:** Robust estimation (RANSAC).

### Study Tasks
*   Implement the normalized 8-point algorithm to compute the Fundamental Matrix.
*   Extract the Essential Matrix (using known intrinsics) and decompose it into rotation and translation.
*   Write a RANSAC wrapper to filter outlier feature matches.
*   Triangulate inlier matches to instantiate a 3D point cloud.

---

## Module 5: Back-End Optimization (Bundle Adjustment)
**Goal:** Jointly refine camera poses and 3D map points to minimize global reprojection error.

### Topics
*   **UZH Lecture 13:** Visual-Inertial Fusion & Bundle Adjustment.
*   **HZ Section 18.1:** Projective reconstruction and bundle adjustment.
*   **HZ Appendix 5:** Least-squares Minimization.

### Study Tasks
*   Formulate the bundle adjustment problem as a non-linear least-squares optimization.
*   Understand the sparsity structure of the Jacobian matrix in visual odometry.
*   Implement a toy bundle adjuster using Gauss-Newton or Levenberg-Marquardt (or utilize a library like Ceres Solver/g2o).

---

## Module 6: Advanced Topics in Autonomous Navigation
**Goal:** Explore modern augmentations to the standard SLAM pipeline.

### Topics
*   **UZH Lecture 10:** Dense 3D Reconstruction.
*   **UZH Lecture 12:** Place Recognition and Deep Learning for visual navigation.
*   **UZH Lecture 14:** Event-based Vision.

### Study Tasks
*   Investigate loop closure detection using Bag-of-Words (BoW).
*   Understand how event cameras differ from standard frame-based cameras in high-speed robotics.

## Resources
*   **Textbook(HZ):** Multiple-View Geometry in Computer Vision by Hartley and Zisserman
*   **Course(UZH):** Vision Algorithms for Mobile Robotics at UZH. Online [here](https://rpg.ifi.uzh.ch/teaching.html#VAMR)
