#include "LfTrackerTestHelpers.hpp"

namespace PoolSLAM {

// =============================================================================
// 1. Thread Lifecycle & State Machine
// =============================================================================

TEST_F(TestLfTracker, ThreadStartIdempotent) {
    auto tracker = createTracker();
    EXPECT_FALSE(tracker.isRunning());

    tracker.start();
    EXPECT_TRUE(tracker.isRunning());

    // Repeated call is safe no-op
    tracker.start();
    EXPECT_TRUE(tracker.isRunning());

    tracker.end();
    EXPECT_FALSE(tracker.isRunning());
}

TEST_F(TestLfTracker, ThreadEndHaltsSafely) {
    auto tracker = createTracker();
    tracker.start();
    EXPECT_TRUE(tracker.isRunning());

    tracker.end();
    EXPECT_FALSE(tracker.isRunning());

    // Repeated end call is safe
    tracker.end();
    EXPECT_FALSE(tracker.isRunning());
}

TEST_F(TestLfTracker, PassKeyframeRejectsWhenBusy) {
    auto tracker = createTracker();
    setBusy(tracker, true);

    std::vector<uint8_t> img(W * H, 128);
    std::vector<TrackedFeature<5>> features = {
        makeTrackedFeature(1, 100.0f, 100.0f)
    };

    EXPECT_FALSE(tracker.passKeyframe(1, 1.0, img.data(), features));
}

// =============================================================================
// 2. Keyframe Ingestion & Margin Pruning
// =============================================================================

TEST_F(TestLfTracker, PassKeyframeNullAndEmptyRejected) {
    auto tracker = createTracker();
    std::vector<uint8_t> img(W * H, 128);
    std::vector<TrackedFeature<5>> features = {
        makeTrackedFeature(1, 100.0f, 100.0f)
    };

    EXPECT_FALSE(tracker.passKeyframe(1, 1.0, nullptr, features));
    EXPECT_FALSE(tracker.passKeyframe(1, 1.0, img.data(), {}));
}

TEST_F(TestLfTracker, PassKeyframeMarginPruning) {
    auto tracker = createTracker();
    std::vector<uint8_t> img(W * H, 128);

    std::vector<TrackedFeature<5>> features = {
        makeTrackedFeature(1, 10.0f, 100.0f),                      // x < 15 -> pruned
        makeTrackedFeature(2, static_cast<float>(W - 10), 100.0f), // x >= W - 15 -> pruned
        makeTrackedFeature(3, 100.0f, 10.0f),                      // y < 15 -> pruned
        makeTrackedFeature(4, 100.0f, static_cast<float>(H - 10)), // y >= H - 15 -> pruned
        makeTrackedFeature(5, 100.0f, 100.0f)                      // Interior -> kept
    };

    ASSERT_TRUE(tracker.passKeyframe(1, 1.0, img.data(), features));

    const auto& kfs = getKeyframes(tracker);
    ASSERT_EQ(kfs.size(), 1);
    ASSERT_EQ(kfs[0].features.size(), 1);
    EXPECT_EQ(kfs[0].features[0].featureId, 5);
    EXPECT_EQ(kfs[0].features[0].pt.x(), 100);
    EXPECT_EQ(kfs[0].features[0].pt.y(), 100);
}

TEST_F(TestLfTracker, PassKeyframeAllFeaturesInMarginRejected) {
    auto tracker = createTracker();
    std::vector<uint8_t> img(W * H, 128);

    std::vector<TrackedFeature<5>> features = {
        makeTrackedFeature(1, 5.0f, 5.0f),
        makeTrackedFeature(2, 10.0f, 10.0f)
    };

    EXPECT_FALSE(tracker.passKeyframe(1, 1.0, img.data(), features));
    EXPECT_TRUE(getKeyframes(tracker).empty());
}

// =============================================================================
// 3. Landmark Updates (updateKeyframe)
// =============================================================================

TEST_F(TestLfTracker, UpdateKeyframeMatchesById) {
    auto tracker = createTracker();
    std::vector<uint8_t> img(W * H, 128);

    std::vector<TrackedFeature<5>> features = {
        makeTrackedFeature(101, 100.0f, 100.0f),
        makeTrackedFeature(102, 200.0f, 200.0f)
    };
    ASSERT_TRUE(tracker.passKeyframe(42, 1.0, img.data(), features));

    BackendKeyframeUpdate update;
    update.id = 42;
    update.pose = gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(1.0, 2.0, 3.0));
    update.landmarks = {
        {Eigen::Vector3d(1.0, 1.0, 5.0), 101},
        {Eigen::Vector3d(2.0, 2.0, 6.0), 102}
    };

    EXPECT_TRUE(tracker.updateKeyframe(update));

    const auto& kfs = getKeyframes(tracker);
    ASSERT_EQ(kfs.size(), 1);
    EXPECT_TRUE(kfs[0].pose.equals(update.pose));
    ASSERT_EQ(kfs[0].landmarks.size(), 2);
    EXPECT_EQ(kfs[0].landmarks[0].featureId, 101);
    EXPECT_DOUBLE_EQ(kfs[0].landmarks[0].pt3d.x(), 1.0);
    EXPECT_EQ(kfs[0].landmarks[1].featureId, 102);
    EXPECT_DOUBLE_EQ(kfs[0].landmarks[1].pt3d.x(), 2.0);
}

TEST_F(TestLfTracker, UpdateKeyframeNonexistentIdReturnsFalse) {
    auto tracker = createTracker();

    BackendKeyframeUpdate update;
    update.id = 9999;
    EXPECT_FALSE(tracker.updateKeyframe(update));
}

// =============================================================================
// 4. Image Filtering & Intensity Moments
// =============================================================================

TEST_F(TestLfTracker, GaussianBlurUniformImagePreserved) {
    auto tracker = createTracker();
    uint8_t* buf = getLocalBuffer(tracker);

    // Uniform image
    for (size_t i = 0; i < W * H; ++i) {
        buf[i] = 120;
    }

    callGaussianBlur(tracker);

    // Interior pixels should remain ~120
    for (size_t y = 10; y < H - 10; y += 50) {
        for (size_t x = 10; x < W - 10; x += 50) {
            EXPECT_NEAR(buf[y * W + x], 120, 2);
        }
    }
}

TEST_F(TestLfTracker, GaussianBlurSmoothsImpulse) {
    auto tracker = createTracker();
    uint8_t* buf = getLocalBuffer(tracker);

    for (size_t i = 0; i < W * H; ++i) {
        buf[i] = 0;
    }

    // Single high-intensity impulse at center
    size_t cx = 200;
    size_t cy = 200;
    buf[cy * W + cx] = 255;

    callGaussianBlur(tracker);

    // Center pixel should be attenuated from 255 to ~36 (36/256 of 255)
    EXPECT_LT(buf[cy * W + cx], 60);
    EXPECT_GT(buf[cy * W + cx], 20);

    // Neighbors should have received energy
    EXPECT_GT(buf[cy * W + (cx + 1)], 10);
    EXPECT_GT(buf[(cy + 1) * W + cx], 10);
}

TEST_F(TestLfTracker, OrientationEstimationFromGradients) {
    auto tracker = createTracker();
    uint8_t* buf = getLocalBuffer(tracker);

    // Draw horizontal gradient (intensity increases with X)
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            buf[y * W + x] = static_cast<uint8_t>(x % 256);
        }
    }

    // Ingest keyframe with one feature in gradient region
    std::vector<TrackedFeature<5>> features = {
        makeTrackedFeature(1, 100.0f, 100.0f)
    };
    ASSERT_TRUE(tracker.passKeyframe(1, 1.0, buf, features));

    // Calculate descriptors computes orientation
    callCalculateDescriptors(tracker);

    const auto& kfs = getKeyframes(tracker);
    ASSERT_EQ(kfs.size(), 1);
    ASSERT_EQ(kfs[0].features.size(), 1);

    // Descriptor bits should not be all-zeros
    bool hasNonZeroBits = false;
    for (uint64_t w : kfs[0].features[0].des.bits) {
        if (w != 0) hasNonZeroBits = true;
    }
    EXPECT_TRUE(hasNonZeroBits);
}

// =============================================================================
// 5. Feature Matching Filters (matchFeatures)
// =============================================================================

TEST_F(TestLfTracker, MatchFeaturesParentBranchRestriction) {
    auto tracker = createTracker();
    auto& tree = getVocabTree(tracker);
    tree.vocabulary.resize(10000);

    // Word 100 and Word 200 have different parents
    tree.vocabulary[100].parent = 50;
    tree.vocabulary[200].parent = 60;

    auto& kfs = getKeyframes(tracker);

    // Frame 0 (historical)
    kfs.emplace_back(1, 0, 0.0, 0);
    kfs[0].features.emplace_back(Eigen::Vector2i(100, 100), 1);
    kfs[0].features[0].wordId = 100;
    kfs[0].features[0].des.bits = {0xAAAAAAAAAAAAAAAAULL, 0, 0, 0};
    kfs[0].landmarks.push_back({Eigen::Vector3d(1.0, 1.0, 5.0), 1});

    // Frame 1 (current)
    kfs.emplace_back(1, 1, 1.0, 1);
    kfs[1].features.emplace_back(Eigen::Vector2i(100, 100), 2);
    kfs[1].features[0].wordId = 200; // Different parent
    kfs[1].features[0].des.bits = {0xAAAAAAAAAAAAAAAAULL, 0, 0, 0}; // Identical descriptor

    // Because parents differ, matchFeatures must reject
    size_t matches = callMatchFeatures(tracker, 1, 0);
    EXPECT_EQ(matches, 0);

    // Now set matching parent
    tree.vocabulary[200].parent = 50;
    matches = callMatchFeatures(tracker, 1, 0);
    EXPECT_EQ(matches, 1);
}

TEST_F(TestLfTracker, MatchFeaturesLowesRatioTest) {
    auto tracker = createTracker();
    auto& tree = getVocabTree(tracker);
    tree.vocabulary.resize(10000);
    tree.vocabulary[100].parent = 50;

    auto& kfs = getKeyframes(tracker);

    // Frame 0 (historical) with two features
    kfs.emplace_back(2, 0, 0.0, 0);
    kfs[0].features.emplace_back(Eigen::Vector2i(100, 100), 1);
    kfs[0].features[0].wordId = 100;
    kfs[0].features[0].des.bits = {0x0000000000000001ULL, 0, 0, 0}; // Dist ~1
    kfs[0].landmarks.push_back({Eigen::Vector3d(1.0, 1.0, 5.0), 1});

    kfs[0].features.emplace_back(Eigen::Vector2i(102, 100), 2);
    kfs[0].features[1].wordId = 100;
    kfs[0].features[1].des.bits = {0x0000000000000003ULL, 0, 0, 0}; // Dist ~2
    kfs[0].landmarks.push_back({Eigen::Vector3d(2.0, 1.0, 5.0), 2});

    // Frame 1 (current) with query feature at 0
    kfs.emplace_back(1, 1, 1.0, 1);
    kfs[1].features.emplace_back(Eigen::Vector2i(100, 100), 10);
    kfs[1].features[0].wordId = 100;
    kfs[1].features[0].des.bits = {0, 0, 0, 0};

    // d1 = 1, d2 = 2 -> ratio = 1/2 = 0.5 <= 0.80 -> Accepted
    size_t matches = callMatchFeatures(tracker, 1, 0);
    EXPECT_EQ(matches, 1);

    // Ambiguous case: change second feature to dist ~1
    kfs[0].features[1].des.bits = {0x0000000000000001ULL, 0, 0, 0}; // d1 = 1, d2 = 1 -> ratio = 1.0 > 0.80
    matches = callMatchFeatures(tracker, 1, 0);
    EXPECT_EQ(matches, 0);
}

// =============================================================================
// 6. Match Retrieval Protocol (getMatch)
// =============================================================================

TEST_F(TestLfTracker, GetMatchReturnsFalseWhenNoData) {
    auto tracker = createTracker();
    BackendLfPacket packet;

    EXPECT_FALSE(tracker.getMatch(packet));
}

TEST_F(TestLfTracker, GetMatchTransfersPacketAndClearsDataReady) {
    auto tracker = createTracker();

    BackendLfPacket staged;
    staged.currentFrameId = 50;
    staged.matchedFrameId = 10;
    staged.timestamp = 2.5;
    staged.inlierCount = 35;
    staged.inlierRatio = 0.75f;
    staged.matchScore = 0.42f;
    staged.relativePose = gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.5, 0.0, 1.0));

    setStagedMatch(tracker, staged);
    EXPECT_TRUE(getDataReady(tracker));

    BackendLfPacket retrieved;
    ASSERT_TRUE(tracker.getMatch(retrieved));

    EXPECT_EQ(retrieved.currentFrameId, 50);
    EXPECT_EQ(retrieved.matchedFrameId, 10);
    EXPECT_DOUBLE_EQ(retrieved.timestamp, 2.5);
    EXPECT_EQ(retrieved.inlierCount, 35);
    EXPECT_FLOAT_EQ(retrieved.inlierRatio, 0.75f);
    EXPECT_FLOAT_EQ(retrieved.matchScore, 0.42f);
    EXPECT_TRUE(retrieved.relativePose.equals(staged.relativePose));

    // dataReady must now be cleared and stagedMatch reset
    EXPECT_FALSE(getDataReady(tracker));
    EXPECT_FALSE(tracker.getMatch(retrieved));
}

} // namespace PoolSLAM
