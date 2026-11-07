#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>

#include "descriptor/BinaryBow.hpp"
#include "descriptor/Qov.hpp"

namespace PoolSLAM {
namespace descriptor {

// ============================================================================
// 1. Hamming Distance Tests (fastHamming)
// ============================================================================

// Verifies that fastHamming evaluates identity, inverted bitmasks, and arbitrary
// bit differences correctly across the full 256-bit representation.
TEST(QovDescriptorTest, fastHammingDistanceCalculations) {
    QovDescriptor d1;
    QovDescriptor d2;

    // Zero distance on identity
    EXPECT_EQ(fastHamming(d1, d2), 0);

    // Exact full inversion across 256 bits
    d2.bits.fill(~0ULL);
    EXPECT_EQ(fastHamming(d1, d2), 256);

    // Single bit flip in each 64-bit segment
    d1.bits = {1ULL, 1ULL << 15, 1ULL << 31, 1ULL << 63};
    d2.bits = {0ULL, 0ULL, 0ULL, 0ULL};
    EXPECT_EQ(fastHamming(d1, d2), 4);

    // Symmetry check: d(A, B) == d(B, A)
    EXPECT_EQ(fastHamming(d1, d2), fastHamming(d2, d1));
}

// ============================================================================
// 2. Rotated Pattern Lookup Tests (RotatedQovPairs)
// ============================================================================

// Verifies that RotatedQovPairs correctly wraps negative and out-of-range angles,
// and produces identical offset tables for equivalent degree and radian inputs.
TEST(QovDescriptorTest, rotatedPairsAngleWrappingAndEquivalence) {
    constexpr size_t kImageStride = 640;
    RotatedQovPairs pattern(kImageStride);

    // Exact degree and radian equivalence at 90 degrees
    constexpr double kPi = 3.14159265358979323846;
    const auto &deg90 = pattern.offsetAtAngleDeg(90.0);
    const auto &rad90 = pattern.offsetAtAngleRad(kPi / 2.0);

    for (size_t i = 0; i < QOV_BITS; i++) {
        EXPECT_EQ(deg90[i].pt1, rad90[i].pt1);
        EXPECT_EQ(deg90[i].pt2, rad90[i].pt2);
    }

    // Normalization check: 0 deg, 360 deg, and -360 deg map to the same bin
    const auto &deg0 = pattern.offsetAtAngleDeg(0.0);
    const auto &deg360 = pattern.offsetAtAngleDeg(360.0);
    const auto &degNeg360 = pattern.offsetAtAngleDeg(-360.0);

    for (size_t i = 0; i < QOV_BITS; i++) {
        EXPECT_EQ(deg0[i].pt1, deg360[i].pt1);
        EXPECT_EQ(deg0[i].pt2, deg360[i].pt2);
        EXPECT_EQ(deg0[i].pt1, degNeg360[i].pt1);
        EXPECT_EQ(deg0[i].pt2, degNeg360[i].pt2);
    }
}

// ============================================================================
// 3. Bit Generation Tests (generateDescriptor)
// ============================================================================

// Verifies that generateDescriptor evaluates pixel comparisons (I(pt1) < I(pt2))
// and populates the 64-bit word bitfields without index shifting errors.
TEST(QovDescriptorTest, generateDescriptorBitEvaluation) {
    // Synthetic 1D pixel patch
    // Base coordinate at center index 10: pt1 will look left (< 10), pt2 will look right (> 10)
    std::array<uint8_t, 32> mockPatch;
    mockPatch.fill(50);
    mockPatch[5] = 20;  // Dark
    mockPatch[15] = 80; // Bright

    // Offsets relative to mockPatch[10]: pt1 -> -5 (index 5), pt2 -> +5 (index 15)
    std::array<QovCompPair, QOV_BITS> mockOffsets;
    for (size_t i = 0; i < QOV_BITS; i++) {
        mockOffsets[i].pt1 = -5;
        mockOffsets[i].pt2 = 5;
    }

    // I(pt1) < I(pt2) is true (20 < 80) for all pairs -> all 256 bits must be 1
    const uint8_t *center = &mockPatch[10];
    QovDescriptor des = generateDescriptor(center, mockOffsets);

    for (size_t w = 0; w < 4; w++) {
        EXPECT_EQ(des.bits[w], ~0ULL);
    }

    // Invert condition: pt1 -> +5 (80), pt2 -> -5 (20) -> all bits must be 0
    for (size_t i = 0; i < QOV_BITS; i++) {
        mockOffsets[i].pt1 = 5;
        mockOffsets[i].pt2 = -5;
    }

    des = generateDescriptor(center, mockOffsets);
    for (size_t w = 0; w < 4; w++) {
        EXPECT_EQ(des.bits[w], 0ULL);
    }
}

// ============================================================================
// 4. Inverted Index Database Tests (BowDatabase)
// ============================================================================

// Verifies that BowDatabase accumulates TF-IDF weights when the same word occurs
// multiple times in a keyframe, appends separate entries for new keyframes, and clears properly.
TEST(BinaryBowTest, databaseAccumulationAndLifecycle) {
    constexpr size_t kVocabSize = 100;
    BowDatabase db(kVocabSize);

    constexpr uint32_t kWordId = 42;
    constexpr uint32_t kKeyframe0 = 10;
    constexpr uint32_t kKeyframe1 = 11;

    // First observation in keyframe 10
    db.addWord(kWordId, kKeyframe0, 1.5f);
    ASSERT_EQ(db.getEntries(kWordId).size(), 1u);
    EXPECT_EQ(db.getEntries(kWordId)[0].keyframeId, kKeyframe0);
    EXPECT_FLOAT_EQ(db.getEntries(kWordId)[0].weight, 1.5f);

    // Second observation in keyframe 10 -> accumulates weight
    db.addWord(kWordId, kKeyframe0, 0.5f);
    ASSERT_EQ(db.getEntries(kWordId).size(), 1u);
    EXPECT_FLOAT_EQ(db.getEntries(kWordId)[0].weight, 2.0f);

    // New observation in keyframe 11 -> appends new entry
    db.addWord(kWordId, kKeyframe1, 1.2f);
    ASSERT_EQ(db.getEntries(kWordId).size(), 2u);
    EXPECT_EQ(db.getEntries(kWordId)[1].keyframeId, kKeyframe1);
    EXPECT_FLOAT_EQ(db.getEntries(kWordId)[1].weight, 1.2f);

    // Clear operation resets lists without deallocating outer table capacity
    db.clear();
    EXPECT_TRUE(db.getEntries(kWordId).empty());
    EXPECT_EQ(db.invertedIndex.size(), kVocabSize);
}

// ============================================================================
// 5. Tree Traversal Routing Tests (BinaryBowTree)
// ============================================================================

// Verifies that associatedWord traverses the multi-level tree hierarchy deterministically
// toward the closest centroid match at each branch level.
TEST(BinaryBowTest, associatedWordRoutesToClosestCentroid) {
    BinaryBowTree bowTree;
    bowTree.tree.resize(BinaryBowTree::TOTAL_NODES);

    // Initialize all centroids to ~0ULL (256 set bits)
    for (size_t i = 0; i < BinaryBowTree::TOTAL_NODES; i++) {
        bowTree.tree[i].bits.fill(~0ULL);
    }

    // Set branch 0 at each level along the path to 0ULL (all zero bits)
    // Level 0: Node 0
    bowTree.tree[0].bits.fill(0ULL);
    // Level 1: First child of Node 0 -> Node 10
    bowTree.tree[10].bits.fill(0ULL);
    // Level 2: First child of Node 10 -> Node 110
    bowTree.tree[110].bits.fill(0ULL);
    // Level 3 (Leaves): First child of Node 110 -> Node 1110 (which corresponds to leaf word 0)
    bowTree.tree[1110].bits.fill(0ULL);

    // Query descriptor with all zero bits should route down the 0-centroid path to word 0
    QovDescriptor queryDes;
    queryDes.bits.fill(0ULL);

    const uint32_t wordId = bowTree.associatedWord(queryDes);
    EXPECT_EQ(wordId, 0u);
}

} // namespace descriptor
} // namespace PoolSLAM