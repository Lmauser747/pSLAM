#include "TestDescriptorHelpers.hpp"

namespace PoolSLAM {
namespace descriptor {

// =============================================================================
// 1. Fast Hamming Distance Metric Axioms
// =============================================================================

TEST_F(TestDescriptor, FastHammingIdentity) {
    auto zero = makeDescriptor(0, 0, 0, 0);
    EXPECT_EQ(fastHamming(zero, zero), 0);

    auto ones = makeDescriptor(~0ULL, ~0ULL, ~0ULL, ~0ULL);
    EXPECT_EQ(fastHamming(ones, ones), 0);

    auto arbitrary = makeDescriptor(0xDEADBEEFCAFEBABEULL, 0x0123456789ABCDEFULL,
                                    0xFEDCBA9876543210ULL, 0x5555555555555555ULL);
    EXPECT_EQ(fastHamming(arbitrary, arbitrary), 0);
}

TEST_F(TestDescriptor, FastHammingMaxDistance) {
    auto zero = makeDescriptor(0, 0, 0, 0);
    auto ones = makeDescriptor(~0ULL, ~0ULL, ~0ULL, ~0ULL);
    EXPECT_EQ(fastHamming(zero, ones), 256);

    auto patA = makeDescriptor(0xAAAAAAAAAAAAAAAAULL, 0xAAAAAAAAAAAAAAAAULL,
                               0xAAAAAAAAAAAAAAAAULL, 0xAAAAAAAAAAAAAAAAULL);
    auto patB = makeDescriptor(0x5555555555555555ULL, 0x5555555555555555ULL,
                               0x5555555555555555ULL, 0x5555555555555555ULL);
    EXPECT_EQ(fastHamming(patA, patB), 256);
}

TEST_F(TestDescriptor, FastHammingLimbBoundaries) {
    auto base = makeDescriptor(0, 0, 0, 0);
    const size_t boundaryBits[] = {0, 63, 64, 127, 128, 191, 192, 255};

    for (size_t bit : boundaryBits) {
        auto flipped = base;
        setBit(flipped, bit);
        EXPECT_EQ(fastHamming(base, flipped), 1);
    }
}

TEST_F(TestDescriptor, FastHammingMultiBitDiff) {
    auto des1 = makeDescriptor(0, 0, 0, 0);
    auto des2 = des1;

    setBit(des2, 5);
    setBit(des2, 63);
    setBit(des2, 70);
    setBit(des2, 130);
    setBit(des2, 200);
    EXPECT_EQ(fastHamming(des1, des2), 5);
}

TEST_F(TestDescriptor, FastHammingSymmetry) {
    auto desA = makeDescriptor(0x1234567890ABCDEFULL, 0xFEDCBA0987654321ULL,
                               0xCAFEBABE11223344ULL, 0xA5A5A5A55A5A5A5AULL);
    auto desB = makeDescriptor(0x00FF00FF00FF00FFULL, 0xFF00FF00FF00FF00ULL,
                               0x1111111111111111ULL, 0xEEEEEEEEEEEEEEEEULL);

    EXPECT_EQ(fastHamming(desA, desB), fastHamming(desB, desA));
}

TEST_F(TestDescriptor, FastHammingTriangleInequality) {
    auto a = makeDescriptor(0xAAAAAAAAAAAAAAAAULL, 0x0ULL, 0x0ULL, 0x0ULL);
    auto b = makeDescriptor(0xAAAAAAAA55555555ULL, 0x0ULL, 0x0ULL, 0x0ULL);
    auto c = makeDescriptor(0x5555555555555555ULL, 0x0ULL, 0x0ULL, 0x0ULL);

    EXPECT_LE(fastHamming(a, c), fastHamming(a, b) + fastHamming(b, c));
}

TEST_F(TestDescriptor, EqualityOperators) {
    auto a = makeDescriptor(1, 2, 3, 4);
    auto b = makeDescriptor(1, 2, 3, 4);
    auto c = makeDescriptor(1, 2, 3, 5);

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    EXPECT_FALSE(a == c);
    EXPECT_TRUE(a != c);
}

// =============================================================================
// 2. Rotated Pattern Geometry & Angular Quantization
// =============================================================================

TEST_F(TestDescriptor, AngleEquivalenceDegRad) {
    constexpr double PI = 3.14159265358979323846;
    RotatedQovPairs rot(640);

    const double anglesDeg[] = {0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0};
    for (double deg : anglesDeg) {
        double rad = (deg / 180.0) * PI;
        EXPECT_TRUE(compareOffsets(rot.offsetAtAngleDeg(deg), rot.offsetAtAngleRad(rad)));
    }
}

TEST_F(TestDescriptor, PeriodicityAndNegativeAngles) {
    constexpr double PI = 3.14159265358979323846;
    RotatedQovPairs rot(640);

    EXPECT_TRUE(compareOffsets(rot.offsetAtAngleDeg(0.0), rot.offsetAtAngleDeg(360.0)));
    EXPECT_TRUE(compareOffsets(rot.offsetAtAngleDeg(0.0), rot.offsetAtAngleDeg(-360.0)));
    EXPECT_TRUE(compareOffsets(rot.offsetAtAngleDeg(0.0), rot.offsetAtAngleDeg(720.0)));

    EXPECT_TRUE(compareOffsets(rot.offsetAtAngleDeg(-90.0), rot.offsetAtAngleDeg(270.0)));
    EXPECT_TRUE(compareOffsets(rot.offsetAtAngleRad(-PI / 2.0), rot.offsetAtAngleRad(1.5 * PI)));
}

TEST_F(TestDescriptor, HalfBinRounding) {
    RotatedQovPairs rot(640);

    // Bin 0 spans [-5.625, 5.625) -> [0, 5.62) maps to bin 0
    EXPECT_TRUE(compareOffsets(rot.offsetAtAngleDeg(0.0), rot.offsetAtAngleDeg(5.0)));

    // Bin 1 center is 11.25 -> [5.63, 16.87) maps to bin 1
    EXPECT_TRUE(compareOffsets(rot.offsetAtAngleDeg(11.25), rot.offsetAtAngleDeg(6.0)));
    EXPECT_TRUE(compareOffsets(rot.offsetAtAngleDeg(11.25), rot.offsetAtAngleDeg(16.8)));

    // Bin 2 center is 22.5 -> 17.0 maps to bin 2
    EXPECT_TRUE(compareOffsets(rot.offsetAtAngleDeg(22.5), rot.offsetAtAngleDeg(17.0)));
}

TEST_F(TestDescriptor, RowStrideScaling) {
    constexpr size_t stride = 640;
    RotatedQovPairs rot(stride);

    const auto& offsets = rot.offsetAtAngleDeg(0.0);
    for (size_t i = 0; i < QOV_BITS; ++i) {
        ptrdiff_t expectedPt1 = QOV_BASE_OFFSETS[i * 4 + 0] + static_cast<ptrdiff_t>(stride) * QOV_BASE_OFFSETS[i * 4 + 1];
        ptrdiff_t expectedPt2 = QOV_BASE_OFFSETS[i * 4 + 2] + static_cast<ptrdiff_t>(stride) * QOV_BASE_OFFSETS[i * 4 + 3];

        EXPECT_EQ(offsets[i].pt1, expectedPt1);
        EXPECT_EQ(offsets[i].pt2, expectedPt2);
    }
}

// =============================================================================
// 3. Bit Evaluation & Descriptor Synthesis
// =============================================================================

TEST_F(TestDescriptor, UniformPatchAllBitsSet) {
    std::vector<uint8_t> buffer(512, 100);
    const uint8_t* pt = buffer.data() + 256;

    buffer[256 - 1] = 50;  // pt1
    buffer[256 + 1] = 150; // pt2

    std::array<QovCompPair, QOV_BITS> offsets;
    for (size_t i = 0; i < QOV_BITS; ++i) {
        offsets[i].pt1 = -1;
        offsets[i].pt2 = 1;
    }

    auto des = generateDescriptor(pt, offsets);
    for (size_t w = 0; w < 4; ++w) {
        EXPECT_EQ(des.bits[w], ~0ULL);
    }
}

TEST_F(TestDescriptor, UniformPatchAllBitsCleared) {
    std::vector<uint8_t> buffer(512, 100);
    const uint8_t* pt = buffer.data() + 256;

    buffer[256 - 1] = 150; // pt1
    buffer[256 + 1] = 50;  // pt2

    std::array<QovCompPair, QOV_BITS> offsets;
    for (size_t i = 0; i < QOV_BITS; ++i) {
        offsets[i].pt1 = -1;
        offsets[i].pt2 = 1;
    }

    auto des = generateDescriptor(pt, offsets);
    for (size_t w = 0; w < 4; ++w) {
        EXPECT_EQ(des.bits[w], 0ULL);
    }
}

TEST_F(TestDescriptor, EqualIntensitiesAllBitsCleared) {
    std::vector<uint8_t> buffer(512, 100);
    const uint8_t* pt = buffer.data() + 256;

    std::array<QovCompPair, QOV_BITS> offsets;
    for (size_t i = 0; i < QOV_BITS; ++i) {
        offsets[i].pt1 = 0;
        offsets[i].pt2 = 0;
    }

    auto des = generateDescriptor(pt, offsets);
    for (size_t w = 0; w < 4; ++w) {
        EXPECT_EQ(des.bits[w], 0ULL);
    }
}

TEST_F(TestDescriptor, AlternatingBitPattern) {
    std::vector<uint8_t> buffer(512, 100);
    const uint8_t* pt = buffer.data() + 256;

    buffer[256 - 1] = 50;
    buffer[256 + 1] = 150;

    std::array<QovCompPair, QOV_BITS> offsets;
    for (size_t i = 0; i < QOV_BITS; ++i) {
        if (i % 2 == 0) {
            offsets[i].pt1 = -1; // 50 < 150 -> bit = 1
            offsets[i].pt2 = 1;
        } else {
            offsets[i].pt1 = 1;  // 150 < 50 -> bit = 0
            offsets[i].pt2 = -1;
        }
    }

    auto des = generateDescriptor(pt, offsets);
    for (size_t w = 0; w < 4; ++w) {
        EXPECT_EQ(des.bits[w], 0x5555555555555555ULL);
    }
}

TEST_F(TestDescriptor, WordLimbBoundaries) {
    std::vector<uint8_t> buffer(512, 100);
    const uint8_t* pt = buffer.data() + 256;

    buffer[256 - 1] = 50;
    buffer[256 + 1] = 150;

    const size_t testPairs[] = {63, 64, 127, 128, 191, 192, 255};
    for (size_t pairIdx : testPairs) {
        std::array<QovCompPair, QOV_BITS> offsets;
        for (size_t i = 0; i < QOV_BITS; ++i) {
            offsets[i].pt1 = 0;
            offsets[i].pt2 = 0;
        }
        offsets[pairIdx].pt1 = -1;
        offsets[pairIdx].pt2 = 1;

        auto des = generateDescriptor(pt, offsets);

        size_t expectedWord = pairIdx / 64;
        size_t expectedBit = pairIdx % 64;

        for (size_t w = 0; w < 4; ++w) {
            if (w == expectedWord) {
                EXPECT_EQ(des.bits[w], (1ULL << expectedBit));
            } else {
                EXPECT_EQ(des.bits[w], 0ULL);
            }
        }
    }
}

// =============================================================================
// 4. Inverted Index Database (BowDatabase)
// =============================================================================

TEST_F(TestDescriptor, DatabaseAccumulateSameKeyframe) {
    BowDatabase db(10000);

    db.addWord(42, 10, 0.5f);
    db.addWord(42, 10, 0.3f);

    const auto& entries = db.getEntries(42);
    ASSERT_EQ(entries.size(), 1);
    EXPECT_EQ(entries[0].keyframeId, 10);
    EXPECT_FLOAT_EQ(entries[0].weight, 0.8f);
}

TEST_F(TestDescriptor, DatabasePartitionDistinctKeyframes) {
    BowDatabase db(10000);

    db.addWord(42, 10, 0.5f);
    db.addWord(42, 11, 0.7f);
    db.addWord(42, 12, 0.2f);

    const auto& entries = db.getEntries(42);
    ASSERT_EQ(entries.size(), 3);
    EXPECT_EQ(entries[0].keyframeId, 10);
    EXPECT_FLOAT_EQ(entries[0].weight, 0.5f);
    EXPECT_EQ(entries[1].keyframeId, 11);
    EXPECT_FLOAT_EQ(entries[1].weight, 0.7f);
    EXPECT_EQ(entries[2].keyframeId, 12);
    EXPECT_FLOAT_EQ(entries[2].weight, 0.2f);
}

TEST_F(TestDescriptor, DatabaseOutOfBoundsSafelyIgnored) {
    BowDatabase db(100);

    EXPECT_NO_THROW(db.addWord(100, 1, 1.0f));
    EXPECT_NO_THROW(db.addWord(99999, 1, 1.0f));
}

TEST_F(TestDescriptor, DatabaseClearResetsEntriesPreservesCapacity) {
    BowDatabase db(10000);

    db.addWord(42, 10, 0.5f);
    db.addWord(100, 10, 1.2f);
    EXPECT_FALSE(db.getEntries(42).empty());
    EXPECT_FALSE(db.getEntries(100).empty());

    db.clear();

    EXPECT_EQ(db.invertedIndex.size(), 10000);
    EXPECT_TRUE(db.getEntries(42).empty());
    EXPECT_TRUE(db.getEntries(100).empty());
}

TEST_F(TestDescriptor, DatabaseCustomVocabSizeConstructor) {
    BowDatabase db(500);
    EXPECT_EQ(db.invertedIndex.size(), 500);
    db.addWord(499, 1, 0.5f);
    EXPECT_EQ(db.getEntries(499).size(), 1);
}

// =============================================================================
// 5. BoW Tree Traversal & Routing Determinism
// =============================================================================

TEST_F(TestDescriptor, TreeDeterministicRouting) {
    BinaryBowTree tree;
    auto targetDes = makeDescriptor(0, 0, 0, 0);
    auto otherDes = makeDescriptor(~0ULL, ~0ULL, ~0ULL, ~0ULL);

    const uint32_t testWords[] = {0, 42, 1337, 7245, 9999};
    for (uint32_t targetWord : testWords) {
        setupDeterministicTree(tree, targetWord, targetDes, otherDes);
        EXPECT_EQ(tree.associatedWord(targetDes), targetWord);
    }
}

TEST_F(TestDescriptor, TreeAssociatedWordUninitializedTree) {
    BinaryBowTree tree;
    auto des = makeDescriptor(0, 0, 0, 0);
    EXPECT_EQ(tree.associatedWord(des), 0);

    tree.tree.resize(100);
    EXPECT_EQ(tree.associatedWord(des), 0);
}

TEST_F(TestDescriptor, TreeMatchWordAccumulatesToDatabase) {
    BinaryBowTree tree;
    auto targetDes = makeDescriptor(0, 0, 0, 0);
    auto otherDes = makeDescriptor(~0ULL, ~0ULL, ~0ULL, ~0ULL);

    uint32_t targetWord = 4567;
    setupDeterministicTree(tree, targetWord, targetDes, otherDes);
    tree.vocabulary[targetWord].idf = 2.5f;

    BowDatabase db(10000);
    uint32_t matchedWord = tree.matchWord(targetDes, 3.0f, 101, db);

    EXPECT_EQ(matchedWord, targetWord);

    const auto& entries = db.getEntries(targetWord);
    ASSERT_EQ(entries.size(), 1);
    EXPECT_EQ(entries[0].keyframeId, 101);
    EXPECT_FLOAT_EQ(entries[0].weight, 7.5f);
}

TEST_F(TestDescriptor, TreeMatchWordGuardsUninitialized) {
    BinaryBowTree tree;
    BowDatabase db(10000);
    auto des = makeDescriptor(0, 0, 0, 0);

    EXPECT_EQ(tree.matchWord(des, 1.0f, 1, db), 0);
}

// =============================================================================
// 6. Vocabulary File Serialization
// =============================================================================

TEST_F(TestDescriptor, RoundTripFileRead) {
    const std::string tmpPath = "/tmp/test_vocab_valid.bin";
    std::remove(tmpPath.c_str());

    std::vector<QovDescriptor> internalNodes(VOCAB_TREE_SIZE);
    for (size_t i = 0; i < VOCAB_TREE_SIZE; ++i) {
        internalNodes[i] = makeDescriptor(i, i + 1, i + 2, i + 3);
    }

    std::vector<QovDescriptor> leafNodes(VOCAB_WORDS);
    std::vector<float> idfValues(VOCAB_WORDS);
    for (size_t i = 0; i < VOCAB_WORDS; ++i) {
        leafNodes[i] = makeDescriptor(i + 10000, i + 10001, i + 10002, i + 10003);
        idfValues[i] = 1.0f + static_cast<float>(i) * 0.001f;
    }

    ASSERT_TRUE(writeVocabFile(tmpPath, internalNodes, leafNodes, idfValues));

    BinaryBowTree tree;
    ASSERT_TRUE(tree.readVocabFile(tmpPath));

    ASSERT_EQ(tree.tree.size(), VOCAB_TOTAL_NODES);
    ASSERT_EQ(tree.vocabulary.size(), VOCAB_WORDS);

    // Verify first and last internal nodes
    EXPECT_EQ(tree.tree[0], internalNodes[0]);
    EXPECT_EQ(tree.tree[VOCAB_TREE_SIZE - 1], internalNodes.back());

    // Verify first and last leaf nodes
    EXPECT_EQ(tree.tree[VOCAB_TREE_SIZE], leafNodes[0]);
    EXPECT_EQ(tree.tree[VOCAB_TOTAL_NODES - 1], leafNodes.back());

    // Verify vocabulary metadata & parent index formula
    for (size_t i : {0UL, 42UL, 500UL, 9999UL}) {
        EXPECT_FLOAT_EQ(tree.vocabulary[i].idf, idfValues[i]);
        uint32_t expectedParent = static_cast<uint32_t>(i / 10 + 110);
        EXPECT_EQ(tree.vocabulary[i].parent, expectedParent);
    }

    std::remove(tmpPath.c_str());
}

TEST_F(TestDescriptor, TruncatedFileReturnsFalse) {
    const std::string tmpPath = "/tmp/test_vocab_truncated.bin";
    std::remove(tmpPath.c_str());

    ASSERT_TRUE(writeTruncatedFile(tmpPath, 1000)); // Far less than 395,520 bytes

    BinaryBowTree tree;
    EXPECT_FALSE(tree.readVocabFile(tmpPath));

    std::remove(tmpPath.c_str());
}

TEST_F(TestDescriptor, NonexistentFileReturnsFalse) {
    BinaryBowTree tree;
    EXPECT_FALSE(tree.readVocabFile("/tmp/nonexistent_vocab_file_98765.bin"));
}

TEST_F(TestDescriptor, ParameterizedConstructorThrowsOnNonexistent) {
    EXPECT_THROW(BinaryBowTree("/tmp/nonexistent_vocab_file_98765.bin"), std::runtime_error);
}

TEST_F(TestDescriptor, BenchmarkVocabLoadsSuccessfully) {
    const std::vector<std::string> candidatePaths = {
        "benchmarks/accuracy/vocab.bin",
        "../benchmarks/accuracy/vocab.bin",
        "../../benchmarks/accuracy/vocab.bin",
        "../../../benchmarks/accuracy/vocab.bin"
    };
    std::string vocabPath;
    for (const auto& path : candidatePaths) {
        std::ifstream f(path);
        if (f.good()) {
            vocabPath = path;
            break;
        }
    }
    ASSERT_FALSE(vocabPath.empty()) << "Could not locate benchmarks/accuracy/vocab.bin";

    BinaryBowTree tree;

    // Load actual repo vocabulary binary
    ASSERT_TRUE(tree.readVocabFile(vocabPath));
    EXPECT_EQ(tree.tree.size(), VOCAB_TOTAL_NODES);
    EXPECT_EQ(tree.vocabulary.size(), VOCAB_WORDS);

    // Check reasonable IDF values
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_GT(tree.vocabulary[i].idf, 0.0f);
        uint32_t expectedParent = static_cast<uint32_t>(i / 10 + 110);
        EXPECT_EQ(tree.vocabulary[i].parent, expectedParent);
    }
}

} // namespace descriptor
} // namespace PoolSLAM
