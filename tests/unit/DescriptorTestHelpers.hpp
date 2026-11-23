#pragma once

#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "descriptor/Qov.hpp"
#include "descriptor/BinaryBow.hpp"
#include "descriptor/QovBaseOffsets.hpp"

namespace PoolSLAM {
namespace descriptor {

class TestDescriptor : public ::testing::Test {
protected:
    static constexpr size_t VOCAB_TOTAL_NODES = BinaryBowTree::TOTAL_NODES;
    static constexpr size_t VOCAB_TREE_SIZE = BinaryBowTree::TREE_SIZE;
    static constexpr size_t VOCAB_WORDS = BinaryBowTree::VOCAB_SIZE;

    // Helper to construct a QovDescriptor from four 64-bit limbs
    static QovDescriptor makeDescriptor(uint64_t w0 = 0, uint64_t w1 = 0, uint64_t w2 = 0, uint64_t w3 = 0) {
        QovDescriptor des;
        des.bits = {w0, w1, w2, w3};
        return des;
    }

    // Helper to set a specific bit [0..255]
    static void setBit(QovDescriptor& des, size_t bitIdx) {
        if (bitIdx < QOV_BITS) {
            size_t word = bitIdx / 64;
            size_t b = bitIdx % 64;
            des.bits[word] |= (1ULL << b);
        }
    }

    // Helper to clear a specific bit [0..255]
    static void clearBit(QovDescriptor& des, size_t bitIdx) {
        if (bitIdx < QOV_BITS) {
            size_t word = bitIdx / 64;
            size_t b = bitIdx % 64;
            des.bits[word] &= ~(1ULL << b);
        }
    }

    // Helper to test a specific bit [0..255]
    static bool getBit(const QovDescriptor& des, size_t bitIdx) {
        if (bitIdx < QOV_BITS) {
            size_t word = bitIdx / 64;
            size_t b = bitIdx % 64;
            return (des.bits[word] & (1ULL << b)) != 0;
        }
        return false;
    }

    // Compare two QovCompPair arrays for equality
    static bool compareOffsets(const std::array<QovCompPair, QOV_BITS>& a,
                              const std::array<QovCompPair, QOV_BITS>& b) {
        for (size_t i = 0; i < QOV_BITS; ++i) {
            if (a[i].pt1 != b[i].pt1 || a[i].pt2 != b[i].pt2) {
                return false;
            }
        }
        return true;
    }

    // Helper to configure a mock BinaryBowTree with a deterministic path to targetWord
    static void setupDeterministicTree(BinaryBowTree& tree,
                                      uint32_t targetWord,
                                      const QovDescriptor& targetDes,
                                      const QovDescriptor& otherDes) {
        tree.tree.assign(VOCAB_TOTAL_NODES, otherDes);
        tree.vocabulary.resize(VOCAB_WORDS);
        for (size_t i = 0; i < VOCAB_WORDS; ++i) {
            tree.vocabulary[i].idf = 1.0f;
            tree.vocabulary[i].parent = static_cast<uint32_t>(i / 10 + 110);
        }

        uint32_t d0 = targetWord / 1000;
        uint32_t d1 = (targetWord / 100) % 10;
        uint32_t d2 = (targetWord / 10) % 10;
        uint32_t d3 = targetWord % 10;

        uint32_t b0 = d0;
        uint32_t b1 = 10 * b0 + 10 + d1;
        uint32_t b2 = 10 * b1 + 10 + d2;
        uint32_t b3 = 10 * b2 + 10 + d3;

        tree.tree[b0] = targetDes;
        tree.tree[b1] = targetDes;
        tree.tree[b2] = targetDes;
        tree.tree[b3] = targetDes;
    }

    // Writes a valid binary vocabulary file (1110 internal centroids + 10000 leaf word records)
    static bool writeVocabFile(const std::string& path,
                              const std::vector<QovDescriptor>& internalNodes,
                              const std::vector<QovDescriptor>& leafNodes,
                              const std::vector<float>& idfValues) {
        std::ofstream out(path, std::ios::binary);
        if (!out.is_open()) {
            return false;
        }

        for (const auto& node : internalNodes) {
            out.write(reinterpret_cast<const char*>(node.bits.data()), sizeof(node.bits));
        }

        for (size_t i = 0; i < leafNodes.size(); ++i) {
            out.write(reinterpret_cast<const char*>(leafNodes[i].bits.data()), sizeof(leafNodes[i].bits));
            float idf = idfValues[i];
            out.write(reinterpret_cast<const char*>(&idf), sizeof(float));
        }

        return out.good();
    }

    // Writes a truncated mock file
    static bool writeTruncatedFile(const std::string& path, size_t numBytes) {
        std::ofstream out(path, std::ios::binary);
        if (!out.is_open()) {
            return false;
        }
        std::vector<char> buffer(numBytes, 0x5A);
        out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        return out.good();
    }
};

} // namespace descriptor
} // namespace PoolSLAM
