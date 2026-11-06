#pragma once

#include <cstddef>
#include <cstdio>
#include <vector>
#include <string>
#include <cstdint>
#include <utility>
#include <bitset>

#include "descriptor/Qov.hpp"
#include "simd/Utils.hpp"

// Implement a Bag of Words tree using the binary features described in Qov.hpp

namespace PoolSLAM{

namespace descriptor{

struct WordData{
    size_t id; // word id
    size_t parent; // parent node
    QovDescriptor des; // Descriptor associated with the word
    float idf = 0; // Stores the inverse document frequency of the word
    std::vector<std::pair<size_t, float>> images; // Stores frame index in keyframes and associated TF-IDF score
};

namespace{

constexpr size_t iPowCt(size_t base, int pwr){
    size_t acc = 1;
    for(int i = 0; i < pwr; i++){
        acc *= base;
    }
    return acc;
}

}

// Find the hamming distance between two descriptors
FORCE_INLINE int fastHamming(const QovDescriptor &w1, const QovDescriptor &w2); 

class BinaryBowTree{
public:
    static constexpr size_t NUM_BRANCHES = 10; // Number of BoW tree branches
    static constexpr size_t TREE_DEPTH = 4; // Number of levels in BoW tree, not including root
    static constexpr size_t TREE_SIZE = NUM_BRANCHES * (iPowCt(NUM_BRANCHES, TREE_DEPTH - 1) - 1)  / (NUM_BRANCHES - 1); // Total entries in tree
    static constexpr size_t VOCAB_SIZE = iPowCt(NUM_BRANCHES, TREE_DEPTH ); // Number of words in the vocabulary
    static constexpr size_t NUM_PARENTS = iPowCt(NUM_BRANCHES, TREE_DEPTH - 1); 

    std::vector<QovDescriptor> tree; // Tree to match features to words
    std::vector<WordData> vocabulary; // Tree to store vocabulary

    // Read in the vocabulary from the file storing it
    bool readVocabFile(const std::string &vocabPath);

    BinaryBowTree(const std::string &vocabPath){
        tree.reserve(TREE_SIZE);
        if(!readVocabFile(vocabPath)){
            return;
        }
    }

    // Matches word to the frame. Returns the word index
    const WordData &matchWord(const QovDescriptor& des, float frequency, size_t keyframe);

    // Gets the word at the given index
    FORCE_INLINE const WordData getWord(size_t idx) const;

    // Gets the parent node of the leaf
    FORCE_INLINE size_t getParent(size_t leaf) const;
};

FORCE_INLINE int fastHamming(const QovDescriptor &w1, const QovDescriptor &w2){
    return __builtin_popcountll(w1.bits[0] ^ w2.bits[0]) +
           __builtin_popcountll(w1.bits[1] ^ w2.bits[1]) +
           __builtin_popcountll(w1.bits[2] ^ w2.bits[2]) +
           __builtin_popcountll(w1.bits[3] ^ w2.bits[3]);
} 

FORCE_INLINE const WordData BinaryBowTree::getWord(size_t idx) const{
    return vocabulary[idx];
}

}

}