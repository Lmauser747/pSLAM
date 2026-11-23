#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>

#include "descriptor/Qov.hpp"
#ifndef FORCE_INLINE
#if defined(__GNUC__) || defined(__clang__)
  #define FORCE_INLINE inline __attribute__((always_inline))
#else
  #define FORCE_INLINE inline
#endif
#endif

// Implement a Bag of Words tree using the binary features described in Qov.hpp
namespace PoolSLAM{
namespace descriptor{

class BowDatabase;

// Compact 8-byte visual word metadata (POD / Trivially Copyable)
struct alignas(8) WordData{
    float idf = 0.0f; // Stores the inverse document frequency of the word
    uint32_t parent = 0; // Parent node in the tree
};

// 8-byte inverted index entry
struct IndexEntry{
    uint32_t keyframeId = 0; // Keyframe index
    float weight = 0.0f; // TF-IDF weight for this keyframe
};

constexpr size_t iPowCt(size_t base, int pwr){
    size_t acc = 1;
    for(int i = 0; i < pwr; i++){
        acc *= base;
    }
    return acc;
}

class BinaryBowTree{
public:
    static constexpr size_t NUM_BRANCHES = 10; // Number of BoW tree branches
    static constexpr size_t TREE_DEPTH = 4; // Number of levels in BoW tree, not including root
    static constexpr size_t TREE_SIZE = NUM_BRANCHES * (iPowCt(NUM_BRANCHES, TREE_DEPTH - 1) - 1) / (NUM_BRANCHES - 1); // Total internal branch entries in tree (1110)
    static constexpr size_t VOCAB_SIZE = iPowCt(NUM_BRANCHES, TREE_DEPTH); // Number of words in the vocabulary (10000)
    static constexpr size_t NUM_PARENTS = iPowCt(NUM_BRANCHES, TREE_DEPTH - 1); // Number of parent nodes at level (TREE_DEPTH - 1) (1000)
    static constexpr size_t TOTAL_NODES = TREE_SIZE + VOCAB_SIZE; // Total entries across branches and leaves (11110)

    std::vector<QovDescriptor> tree; // Flat array storing all internal branch and leaf centroids
    std::vector<WordData> vocabulary; // Flat array storing compact vocabulary word metadata

    // Read in the vocabulary from the file storing it
    bool readVocabFile(const std::string &vocabPath);

    // Find the word associated with the given descriptor
    uint32_t associatedWord(const QovDescriptor &des) const;

    BinaryBowTree() = default;

    explicit BinaryBowTree(const std::string &vocabPath){
        if(!vocabPath.empty()){
            if(!readVocabFile(vocabPath)){
                throw std::runtime_error("BinaryBowTree: Failed to read vocabulary file: " + vocabPath);
            }
        }
    }

    // Matches descriptor to word and updates the database. Returns the word ID
    uint32_t matchWord(const QovDescriptor& des, float frequency, uint32_t keyframe, BowDatabase& db) const;
};

// Standalone inverted index database for place recognition
class BowDatabase{
public:
    std::vector<std::vector<IndexEntry>> invertedIndex;

    BowDatabase(){
        invertedIndex.resize(BinaryBowTree::VOCAB_SIZE);
    }

    explicit BowDatabase(size_t vocabSize){
        invertedIndex.resize(vocabSize);
    }

    void addWord(uint32_t wordId, uint32_t keyframeId, float weight){
        if(wordId >= invertedIndex.size()){
            return;
        }
        auto &entries = invertedIndex[wordId];
        if(!entries.empty() && entries.back().keyframeId == keyframeId){
            entries.back().weight += weight;
        }else{
            entries.push_back({keyframeId, weight});
        }
    }

    void clear(){
        for(size_t i = 0; i < invertedIndex.size(); i++){
            invertedIndex[i].clear();
        }
    }

    FORCE_INLINE const std::vector<IndexEntry>& getEntries(uint32_t wordId) const{
        return invertedIndex[wordId];
    }
};

}

} 