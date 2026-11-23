#include <string>
#include <fstream>
#include <cstddef>
#include <cstdint>
#include <ios>

#include "descriptor/BinaryBow.hpp"
#include "descriptor/Qov.hpp"

namespace PoolSLAM{
namespace descriptor{

uint32_t BinaryBowTree::associatedWord(const QovDescriptor &des) const{
    if(tree.size() < TOTAL_NODES){
        return 0;
    }
    size_t bestBranch = 0;
    int bestScore = fastHamming(tree[0], des);
    // BEGIN VECTORIZE
    for(size_t branch = 1; branch < NUM_BRANCHES; branch++){
        auto score = fastHamming(tree[branch], des);
        if(score < bestScore){
            bestBranch = branch;
            bestScore = score;
        }
    }

    for(size_t level = 1; level < TREE_DEPTH; level++){
        auto firstBranch = NUM_BRANCHES * bestBranch + NUM_BRANCHES;
        bestBranch = firstBranch;
        bestScore = fastHamming(tree[firstBranch], des);
        for(size_t branch = firstBranch + 1; branch < firstBranch + NUM_BRANCHES; branch++){
            auto score = fastHamming(tree[branch], des);
            if(score < bestScore){
                bestBranch = branch;
                bestScore = score;
            }
        }
    }
    // END VECTORIZE
    return static_cast<uint32_t>(bestBranch - TREE_SIZE);
}

uint32_t BinaryBowTree::matchWord(const QovDescriptor& des, float frequency, uint32_t keyframe, BowDatabase& db) const{
    if(tree.size() < TOTAL_NODES || vocabulary.size() < VOCAB_SIZE){
        return 0;
    }
    uint32_t wordId = associatedWord(des);
    db.addWord(wordId, keyframe, frequency * vocabulary[wordId].idf);
    return wordId;
}

bool BinaryBowTree::readVocabFile(const std::string &vocabPath){
    std::ifstream stream(vocabPath, std::ios::binary);
    if(!stream){
        return false;
    }

    tree.resize(TOTAL_NODES);
    vocabulary.resize(VOCAB_SIZE);

    // Read in internal branch nodes (levels 1 to TREE_DEPTH - 1)
    for(size_t i = 0; i < TREE_SIZE; i++){
        stream.read(reinterpret_cast<char *>(tree[i].bits.data()), sizeof(tree[i].bits));
    }

    // Read in vocabulary leaf words (level TREE_DEPTH)
    for(size_t wordId = 0; wordId < VOCAB_SIZE; wordId++){
        stream.read(reinterpret_cast<char *>(tree[TREE_SIZE + wordId].bits.data()), sizeof(tree[TREE_SIZE + wordId].bits));
        stream.read(reinterpret_cast<char *>(&(vocabulary[wordId].idf)), sizeof(vocabulary[wordId].idf));
        vocabulary[wordId].parent = static_cast<uint32_t>((wordId / NUM_BRANCHES) + (TREE_SIZE - NUM_PARENTS));
    }

    return stream.good();
}

}
}