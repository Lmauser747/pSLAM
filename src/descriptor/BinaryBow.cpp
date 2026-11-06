#include <string>
#include <fstream>
#include <cstddef>
#include <ios>

#include "BinaryBow.hpp"
#include "CoreTypes.hpp"
#include "descriptor/Qov.hpp"

namespace PoolSLAM{

namespace descriptor{

const WordData &BinaryBowTree::matchWord(const QovDescriptor& des, float frequency, size_t keyframe){
    size_t bestBranch = 0;
    int bestScore = fastHamming(tree[0], des);
    for(size_t branch = 1; branch < NUM_BRANCHES; branch++){
        auto score = fastHamming(tree[branch], des);
        if(score < bestScore){
            bestBranch = branch;
            bestScore = score;
        }
    }

    for(size_t level = 1; level < TREE_DEPTH - 1; level++){
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
    auto word = NUM_BRANCHES * bestBranch + NUM_BRANCHES - TREE_SIZE;
    if(!vocabulary[word].images.empty() && vocabulary[word].images.back().first == keyframe){
        vocabulary[word].images.back().second += frequency * vocabulary[word].idf;
    }else{
        vocabulary[word].images.emplace_back(keyframe, frequency);
    }
    return vocabulary[word];
}

bool BinaryBowTree::readVocabFile(const std::string &vocabPath){
    std::fstream stream(vocabPath, std::ios::binary);
    if(!stream){
        return false;
    }
    // Read in the tree
    for(auto &node : tree){
        stream.read(reinterpret_cast<char *>(&(node.bits)), sizeof(node.bits));
    }

    // Read in the words
    size_t wordId = 0;
    for(auto &word : vocabulary){
        stream.read(reinterpret_cast<char *>(&(word.des.bits)), sizeof(word.des.bits));
        stream.read(reinterpret_cast<char*>(&word.idf), sizeof(word.idf));
        word.id = wordId;
        word.parent = (wordId / TREE_DEPTH) + (TREE_SIZE - NUM_PARENTS); 
        wordId++;
    }

    return true;
}

}

}