#pragma once

#include <cstdint>
#include <string>
#include <set>

namespace llvm { class LoadInst; class Loop; }

namespace prefetching {

// struct PrefetchCandidateInfo {
//     std::string loadText;
//     std::string pointerText;
//     std::string scevText;
//     unsigned loopDepth = 0;
//     bool affine = false;
//     bool loopVariant = false;
//     bool isLoad = true;
//     int64_t prefetchDistance = 0;
// };

enum class PrefetchBenefit {
    High,    // stride <= cache line (64B): one prefetch covers multiple iterations
    Medium,  // stride <= 4x cache line (256B): hardware prefetcher likely misses this
    Low,     // stride > 256B: large stride, each access lands in a new region
    Unknown, // stride not statically known
};

enum class BoundsStatus {
    Safe,              // static trip count proves prefetch stays in bounds
    Unsafe,            // static trip count proves prefetch goes out of bounds
    NeedsRuntimeGuard, // trip count unknown at compile time
    NotApplicable,     // candidate=no, bounds irrelevant
};

struct PrefetchCandidateInfo {
    std::string loadText;
    std::string pointerText;   // SSA pointer, e.g. %42
    std::string baseText;      // normalized base, e.g. A or arg0
    std::string scevText;
    int loopDepth = 0;
    bool affine = false;
    bool loopVariant = false;
    int64_t prefetchDistance = 0;
    BoundsStatus boundsStatus = BoundsStatus::NotApplicable;
    int64_t strideBytes = 0;         // 0 = unknown; extracted from AddRec step
    PrefetchBenefit benefit = PrefetchBenefit::Unknown;
    llvm::LoadInst *loadInst = nullptr; // owning load; used by the transform phase
    llvm::Loop     *loop     = nullptr; // enclosing loop; used to find the bound
};

} // namespace prefetching