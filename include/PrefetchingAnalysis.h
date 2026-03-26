#pragma once

#include <cstdint>
#include <string>

namespace prefetching {

struct PrefetchCandidateInfo {
    std::string loadText;
    std::string pointerText;
    std::string scevText;
    unsigned loopDepth = 0;
    bool affine = false;
    bool loopVariant = false;
    bool isLoad = true;
    int64_t prefetchDistance = 0;
};

} // namespace prefetching