#pragma once

#include <string>
#include <optional>

namespace coalescing {

struct AccessSummary {
    bool threadDependent = false;
    int64_t stride = 0;
    int64_t offset = 0;
    int64_t elementSizeBytes = 0;
    int64_t estimatedTransactions = 0;
    std::string classification;
    std::string prettyIndex;
};

std::string classifyAccess(bool threadDependent, int64_t stride, int64_t estimatedTransactions);
int64_t estimateTransactionsPerWarp(int64_t stride, int64_t elementSizeBytes, int64_t warpSize = 32, int64_t segmentSizeBytes = 128);

} // namespace coalescing
