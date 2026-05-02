cmake_minimum_required(VERSION 3.16)
project(GPUOptimizationPasses LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM include dirs: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM definitions: ${LLVM_DEFINITIONS}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(PrefetchPass MODULE
    src/PrefetchPass.cpp
)

set_target_properties(PrefetchPass PROPERTIES
    COMPILE_FLAGS "-fno-rtti"
)

if(APPLE)
  set_target_properties(CoalescingPass PROPERTIES
      LINK_FLAGS "-undefined dynamic_lookup"
  )

  set_target_properties(PrefetchPass PROPERTIES
      LINK_FLAGS "-undefined dynamic_lookup"
  )
endif()
