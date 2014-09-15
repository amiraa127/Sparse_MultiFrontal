cmake_minimum_required(VERSION 2.8)
project(SparseMultiFrontal)

# Enable test running via 'make test' command
enable_testing()

set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++0x -O3 -Wall")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(benchmarks)

file(COPY test/data DESTINATION ${SparseMultiFrontal_BINARY_DIR}/test)
file(COPY benchmarks/data DESTINATION ${SparseMultiFrontal_BINARY_DIR}/benchmarks)
