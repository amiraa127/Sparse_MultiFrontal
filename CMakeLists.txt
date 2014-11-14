cmake_minimum_required(VERSION 2.8)
project(SparseMultiFrontal)
set (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# Enable test running via 'make test' command
enable_testing()

set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++11 -O3 -Wall -Wextra")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

# Find the dependencies.
find_package (Eigen REQUIRED)
find_package (CppUnit REQUIRED)
set(SCOTCH_TEST_RUNS "FALSE")
#set(SCOTCH_DEBUG "TRUE")

find_package (SCOTCH REQUIRED)

if (EIGEN_FOUND)
    message ("-- Found Eigen version ${EIGEN_VERSION}: ${EIGEN_INCLUDE_DIRS}")
endif (EIGEN_FOUND)

if (CPPUNIT_FOUND)
    message ("-- Found CppUnit version ${CPPUNIT_VERSION}: ${CPPUNIT_INCLUDE_DIRS}")
endif (CPPUNIT_FOUND)

if (SCOTCH_FOUND)
    message ("-- Found SCOTCH version ${SCOTCH_VERSION}: ${SCOTCH_INCLUDE_DIRS}")
endif (SCOTCH_FOUND)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(benchmarks)

file(COPY test/data DESTINATION ${SparseMultiFrontal_BINARY_DIR}/test)
file(COPY benchmarks/data DESTINATION ${SparseMultiFrontal_BINARY_DIR}/benchmarks)
