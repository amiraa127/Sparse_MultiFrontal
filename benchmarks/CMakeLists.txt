# Directories to look for include files
include_directories(${SparseMultiFrontal_SOURCE_DIR}/include)
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${SCOTCH_INCLUDE_DIRS})
include_directories(${PASTIX_INCLUDE_DIRS})
include_directories(${HODLR_Path}/include/)
include_directories(${HODLR_Path}/IML/include/)


# Add HODLR library
set(HODLR_Path /Users/Amir/Documents/Uni\ Works/PhD/Research/Solver\ Codes/Dense_HODLR)
include_directories(${HODLR_Path}/include/)

# Directories to look for libraries
link_directories(${SparseMultiFrontal_BINARY_DIR}/src)

# Compile tests from test.cpp
add_executable(sparseBenchmarks benchmarks.cpp)

#find_package(BLAS REQUIRED)
#add_executable(bench EigenVsBLAS.cpp)

# test should link to solver libraries
target_link_libraries(sparseBenchmarks sparseSolver HODLRSolver  ${SCOTCH_LIBRARIES} ${CPPUNIT_LIBRARIES}  ${PASTIX_LIBRARIES})


# Tests to run
# add_test(BasicTests sparseTest)

