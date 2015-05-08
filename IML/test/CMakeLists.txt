# Directories to look for include files
include_directories(${HODLR_Path}/include/)
include_directories(${HODLR_Path}/IML/include/)
include_directories(${CPPUNIT_INCLUDE_DIRS})
include_directories(${SparseMultiFrontal_SOURCE_DIR}/include)
include_directories(${SparseMultiFrontal_SOURCE_DIR}/IML/include)
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${SCOTCH_INCLUDE_DIRS})
include_directories(${PASTIX_INCLUDE_DIRS})

# Directories to look for libraries
link_directories(${SparseMultiFrontal_BINARY_DIR}/src)

# Compile tests from test.cpp
add_executable(IML_Test test.cpp)

#find_package(BLAS REQUIRED)
#add_executable(bench EigenVsBLAS.cpp)

# test should link to solver libraries
target_link_libraries(IML_Test sparseSolver HODLRSolver ${SCOTCH_LIBRARY} ${SCOTCH_scotcherr_LIBRARY} ${SCOTCH_scotcherrexit_LIBRARY} ${CPPUNIT_LIBRARIES} ${PASTIX_LIBRARIES})

#target_link_libraries(bench ${BLAS_LIBRARIES})

# Tests to run
add_test(BasicTests sparseTest)

