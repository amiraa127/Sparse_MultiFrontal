# Directories to look for include files
include_directories(${SparseMultiFrontal_SOURCE_DIR}/include)
include_directories(/usr/local/include/)
include_directories(/usr/local/include/eigen3/)
include_directories(/opt/local/include/openmpi-mp/)
include_directories(/opt/local/include/)

# Add HODLR library
set(HODLR_Path /Users/Amir/Documents/Uni\ Works/PhD/Research/Solver\ Codes/Dense_HODLR)
include_directories(${HODLR_Path}/include/)

# Add SCOTCH library
add_library(libSCOTCH STATIC IMPORTED)
set_target_properties(libSCOTCH PROPERTIES IMPORTED_LOCATION /usr/local/lib/libscotch.a)

add_library(libSCOTCHerr STATIC IMPORTED)
set_target_properties(libSCOTCHerr PROPERTIES IMPORTED_LOCATION /usr/local/lib/libscotcherr.a)

# Add CPPUNIT
add_library(libCPPUNIT STATIC IMPORTED)
set_target_properties(libCPPUNIT PROPERTIES IMPORTED_LOCATION /usr/local/lib/libcppunit.a)

# Directories to look for libraries
link_directories(${SparseMultiFrontal_BINARY_DIR}/src)

# Compile tests from test.cpp
add_executable(sparseTest test.cpp)

#find_package(BLAS REQUIRED)
#add_executable(bench EigenVsBLAS.cpp)

# test should link to solver libraries
target_link_libraries(sparseTest sparseSolver HODLRSolver libSCOTCH libSCOTCHerr libCPPUNIT)
#target_link_libraries(bench ${BLAS_LIBRARIES})

# Tests to run
add_test(BasicTests sparseTest)

