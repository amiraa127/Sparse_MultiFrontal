
# Directories to look for include files
include_directories(${SparseMultiFrontal_SOURCE_DIR}/include)
include_directories(/usr/local/include/)
include_directories(/usr/local/include/eigen3/)

# Add HODLR library
set(HODLR_Path /Users/Amir/Documents/Uni\ Works/PhD/Research/Solver\ Codes/Dense_HODLR)
include_directories(${HODLR_Path}/include/)
set(HODLR_source_files
  ${HODLR_Path}/src/HODLR_Matrix.cpp	
  ${HODLR_Path}/src/HODLR_Tree.cpp		
  ${HODLR_Path}/src/helperFunctions.cpp	
  ${HODLR_Path}/src/recLU_FactorTree.cpp	
  ${HODLR_Path}/src/user_IndexTree.cpp
  ${HODLR_Path}/src/lowRank.cpp
  ${HODLR_Path}/src/kernel.cpp
)
add_library(HODLRSolver ${HODLR_source_files})	

# Define a variable sparseSolver_source_files containing a list of the source files
set(sparseSolver_source_files
  sparseMF.cpp
  extendAdd.cpp
  eliminationTree.cpp
  helperFunctions.cpp
  matrixIO.cpp
)

# Create a library called sparseSolver which includes the source files defined
add_library(sparseSolver ${sparseSolver_source_files})
