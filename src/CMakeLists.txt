# Directories to look for include files
include_directories(${SparseMultiFrontal_SOURCE_DIR}/include)
include_directories(${SparseMultiFrontal_SOURCE_DIR}/IML/include/)
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${SCOTCH_INCLUDE_DIRS})
include_directories(${HODLR_Path}/include/)
include_directories(${HODLR_Path}/IML/include/)


set (HODLR_src_files "")
####################################################
# Find all C/C++ files recursively from current dir
####################################################
foreach ( ext IN ITEMS "cpp" "cxx" "cc" "c" )
	file ( GLOB_RECURSE _files "${HODLR_Path}/src/*.${ext}" )
	set ( HODLR_src_files ${HODLR_src_files} ${_files} )
endforeach ()


add_library(HODLRSolver ${HODLR_src_files})	

set ( src_files "" )
####################################################
# Find all C/C++ files recursively from current dir
####################################################
foreach ( ext IN ITEMS "cpp" "cxx" "cc" "c" )
	file ( GLOB_RECURSE _files "*.${ext}" )
	set ( src_files ${src_files} ${_files} )
endforeach ()

# Create a library called sparseSolver which includes the source files defined
add_library(sparseSolver ${src_files})
