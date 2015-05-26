# This module looks for HODLR (https://github.com/amiraa127/Dense_HODLR)
# The following variables are provided:
#
#  HODLR_FOUND                 TRUE if CppUnit has been found
#  HODLR_Path                  the path for the HODLR root directory

find_path(HODLR_INCLUDE_DIR HODLR_Tree.hpp 
	  HINTS $ENV{HODLR_DIR}/include $ENV{HODLR_ROOT}/include $ENV{HODLR_INC})

if(HODLR_INCLUDE_DIR)
    set(HODLR_Path ${HODLR_INCLUDE_DIR}/../)
endif(HODLR_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HODLR DEFAULT_MSG HODLR_Path)