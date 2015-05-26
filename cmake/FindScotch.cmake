# This module looks for SCOTCH (http://www.labri.fr/perso/pelegrin/scotch)
# The following variables are provided:
#
#  SCOTCH_FOUND                 TRUE if SCOTCH has been found
#  SCOTCH_VERSION               the version of the library (version.release.patchlevel)
#  SCOTCH_INCLUDE_DIRS          the include path for SCOTCH
#  SCOTCH_scotch_LIBRARY        the main SCOTCH library
#  SCOTCH_scotcherr_LIBRARY     the error handling library
#  SCOTCH_scotcherrexit_LIBRARY the alternative error handling library
#  SCOTCH_LIBRARIES             a list of all libraries provided by SCOTCH

find_path(SCOTCH_INCLUDE_DIR scotch.h 
	  HINTS $ENV{SCOTCH_DIR}/include $ENV{SCOTCH_ROOT}/include $ENV{SCOTCH_INC})

if(SCOTCH_INCLUDE_DIR)
  find_library(SCOTCH_scotch_LIBRARY NAMES scotch 
	       HINTS ${SCOTCH_INCLUDE_DIR}/../lib $ENV{SCOTCH_LIB})
  find_library(SCOTCH_scotcherr_LIBRARY NAMES scotcherr 
	       HINTS ${SCOTCH_INCLUDE_DIR}/../lib $ENV{SCOTCH_LIB})
  find_library(SCOTCH_scotcherrexit_LIBRARY NAMES scotcherrexit 
	       HINTS ${SCOTCH_INCLUDE_DIR}/../lib $ENV{SCOTCH_LIB})
  
  # read version nr rom scotch.h file
  file(STRINGS "${SCOTCH_INCLUDE_DIR}/scotch.h" scotch_version_line 
       REGEX "#define SCOTCH_VERSION [0-9]+" LIMIT_COUNT 1)
  file(STRINGS "${SCOTCH_INCLUDE_DIR}/scotch.h" scotch_release_line 
       REGEX "#define SCOTCH_RELEASE [0-9]+" LIMIT_COUNT 1)
  file(STRINGS "${SCOTCH_INCLUDE_DIR}/scotch.h" scotch_patchlevel_line 
       REGEX "#define SCOTCH_PATCHLEVEL [0-9]+" LIMIT_COUNT 1)
  if(scotch_version_line)
    string(REGEX MATCH "[0-9]+" scotch_version ${scotch_version_line})
    string(REGEX MATCH "[0-9]+" scotch_release ${scotch_release_line})
    string(REGEX MATCH "[0-9]+" scotch_patchlevel ${scotch_patchlevel_line})
    set(SCOTCH_VERSION "${scotch_version}.${scotch_release}.${scotch_patchlevel}")
  endif(scotch_version_line)

  if(SCOTCH_scotch_LIBRARY)
    set(SCOTCH_INCLUDE_DIRS ${SCOTCH_INCLUDE_DIR})
    set(SCOTCH_LIBRARIES ${SCOTCH_scotch_LIBRARY} ${SCOTCH_scotcherr_LIBRARY} ${SCOTCH_scotcherrexit_LIBRARY})
  endif(SCOTCH_scotch_LIBRARY)

endif(SCOTCH_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  SCOTCH
  FOUND_VAR      SCOTCH_FOUND
  REQUIRED_VARS  SCOTCH_INCLUDE_DIRS SCOTCH_scotch_LIBRARY SCOTCH_scotcherr_LIBRARY SCOTCH_scotcherrexit_LIBRARY
  VERSION_VAR    SCOTCH_VERSION)