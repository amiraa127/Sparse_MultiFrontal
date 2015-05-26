# This module looks for CppUnit (http://sourceforge.net/projects/cppunit)
# The following variables are provided:
#
#  CPPUNIT_FOUND                 TRUE if CppUnit has been found
#  CPPUNIT_VERSION               the version of the library (version.release.patchlevel)
#  CPPUNIT_INCLUDE_DIRS          the include path for CppUnit
#  CPPUNIT_LIBRARIES             a list of all libraries provided by CppUnit

find_path(CPPUNIT_INCLUDE_DIR cppunit/TestCase.h 
	  HINTS $ENV{CPPUNIT_DIR}/include $ENV{CPPUNIT_ROOT}/include $ENV{CPPUNIT_INC})

if(CPPUNIT_INCLUDE_DIR)
  find_library(CPPUNIT_LIBRARY cppunit HINTS ${CPPUNIT_INCLUDE_DIR}/../lib $ENV{CPPUNIT_LIB})
  
  # read version nr rom Portability.h file
  file(STRINGS "${CPPUNIT_INCLUDE_DIR}/cppunit/Portability.h" cppunit_version_line 
       REGEX "#define CPPUNIT_VERSION \"[0-9.]+\"" LIMIT_COUNT 1)
  if(cppunit_version_line)
    string(REGEX MATCH "[0-9.]+" cppunit_version ${cppunit_version_line})
    set(CPPUNIT_VERSION "${cppunit_version}")
  endif(cppunit_version_line)
  
  if(CPPUNIT_LIBRARY)
    set(CPPUNIT_INCLUDE_DIRS ${CPPUNIT_INCLUDE_DIR})
    set(CPPUNIT_LIBRARIES ${CPPUNIT_LIBRARY} ${CMAKE_DL_LIBS})
  endif(CPPUNIT_LIBRARY)
endif(CPPUNIT_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  CPPUNIT
  FOUND_VAR      CPPUNIT_FOUND
  REQUIRED_VARS  CPPUNIT_LIBRARIES CPPUNIT_INCLUDE_DIRS
  VERSION_VAR    CPPUNIT_VERSION)
