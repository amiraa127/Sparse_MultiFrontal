cmake_minimum_required(VERSION 2.8)
project(SparseMultiFrontal)
set (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# possible options
option(ENABLE_TESTING false)
option(ENABLE_PASTIX false)


# detect for c++11 - features
if (${CMAKE_VERSION} VERSION_GREATER 3.0)                                                                                                                                                                      
  set (needed_cxx11_features cxx_rvalue_references cxx_alias_templates)
  # add target_compile_features(target PRIVATE ${needed_cxx11_features}) for all your targets
endif ()


# enable c++11
if (CMAKE_COMPILER_IS_GNUCXX)
  execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
  if (GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7)
    add_definitions("-std=gnu++11")
  elseif(GCC_VERSION VERSION_GREATER 4.3 OR GCC_VERSION VERSION_EQUAL 4.3)
    add_definitions("-std=gnu++0x")
  else ()
    message(FATAL_ERROR "C++11 is needed. Therefore a gcc compiler with a version >= 4.3 is needed.")   
  endif()
else (CMAKE_COMPILER_IS_GNUCXX)
  message(WARNING "C++11 needed. Add a compiler-flag to CMAKE_CXX_FLAGS manually or specify the c++11 standard in your build environment.")
endif(CMAKE_COMPILER_IS_GNUCXX)


# Find the dependencies.
find_package (HODLR REQUIRED)
if (HODLR_FOUND)
    message ("-- Found HODLR: ${HODLR_Path}")
endif (HODLR_FOUND)

find_package (Eigen REQUIRED)
if (EIGEN_FOUND)
    message ("-- Found Eigen version ${EIGEN_VERSION}: ${EIGEN_INCLUDE_DIRS}")
endif (EIGEN_FOUND)

find_package (Scotch REQUIRED)
if (SCOTCH_FOUND)
    message ("-- Found SCOTCH version ${SCOTCH_VERSION}: ${SCOTCH_INCLUDE_DIRS}")
endif (SCOTCH_FOUND)


# if option ENABLE_PASTIX is set to ON
if (ENABLE_PASTIX)
  find_package (Pastix REQUIRED)
  if (PASTIX_FOUND)
    message ("-- Found Pastix version ${PASTIX_VERSION}: ${PASTIX_INCLUDE_DIRS}")
  endif (PASTIX_FOUND)
else ()
   add_definitions (-DNOPASTIX)
endif(ENABLE_PASTIX)

add_subdirectory(src)


# if option ENABLE_TESTING is set to ON
if (ENABLE_TESTING)
  # add option for additional benchmarks
  option(ENABLE_BENCHMARKS false)
  
  # Enable test running via 'make test' command
  enable_testing()

  find_package (CppUnit REQUIRED)
  if (CPPUNIT_FOUND)
      message ("-- Found CppUnit version ${CPPUNIT_VERSION}: ${CPPUNIT_INCLUDE_DIRS}")
  endif (CPPUNIT_FOUND)

  add_subdirectory(test)
  add_subdirectory(IML/test)

  if (ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
  endif (ENABLE_BENCHMARKS)
  
endif (ENABLE_TESTING)
