# CMake module to search for Mpi library
#
# If it's found it sets MPI_FOUND to TRUE
# and following variables are set:
#    MPI_INCLUDE_DIR
#    MPI_LIBRARY


FIND_PATH(MPI_INCLUDE_DIR mpi.h 
        /usr/local/include 
        /usr/include 
        c:/msys/local/include 
        /opt/lam-7.1.3/include
)

FIND_LIBRARY(MPI_LIBRARY NAMES mpi PATHS 
       /usr/local/lib 
       /usr/lib 
       c:/msys/local/lib
       /opt/lam-7.1.3/lib
)


IF (MPI_INCLUDE_DIR AND MPI_LIBRARY)
   SET(MPI_FOUND TRUE)
ENDIF (MPI_INCLUDE_DIR AND MPI_LIBRARY)


IF (MPI_FOUND)

   IF (NOT MPI_FIND_QUIETLY)
      MESSAGE(STATUS "Found Mpi: ${MPI_LIBRARY}")
   ENDIF (NOT MPI_FIND_QUIETLY)

ELSE (MPI_FOUND)

   IF (MPI_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Could not find Mpi")
   ELSE (MPI_FIND_REQUIRED)
      IF(NOT MPI_FIND_QUIETLY)
        MESSAGE(STATUS "Could not find MPI")
      ENDIF(NOT MPI_FIND_QUIETLY)
      # Avoid cmake complaints if mpi is not found
      SET(MPI_INCLUDE_DIR "")
      SET(MPI_LIBRARY "")
   ENDIF (MPI_FIND_REQUIRED)

ENDIF (MPI_FOUND)