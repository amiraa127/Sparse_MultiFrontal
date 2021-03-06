#-------------------------------------------------------------------------------
#                ______             ______ ____          __  __
#               |  ____|           |  ____/ __ \   /\   |  \/  |
#               | |__ _ __ ___  ___| |__ | |  | | /  \  | \  / |
#               |  __| '__/ _ \/ _ \  __|| |  | |/ /\ \ | |\/| |
#               | |  | | |  __/  __/ |   | |__| / ____ \| |  | |
#               |_|  |_|  \___|\___|_|    \____/_/    \_\_|  |_|
#
#                   FreeFOAM: The Cross-Platform CFD Toolkit
#
# Copyright (C) 2008-2009 Michael Wild <themiwi@users.sf.net>
#                         Gerber van der Graaf <gerber_graaf@users.sf.net>
#-------------------------------------------------------------------------------
# License
#   This file is part of FreeFOAM.
#
#   FreeFOAM is free software; you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the
#   Free Software Foundation; either version 2 of the License, or (at your
#   option) any later version.
#
#   FreeFOAM is distributed in the hope that it will be useful, but WITHOUT
#   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#   FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
#   for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with FreeFOAM; if not, write to the Free Software Foundation,
#   Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA
#-------------------------------------------------------------------------------

# - Find SCOTCH
#
# This module looks for SCOTCH support and defines the following values
#  SCOTCH_FOUND                 TRUE if SCOTCH has been found
#  SCOTCH_INCLUDE_DIRS          the include path for SCOTCH
#  SCOTCH_LIBRARY               the library to link against
#  SCOTCH_scotcherr_LIBRARY     the error handling library to link against
#  SCOTCH_scotcherrexit_LIBRARY the alternative error handling library to link against

find_path( SCOTCH_INCLUDE_DIR scotch.h PATH_SUFFIXES scotch )

find_library( SCOTCH_LIBRARY
  NAMES scotch
  )

find_library( SCOTCH_scotcherr_LIBRARY
  NAMES scotcherr
  )

find_library( SCOTCH_scotcherrexit_LIBRARY
  NAMES scotcherrexit
  )

set( SCOTCH_INCLUDE_DIRS ${SCOTCH_INCLUDE_DIR} )
set( SCOTCH_LIBRARIES ${SCOTCH_LIBRARY} ${SCOTCH_scotcherrexit_LIBRARY} )

mark_as_advanced(
  SCOTCH_INCLUDE_DIR
  SCOTCH_LIBRARY
  SCOTCH_scotcherr_LIBRARY
  SCOTCH_scotcherrexit_LIBRARY
  )

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( SCOTCH
  DEFAULT_MSG
  SCOTCH_INCLUDE_DIR
  SCOTCH_LIBRARY
  SCOTCH_scotcherr_LIBRARY
  SCOTCH_scotcherrexit_LIBRARY
  )

# ------------------------- vim: set sw=2 sts=2 et: --------------- end-of-file