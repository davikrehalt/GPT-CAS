include(FindPackageHandleStandardArgs)

set(_GMP_HINTS)
if(DEFINED ENV{GMP_ROOT})
  list(APPEND _GMP_HINTS "$ENV{GMP_ROOT}")
endif()
if(APPLE)
  list(APPEND _GMP_HINTS /opt/homebrew/opt/gmp /usr/local/opt/gmp)
endif()

find_path(
  GMP_INCLUDE_DIR
  NAMES gmp.h
  HINTS ${_GMP_HINTS}
  PATH_SUFFIXES include
)
find_path(
  GMPXX_INCLUDE_DIR
  NAMES gmpxx.h
  HINTS ${_GMP_HINTS}
  PATH_SUFFIXES include
)
find_library(
  GMP_LIBRARY
  NAMES gmp
  HINTS ${_GMP_HINTS}
  PATH_SUFFIXES lib
)
find_library(
  GMPXX_LIBRARY
  NAMES gmpxx
  HINTS ${_GMP_HINTS}
  PATH_SUFFIXES lib
)

find_package_handle_standard_args(
  GMP
  REQUIRED_VARS
    GMP_INCLUDE_DIR
    GMPXX_INCLUDE_DIR
    GMP_LIBRARY
    GMPXX_LIBRARY
)

if(GMP_FOUND AND NOT TARGET GMP::gmp)
  add_library(GMP::gmp UNKNOWN IMPORTED)
  set_target_properties(
    GMP::gmp
    PROPERTIES
      IMPORTED_LOCATION "${GMP_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIR}"
  )
endif()

if(GMP_FOUND AND NOT TARGET GMP::gmpxx)
  add_library(GMP::gmpxx UNKNOWN IMPORTED)
  set_target_properties(
    GMP::gmpxx
    PROPERTIES
      IMPORTED_LOCATION "${GMPXX_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIR};${GMPXX_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES GMP::gmp
  )
endif()

mark_as_advanced(
  GMP_INCLUDE_DIR
  GMPXX_INCLUDE_DIR
  GMP_LIBRARY
  GMPXX_LIBRARY
)
