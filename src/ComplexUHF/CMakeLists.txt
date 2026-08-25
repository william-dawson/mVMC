# include guard
if(${CMAKE_PROJECT_NAME} STREQUAL "Project")
  message(FATAL_ERROR "cmake should be executed not for 'src' subdirectory, but for the top directory of UHF.")
endif(${CMAKE_PROJECT_NAME} STREQUAL "Project")

include_directories(include)
include_directories(../common)
set(SOURCES_UHF
        UHFmain.c output.c cal_energy.c green.c makeham.c diag.c initial.c matrixlapack.c readdef.c ../common/setmemory.c
 )

include_directories(../sfmt)
add_definitions(-DMEXP=19937)

set(SOURCES_sfmt
        ../sfmt/SFMT.c   
 )

# UHF is serial and contains no OpenMP region of its own, so unlike vmc.out
# it can safely link a threaded LAPACK -- there is nothing to nest against.
set(UHF_LAPACK_LIBRARIES "${LAPACK_LIBRARIES}" CACHE STRING
    "LAPACK for the serial UHF tool; may be threaded (e.g. -SSL2BLAMP)")
option(UHF_USE_ZHEEVD "Use divide-and-conquer zheevd in UHF's diagonalization" OFF)

add_executable(UHF ${SOURCES_UHF} ${SOURCES_sfmt})
target_link_libraries(UHF ${UHF_LAPACK_LIBRARIES} m)
if(UHF_USE_ZHEEVD)
  target_compile_definitions(UHF PRIVATE UHF_USE_ZHEEVD)
endif()
install(TARGETS UHF RUNTIME DESTINATION bin)
