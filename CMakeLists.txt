CMAKE_MINIMUM_REQUIRED (VERSION 2.8)
PROJECT (ESoinnText)

SET (CMAKE_CXX_FLAGS "-std=c++11")
#SET (CMAKE_CXX_FLAGS "-frtti")

IF(MSVC)
    # Force to always compile with W4
    IF(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
        STRING(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    ELSE()
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    ENDIF()
    ELSEIF(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
        # Update if necessary
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
    ENDIF()

ENABLE_TESTING()
FIND_PACKAGE (Boost 1.50.0 COMPONENTS
    filesystem
    iostreams
    regex
    program_options
    system
    thread
    unit_test_framework
REQUIRED)
INCLUDE_DIRECTORIES (${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES (${PROJECT_SOURCE_DIR}/include)
ADD_SUBDIRECTORY (src)
