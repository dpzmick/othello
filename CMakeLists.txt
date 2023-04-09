cmake_minimum_required(VERSION 3.26)
project(othello LANGUAGES C)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -D_GNU_SOURCE")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror -Wall -Wextra -Wstrict-aliasing=2 -Wconversion -Wdouble-promotion -Wformat-security -Wimplicit-fallthrough")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -fstrict-aliasing")

# libraries
add_subdirectory(libcommon)
add_subdirectory(libothello)
add_subdirectory(libunit)

# applications
add_subdirectory(apps)
add_subdirectory(tests)
