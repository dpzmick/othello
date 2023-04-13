cmake_minimum_required(VERSION 3.26)
project(othello LANGUAGES C)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -D_GNU_SOURCE")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror -Wall -Wextra -Wstrict-aliasing=2 -Wconversion -Wdouble-promotion -Wformat-security -Wimplicit-fallthrough")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -fstrict-aliasing -fverbose-asm")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")

if (NOT EMSCRIPTEN)
  #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mtune=native")
endif()

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -g")

if (EMSCRIPTEN)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s MODULARIZE=1 -s EXPORT_ES6=1 -s EXPORTED_RUNTIME_METHODS=ccall,cwrap")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ASSERTIONS=1")
endif()

# libraries
add_subdirectory(libcommon)
add_subdirectory(libcomputer)
add_subdirectory(libothello)
add_subdirectory(libunit)
add_subdirectory(web_game)

# applications
add_subdirectory(apps)
add_subdirectory(tests)
