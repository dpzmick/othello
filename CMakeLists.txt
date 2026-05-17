cmake_minimum_required(VERSION 3.22)
project(othello LANGUAGES C)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -D_GNU_SOURCE")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror -Wall -Wextra -Wstrict-aliasing=2 -Wconversion -Wdouble-promotion -Wformat-security -Wimplicit-fallthrough")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -fstrict-aliasing -fverbose-asm")

if (NOT EMSCRIPTEN)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
  # On darwin (always M-series here) -march=native is safe and gets us full
  # NEON support. On x86 Linux the binary may be built and run on different
  # CPUs (e.g. Modal's build vs runtime), so use a portable AVX2-baseline
  # target instead. x86-64-v3 covers most CPUs from ~2014 onward (AVX2, FMA,
  # BMI2). Bump to v4 if we ever need AVX-512.
  if (APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mtune=native")
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=x86-64-v3 -mtune=generic")
  endif()
endif()

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -g")

if (EMSCRIPTEN)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s MODULARIZE=1 -s EXPORT_ES6=1 -s EXPORTED_RUNTIME_METHODS=ccall,cwrap")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ASSERTIONS=1")
endif()

if (NOT EMSCRIPTEN)
  find_package(PkgConfig)
  pkg_check_modules(zstd REQUIRED IMPORTED_TARGET libzstd)
endif()

add_subdirectory(third-party)

# libraries
add_subdirectory(libcommon)
add_subdirectory(libcomputer)
add_subdirectory(libothello)
add_subdirectory(libunit)
add_subdirectory(web_game)

# applications
add_subdirectory(apps)
add_subdirectory(tests)
