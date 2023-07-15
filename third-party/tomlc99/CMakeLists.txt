if (NOT EMSCRIPTEN)
  add_library(tomlc99 toml.c)
  target_include_directories(tomlc99 SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/)

  # our default -Wconversion does not cooperate with this codebase
  # out default strict aliasing rules do not cooperate with this codebase
  # disabling strict aliasing for just this compilation unit should be okay
  target_compile_options(tomlc99 PRIVATE -Wno-conversion -fno-strict-aliasing)
endif()
