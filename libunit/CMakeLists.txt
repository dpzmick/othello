if (NOT EMSCRIPTEN)
  add_library(libunit STATIC unit.c)
endif()
