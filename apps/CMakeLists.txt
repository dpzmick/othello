if (NOT EMSCRIPTEN)
  add_executable(data_gen data_gen.c)
  target_link_libraries(data_gen libcommon libothello)
endif()
