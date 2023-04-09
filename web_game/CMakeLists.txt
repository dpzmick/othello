if (EMSCRIPTEN)
  add_executable(wasm_wrapper wasm_wrapper.c)
  target_link_libraries(wasm_wrapper libcomputer libothello libcommon)
endif()
