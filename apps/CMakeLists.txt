find_package(OpenMP)

if (NOT EMSCRIPTEN)
  add_executable(data_gen data_gen.c)
  target_link_libraries(data_gen libcommon libothello)

  if(OpenMP_C_FOUND)
    target_link_libraries(data_gen PUBLIC OpenMP::OpenMP_C)
  endif()
endif()
