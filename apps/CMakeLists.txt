if (NOT EMSCRIPTEN)
  add_executable(data_gen_policy data_gen_policy.c)
  target_link_libraries(data_gen_policy libcommon libothello libcomputer tomlc99 PkgConfig::zstd)

  add_executable(thunderdome thunderdome.c)
  target_link_libraries(thunderdome libcommon libothello libcomputer m)

  add_executable(test_zstd test_zstd.c)
  target_link_libraries(test_zstd libcommon PkgConfig::zstd)
endif()
