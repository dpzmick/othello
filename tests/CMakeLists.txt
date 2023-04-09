add_executable(test test.c test_othello.c test_common.c)
target_link_libraries(test libunit libothello)
