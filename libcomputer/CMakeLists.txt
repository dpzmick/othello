add_library(libcomputer STATIC mcts.c nn.c)
target_link_libraries(libcomputer INTERFACE m)
