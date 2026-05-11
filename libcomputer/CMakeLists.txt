add_library(libcomputer STATIC mcts.c nn.c nn_policy.c)
target_link_libraries(libcomputer INTERFACE m)
