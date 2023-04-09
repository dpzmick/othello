add_library(libothello STATIC othello.c)
target_link_libraries(libothello INTERFACE libcommon)
