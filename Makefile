test_flags=-Wall -Wextra -Wconversion -Werror -g -O3

xxhash_cflags=$(shell pkg-config --cflags libxxhash)
xxhash_ldflags=$(shell pkg-config --libs libxxhash)

# ---------
# local system testing
test: test.c unit.c test_board.c test_tree.c bitboard.h unit.h game_tree.h common.h Makefile
	gcc ${xxhash_cflags} ${xxhash_ldflags} test.c unit.c test_board.c test_tree.c -o test ${test_flags}

run_%: %
	./$^

# ---------------

HEAP_SIZE      = 8388208
STACK_SIZE     = 61800

PRODUCT = Othello.pdx

# Locate the SDK
SDK = ${PLAYDATE_SDK_PATH}
ifeq ($(SDK),)
SDK = $(shell egrep '^\s*SDKRoot' ~/.Playdate/config | head -n 1 | cut -c9-)
endif

ifeq ($(SDK),)
$(error SDK path not found; set ENV value PLAYDATE_SDK_PATH)
endif

# List C source files here
SRC = main.c

# List all user directories here
UINCDIR =

# List user asm files
UASRC =

# List all user C define here, like -D_DEBUG=1
UDEFS =

# Define ASM defines here
UADEFS =

# List the user directory to look for the libraries here
ULIBDIR =

# List all user libraries here
ULIBS =

include $(SDK)/C_API/buildsupport/common.mk
