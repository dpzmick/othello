import numpy as np

WHITE = 1
BLACK = -1
DRAW = 42

def _other_player(player):
    if player == WHITE:
        return BLACK
    elif player == BLACK:
        return WHITE
    else:
        raise ValueError("Player must be either BLACK or WHITE")


def _make_board():
    return np.zeros((8,8), dtype=np.int8)


def _in_bounds(x,y):
    return x > 0 and x < 8 and y > 0 and y < 8


def _gen_moves(board, x, y, dx, dy):
    """
    Walk in direction from (x,y) finding a move that can be made from here, if
    any.

    Returns a full board containing the single move for convinience.
    """

    player_color = board[x,y]
    assert player_color == WHITE or player_color == BLACK, "must be a piece at starting position"

    x += dx
    y += dy

    found_enemy = False
    while True:
        # we hit the edge of the board, cannot outflank anything
        if not _in_bounds(x,y):
            return _make_board()

        curr_piece = board[x,y]

        if curr_piece == _other_player(player_color):
            found_enemy = True

        # we can't capture through our own piece
        # no ability to capture in this direction
        if curr_piece == player_color:
            return _make_board()

        # empy piece ends the run. If we got here, and we actually saw an enemy
        # piece along the way, we can play here
        if curr_piece == 0:
            ret = _make_board()
            if found_enemy:
                ret[x,y] = 1

            return ret

        x += dx
        y += dy


def _gen_flips(board, player_color, x, y, dx, dy):
    """
    Walk in direction from (x,y) finding any pieces that are flipped by playing
    here. Assumes that (x,y) is a valid move for this player, but we aren't sure
    what direction it was a valid move for.

    Returns a board containing the new players color for any flipped piece.
    """

    flips = _make_board()

    x += dx
    y += dy

    # we're starting at empty cell b.c. this is a valid move
    # trying to find out own cell

    found_enemy = False
    while True:
        # we hit the edge of the board, didn't end up outflanking anything
        if not _in_bounds(x,y):
            return _make_board()

        curr_piece = board[x,y]

        if curr_piece == _other_player(player_color):
            found_enemy = True
            flips[x,y] = player_color

        # we can't capture through our own piece
        # no ability to capture in this direction
        if curr_piece == player_color:
            flips[x,y] = player_color

            if found_enemy:
                return flips
            else:
                return _make_board()

        # empy piece ends the run. If we got here, and we actually saw an enemy
        # piece along the way, we can play here
        if curr_piece == 0:
            return _make_board()

        x += dx
        y += dy


class OthelloGame(object):
    def __init__(self):
        self.player = BLACK # always start with black

        self.board = _make_board()
        self.board[3,3] = WHITE
        self.board[4,4] = WHITE
        self.board[3,4] = BLACK
        self.board[4,3] = BLACK


    def __eq__(self, other):
        return np.array_equal(self.board, other.board) and self.player == other.player


    def __hash__(self):
        return hash(str(self.player) + str(self.board.tostring()))


    def _valid_moves(self, player):
        moves = _make_board()

        for x, y in np.ndindex(self.board.shape):
            if self.board[x,y] == player:
                moves |= _gen_moves(self.board, x, y,  0,  1)
                moves |= _gen_moves(self.board, x, y,  0, -1)
                moves |= _gen_moves(self.board, x, y,  1,  0)
                moves |= _gen_moves(self.board, x, y, -1,  0)
                moves |= _gen_moves(self.board, x, y,  1,  1)
                moves |= _gen_moves(self.board, x, y,  1, -1)
                moves |= _gen_moves(self.board, x, y, -1,  1)
                moves |= _gen_moves(self.board, x, y, -1, -1)

        return moves


    def valid_moves(self):
        return self._valid_moves(self.player)


    def make_move(self, mx, my):
        moves = self.valid_moves()
        assert moves[mx,my] == 1, "Not a valid move"

        # pass move?

        # |= will mask in anything truthy, so the -1s can be masked in and will
        # remain -1
        flips  = _make_board()
        flips |= _gen_flips(self.board, self.player, mx, my,  0,  1)
        flips |= _gen_flips(self.board, self.player, mx, my,  0, -1)
        flips |= _gen_flips(self.board, self.player, mx, my,  1,  0)
        flips |= _gen_flips(self.board, self.player, mx, my, -1,  0)
        flips |= _gen_flips(self.board, self.player, mx, my,  1,  1)
        flips |= _gen_flips(self.board, self.player, mx, my,  1, -1)
        flips |= _gen_flips(self.board, self.player, mx, my, -1,  1)
        flips |= _gen_flips(self.board, self.player, mx, my, -1, -1)

        mask = np.nonzero(flips)
        self.board[mx,my] = self.player
        self.board[mask] = flips[mask]

        self.player = _other_player(self.player)


    def is_game_over(self):
        """returns a player, or False"""
        my_moves = self._valid_moves(self.player)
        other_moves = self._valid_moves(_other_player(self.player))

        if np.all(my_moves == 0) and np.all(other_moves == 0):
            white_pieces = np.count_nonzero(self.board == WHITE)
            black_pieces = np.count_nonzero(self.board == BLACK)

            if white_pieces == black_pieces:
                return DRAW

            if white_pieces > black_pieces:
                return WHITE

            if black_pieces > white_pieces:
                return BLACK


    def play_randomly(self):
        """
        make random moves, starting with player, until the game is done
        """
        while True:
            winner = self.is_game_over()
            if winner:
                return winner

            moves = self.valid_moves()
            if np.all(moves == 0):
                self.player = _other_player(self.player)
                continue

            else:
                xs, ys = np.nonzero(moves)
                idx = np.random.choice(len(xs))

                mx, my = (xs[idx], ys[idx])

                self.make_move(mx, my)

                self.player = _other_player(self.player)

# g = OthelloGame()
# print(g.play_randomly(WHITE))
