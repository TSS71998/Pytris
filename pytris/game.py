"""Core Tetris engine. No graphics here, so the same rules are used by the
human-playable GUI, the AI viewer and headless evaluation."""

import random

WIDTH, HEIGHT = 10, 20
PIECE_NAMES = "IOTSZJL"

_SHAPES = {
    "I": ["....", "XXXX", "....", "...."],
    "O": ["XX", "XX"],
    "T": [".X.", "XXX", "..."],
    "S": [".XX", "XX.", "..."],
    "Z": ["XX.", ".XX", "..."],
    "J": ["X..", "XXX", "..."],
    "L": ["..X", "XXX", "..."],
}


def _build_rotations(rows):
    """Return the 4 clockwise rotations as tuples of (row, col) offsets."""
    n = len(rows)
    cells = [(r, c) for r in range(n) for c in range(n) if rows[r][c] == "X"]
    rots = []
    for _ in range(4):
        rots.append(tuple(sorted(cells)))
        cells = [(c, n - 1 - r) for r, c in cells]  # rotate clockwise
    return rots


ROTATIONS = {name: _build_rotations(rows) for name, rows in _SHAPES.items()}
BOX_SIZE = {name: len(rows) for name, rows in _SHAPES.items()}
PIECE_ID = {name: i + 1 for i, name in enumerate(PIECE_NAMES)}  # 0 = empty cell

LINE_SCORES = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
# (dx, dy) offsets tried when a rotation is blocked: a simple wall/floor kick.
KICKS = [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1), (-1, -1), (1, -1)]


class Piece:
    __slots__ = ("name", "rot", "x", "y")

    def __init__(self, name, rot, x, y):
        self.name, self.rot, self.x, self.y = name, rot, x, y

    def cells(self, rot=None, x=None, y=None):
        rot = self.rot if rot is None else rot
        x = self.x if x is None else x
        y = self.y if y is None else y
        return [(y + r, x + c) for r, c in ROTATIONS[self.name][rot]]


def spawn_position(name, rot=0):
    """Top-left (x, y) where a piece appears: centered, top cell on row 0."""
    cells = ROTATIONS[name][rot]
    return (WIDTH - BOX_SIZE[name]) // 2, -min(r for r, _ in cells)


class Tetris:
    def __init__(self, seed=None, queue_size=5):
        self.rng = random.Random(seed)
        self.board = [[0] * WIDTH for _ in range(HEIGHT)]
        self.bag = []
        self.queue = [self._next_from_bag() for _ in range(queue_size)]
        self.hold_name = None
        self.hold_used = False
        self.score = 0
        self.lines = 0
        self.level = 1
        self.pieces = 0
        self.tetrises = 0
        self.last_clear = 0
        self.game_over = False
        self.current = None
        self._spawn()

    # ---------------------------------------------------------------- pieces
    def _next_from_bag(self):
        """7-bag randomizer: every piece once per shuffled bag of seven."""
        if not self.bag:
            self.bag = list(PIECE_NAMES)
            self.rng.shuffle(self.bag)
        return self.bag.pop()

    def _spawn(self, name=None):
        if name is None:
            name = self.queue.pop(0)
            self.queue.append(self._next_from_bag())
        x, y = spawn_position(name)
        self.current = Piece(name, 0, x, y)
        if not self.fits(name, 0, x, y):
            self.game_over = True

    def fits(self, name, rot, x, y):
        board = self.board
        for r, c in ROTATIONS[name][rot]:
            rr, cc = y + r, x + c
            if cc < 0 or cc >= WIDTH or rr >= HEIGHT:
                return False
            if rr >= 0 and board[rr][cc]:
                return False
        return True

    # --------------------------------------------------------------- actions
    def move(self, dx):
        p = self.current
        if not self.game_over and self.fits(p.name, p.rot, p.x + dx, p.y):
            p.x += dx
            return True
        return False

    def rotate(self, direction=1):
        """direction=1 clockwise, -1 counter-clockwise."""
        if self.game_over:
            return False
        p = self.current
        new_rot = (p.rot + direction) % 4
        for dx, dy in KICKS:
            if self.fits(p.name, new_rot, p.x + dx, p.y + dy):
                p.rot, p.x, p.y = new_rot, p.x + dx, p.y + dy
                return True
        return False

    def on_ground(self):
        p = self.current
        return not self.fits(p.name, p.rot, p.x, p.y + 1)

    def gravity_step(self):
        """Move the piece down one row. Returns False if it is resting."""
        if self.game_over or self.on_ground():
            return False
        self.current.y += 1
        return True

    def soft_drop(self):
        if self.gravity_step():
            self.score += 1
            return True
        return False

    def ghost_y(self):
        p = self.current
        y = p.y
        while self.fits(p.name, p.rot, p.x, y + 1):
            y += 1
        return y

    def hard_drop(self):
        if self.game_over:
            return 0
        y = self.ghost_y()
        dist = y - self.current.y
        self.current.y = y
        self.score += 2 * dist
        return self.lock()

    def hold(self):
        if self.game_over or self.hold_used:
            return False
        name = self.current.name
        if self.hold_name is None:
            self.hold_name = name
            self._spawn()
        else:
            self.hold_name, swap = name, self.hold_name
            self._spawn(swap)
        self.hold_used = True
        return True

    def lock(self):
        """Freeze the current piece, clear lines, spawn the next one.
        Returns the number of lines cleared."""
        p = self.current
        pid = PIECE_ID[p.name]
        for rr, cc in p.cells():
            if rr < 0:  # locked above the visible field
                self.game_over = True
                return 0
            self.board[rr][cc] = pid
        full = [r for r in range(HEIGHT) if all(self.board[r])]
        for r in full:
            del self.board[r]
            self.board.insert(0, [0] * WIDTH)
        n = len(full)
        self.last_clear = n
        self.score += LINE_SCORES[n] * self.level
        self.lines += n
        self.tetrises += n == 4
        self.level = self.lines // 10 + 1
        self.pieces += 1
        self.hold_used = False
        self._spawn()
        return n

    def place(self, rot, x):
        """Teleport-and-drop used by headless AI evaluation: put the current
        piece in rotation `rot` at column `x`, then hard drop."""
        p = self.current
        _, y = spawn_position(p.name, rot)
        if not self.fits(p.name, rot, x, y):
            self.game_over = True
            return 0
        p.rot, p.x, p.y = rot, x, y
        return self.hard_drop()

    def gravity_interval(self):
        """Seconds per row at the current level (Tetris guideline curve)."""
        lvl = min(self.level, 20)
        return (0.8 - (lvl - 1) * 0.007) ** (lvl - 1)