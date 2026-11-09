"""The Tetris AI.

How it plays
------------
For every possible placement of the current piece (each rotation x each
column) it simulates the drop, measures 8 features of the resulting board
(holes, bumpiness, wells, ...) and scores the board as a weighted sum of
those features. It picks the placement with the best score.

How it learns
-------------
It is not told which features are good or bad. The weights start random and
are learned with the noisy Cross-Entropy Method (Szita & Lorincz, 2006):

  1. Keep a Gaussian distribution (mean, std) over the weight vector.
  2. Sample a population of weight vectors and let each one play games.
  3. Keep the top 20% (the "elite"), refit the mean/std to the elite.
  4. Add a little noise to the std so it keeps exploring; repeat.

Over generations the distribution narrows onto weights that clear many lines.

The board is simulated with one integer bitmask per row, which keeps the
search fast in pure Python (no numpy or GPU needed).
"""

import json
import math
import os
import random
import time
from multiprocessing import Pool

from .game import HEIGHT, PIECE_NAMES, ROTATIONS, WIDTH

FULL = (1 << WIDTH) - 1
WALLS_ROW = (1 << (WIDTH + 1)) | 1
PAIR_MASK = (1 << (WIDTH + 1)) - 1

FEATURE_NAMES = [
    "landing_height",   # how high the piece came to rest
    "eroded_cells",     # lines cleared x piece cells that were cleared
    "row_transitions",  # filled/empty switches along each row
    "col_transitions",  # filled/empty switches down each column
    "holes",            # empty cells with something above them
    "wells",            # depth of narrow wells (1+2+..+d per well)
    "rows_with_holes",  # how many rows contain at least one hole
    "bumpiness",        # sum of height differences between neighbours
]
N_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------- piece data
def _piece_table():
    """For each piece: one entry per *distinct* rotation, with the data
    needed to drop it quickly in any column."""
    table = {}
    for name in PIECE_NAMES:
        seen, entries = set(), []
        for rot, cells in enumerate(ROTATIONS[name]):
            minc = min(c for _, c in cells)
            minr = min(r for r, _ in cells)
            norm = frozenset((r - minr, c - minc) for r, c in cells)
            if norm in seen:
                continue
            seen.add(norm)
            width = max(c for _, c in cells) - minc + 1
            # lowest piece cell (relative row) in each piece column
            bottoms = [max(r for r, c in cells if c - minc == j) for j in range(width)]
            rowmasks = {}
            for r, c in cells:
                rowmasks[r] = rowmasks.get(r, 0) | (1 << (c - minc))
            maxr = max(r for r, _ in cells)
            entries.append((rot, minc, minr, maxr, width, bottoms,
                            sorted(rowmasks.items())))
        table[name] = entries
    return table


PIECES = _piece_table()


def board_to_rows(board):
    """Convert the engine's 2D board into one bitmask per row."""
    return [sum(1 << c for c in range(WIDTH) if row[c]) for row in board]


def column_tops(rows):
    """Index of the highest filled row in each column (HEIGHT if empty)."""
    tops = [HEIGHT] * WIDTH
    seen = 0
    for r, b in enumerate(rows):
        new = b & ~seen
        if new:
            seen |= new
            for c in range(WIDTH):
                if new >> c & 1:
                    tops[c] = r
            if seen == FULL:
                break
    return tops


# ------------------------------------------------------------------ features
def board_features(rows, landing_height, eroded):
    row_t = col_t = holes = rows_holes = 0
    cover = seen = 0
    heights = [0] * WIDTH
    prev = 0  # the space above the board counts as empty
    for r in range(HEIGHT):
        b = rows[r]
        col_t += (b ^ prev).bit_count()
        prev = b
        if b:
            x = (b << 1) | WALLS_ROW
            row_t += ((x ^ (x >> 1)) & PAIR_MASK).bit_count()
            new = b & ~seen
            if new:
                seen |= new
                h = HEIGHT - r
                for c in range(WIDTH):
                    if new >> c & 1:
                        heights[c] = h
        hole = ~b & cover & FULL
        if hole:
            holes += hole.bit_count()
            rows_holes += 1
        cover |= b
    col_t += (FULL & ~prev).bit_count()  # the floor counts as filled

    wells = bump = 0
    for c in range(WIDTH):
        h = heights[c]
        left = heights[c - 1] if c > 0 else HEIGHT
        right = heights[c + 1] if c < WIDTH - 1 else HEIGHT
        d = min(left, right) - h
        if d > 0:
            wells += d * (d + 1) // 2
        if c < WIDTH - 1:
            bump += abs(h - right)
    return (landing_height, eroded, row_t, col_t, holes, wells, rows_holes, bump)


# --------------------------------------------------------------- placements
def placements(rows, name):
    """Yield (rot, game_x, new_rows, lines_cleared, features) for every legal
    hard-drop placement of piece `name` on the board `rows`."""
    tops = column_tops(rows)
    for rot, minc, minr, maxr, width, bottoms, rowmasks in PIECES[name]:
        for x in range(WIDTH - width + 1):
            # highest row offset y such that the piece sits on the stack
            y = min(tops[x + j] - 1 - bottoms[j] for j in range(width))
            if y + minr < 0:  # would stick out of the top: game over
                continue
            new = rows[:]
            for r, m in rowmasks:
                new[y + r] |= m << x
            cleared = [y + r for r, _ in rowmasks if new[y + r] == FULL]
            eroded = 0
            if cleared:
                eroded = len(cleared) * sum(
                    (m << x).bit_count() for r, m in rowmasks if y + r in cleared)
                keep = [b for b in new if b != FULL]
                new = [0] * len(cleared) + keep
            landing = HEIGHT - (y + (minr + maxr) / 2)
            yield rot, x - minc, new, len(cleared), board_features(new, landing, eroded)


def best_placement(rows, name, weights):
    best, best_score = None, -math.inf
    for rot, gx, new, lines, f in placements(rows, name):
        s = sum(w * v for w, v in zip(weights, f))
        if s > best_score:
            best_score, best = s, (rot, gx, new, lines)
    return best


class Agent:
    def __init__(self, weights):
        self.weights = list(weights)

    def choose(self, game):
        """Return (rotation, x) for the game's current piece, or None."""
        mv = best_placement(board_to_rows(game.board), game.current.name, self.weights)
        return None if mv is None else (mv[0], mv[1])

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls([data["weights"][n] for n in FEATURE_NAMES])


# ------------------------------------------------------------- fast games
def simulate_game(weights, seed, max_pieces):
    """Play one game headlessly. Returns (lines_cleared, pieces_placed)."""
    rng = random.Random(seed)
    rows = [0] * HEIGHT
    bag, lines = [], 0
    for n in range(max_pieces):
        if not bag:
            bag = list(PIECE_NAMES)
            rng.shuffle(bag)
        mv = best_placement(rows, bag.pop(), weights)
        if mv is None:
            return lines, n
        rows, lines = mv[2], lines + mv[3]
    return lines, max_pieces


def _evaluate(args):
    weights, seeds, max_pieces = args
    return sum(simulate_game(weights, s, max_pieces)[0] for s in seeds) / len(seeds)


# ------------------------------------------------------------------ training
def save_weights(path, weights, info=None):
    data = {"weights": dict(zip(FEATURE_NAMES, weights))}
    if info:
        data.update(info)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def train(generations=15, population=40, elite_frac=0.2, games=3,
          max_pieces=400, workers=None, out="weights.json", seed=None,
          log=print):
    """Learn feature weights with the noisy cross-entropy method."""
    rng = random.Random(seed)
    mean = [0.0] * N_FEATURES
    std = [10.0] * N_FEATURES
    n_elite = max(2, int(population * elite_frac))
    history = []
    workers = workers or os.cpu_count() or 1
    pool = Pool(workers) if workers > 1 else None
    best_ever = (-1.0, mean)
    try:
        for gen in range(1, generations + 1):
            t0 = time.time()
            # all candidates face the same piece sequences this generation
            seeds = [rng.randrange(1 << 30) for _ in range(games)]
            cands = [[rng.gauss(m, s) for m, s in zip(mean, std)]
                     for _ in range(population)]
            jobs = [(c, seeds, max_pieces) for c in cands]
            scores = pool.map(_evaluate, jobs) if pool else list(map(_evaluate, jobs))

            ranked = sorted(zip(scores, cands), key=lambda t: -t[0])
            elite = [c for _, c in ranked[:n_elite]]
            noise = max(4.0 - gen * 0.4, 0.1)  # decaying exploration noise
            for i in range(N_FEATURES):
                vals = [e[i] for e in elite]
                mean[i] = sum(vals) / n_elite
                var = sum((v - mean[i]) ** 2 for v in vals) / n_elite
                std[i] = math.sqrt(var + noise)

            if ranked[0][0] > best_ever[0]:
                best_ever = (ranked[0][0], ranked[0][1][:])
            elite_avg = sum(s for s, _ in ranked[:n_elite]) / n_elite
            history.append({"generation": gen, "best": ranked[0][0],
                            "elite_mean": elite_avg,
                            "population_mean": sum(scores) / len(scores)})
            log(f"gen {gen:3d} | best {ranked[0][0]:7.1f} | elite avg "
                f"{elite_avg:7.1f} | pop avg {sum(scores)/len(scores):7.1f} "
                f"lines/game | {time.time() - t0:5.1f}s")
            # the distribution mean is usually the most robust policy
            save_weights(out, mean, {"generation": gen,
                                     "max_pieces_per_game": max_pieces,
                                     "history": history})
    finally:
        if pool:
            pool.close()
            pool.join()
    return mean, history