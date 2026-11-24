# Pytris

Pytris is Tetris written in Python, with an AI that taught itself to play
it. You can play it yourself, train the AI from scratch and watch it learn,
or watch the trained AI play.

- **The game:** standard modern Tetris rules: 7-bag randomizer, hold, ghost
  piece, a five-piece next queue, wall kicks, lock delay, and levels that
  speed up gravity.
- **The AI:** a small, fast evaluator whose weights were learned with the
  cross-entropy method. It starts out knowing nothing about which boards are
  good, and after a minute of training it never tops out. Details are in
  [How the AI works](#how-the-ai-works).
- **No ML libraries:** the rules, the AI and its training are plain Python.
  The only dependency is [pygame](https://www.pygame.org) for the window.

[Rustris](https://github.com/TSS71998/Rustris) is the Rust rewrite of this
project, with an ASCII renderer, a versus mode and an AI trained for score.

## Quick start

```sh
pip install pygame
python main.py play
```

Needs Python 3.10 or newer (the AI uses `int.bit_count()`).

| Command | What it does |
| --- | --- |
| `python main.py play` | play Tetris yourself (also what `python main.py` does) |
| `python main.py watch` | watch the trained AI play |
| `python main.py train` | train the AI from scratch, rewriting `weights.json` |
| `python main.py eval` | measure the AI without opening a window |

The trained weights are in `weights.json`, so `watch` works straight away.
If the file is missing, `watch` plays with random untrained weights, which is
a good way to see what the AI is like before it learns anything.

## Controls

### Playing

| Key | Action |
| --- | --- |
| Left / Right | move (hold to auto-repeat) |
| Down | soft drop |
| Space | hard drop |
| Up / X | rotate clockwise |
| Z | rotate counter-clockwise |
| C / Shift | hold |
| P | pause |
| R | restart |
| Esc | quit |

### Watching the AI

| Key | Action |
| --- | --- |
| + / - | AI speed, 1 (slow) to 6 (instant) |
| P | pause |
| R | new game |
| Esc | quit |

`python main.py watch --speed 6` starts at full speed, and `--seed N` gives
the same pieces every time.

## How the project fits together

The rules, the AI and the window are kept apart. The engine has no graphics
or timing of its own, so the AI can train headless at thousands of games a
minute and the window just draws whatever state the engine is in.

| File | What it does |
| --- | --- |
| `pytris/game.py` | The rules: board, pieces, rotation with wall kicks, 7-bag, hold, line clears, scoring, levels, gravity speed. No graphics or timing. |
| `pytris/agent.py` | The AI: bitboards, move generation, board features, the `Agent` that picks moves, and the cross-entropy trainer. |
| `pytris/gui.py` | The pygame window: human play with gravity and 0.5 s lock delay, and the AI viewer. |
| `main.py` | Command line: `play`, `watch`, `train`, `eval`. |
| `weights.json` | The trained AI weights, plus the training history. |

## How the AI works

The AI does not see pixels or learn a neural network. It is a **one-move
evaluator**: for the piece in play it tries every place the piece could
land, scores each resulting board with a weighted sum of 8 measurements, and
picks the best. All of its "knowledge" is those 8 weights, which were
learned by having many candidate AIs play and keeping the best. For Tetris
this works better in practice than deep learning: it is tiny, fast, and very
strong.

The code is in `pytris/agent.py`. The sections below follow the order in
which a move is made.

### 1. Board representation

The engine stores the board as a 10 x 20 grid of piece ids, which is easy to
draw but slow to search. Each time the AI moves, it converts the board into
**20 integers**, one per row, with bit `c` set if column `c` is filled. A
full row is `0b11_1111_1111` (1023), clearing it is dropping one list entry,
and counting cells or transitions is a few bit operations and
`int.bit_count()`. This keeps the search fast without numpy.

### 2. Generating moves

A move is a rotation plus a column. When the module loads, it works out each
piece's **distinct rotations** (an O has 1, I/S/Z have 2, T/J/L have 4) and,
for each one, the lowest occupied cell in every column it covers.

To place a piece, it finds the highest filled cell in each column of the
board and drops the piece straight down until one of its columns would hit
the stack. That gives every hard-drop landing spot directly, without
simulating the fall. Placements that would stick out of the top are skipped,
since they end the game.

| Piece | I | O | T | S | Z | J | L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Distinct rotations | 2 | 1 | 4 | 2 | 2 | 4 | 4 |
| Placements on an empty board | 17 | 9 | 34 | 17 | 17 | 34 | 34 |

Moves are straight drops, the same as `Tetris.place(rot, x)` in the engine.
During development every simulated placement was checked against what the
engine actually does over a 300-piece game, with no mismatches. Because
moves are straight drops, the AI never slides pieces under overhangs or
performs T-spins.

### 3. Scoring a board

After simulating a placement (including any line clears), the AI measures
the result. The eight measurements follow classic Tetris AI research
(Dellacherie, El-Tetris):

| Feature | What it measures | Trained weight |
| --- | --- | --- |
| `landing_height` | how high the piece came to rest (the middle of the piece) | -12.66 |
| `eroded_cells` | lines cleared x cells of the piece that were cleared | +8.31 |
| `row_transitions` | filled/empty switches along each row (walls count as filled) | -4.55 |
| `col_transitions` | filled/empty switches down each column (the floor counts as filled) | -32.40 |
| `holes` | empty cells with something above them | -3.84 |
| `wells` | depth of one-wide wells, counted as 1 + 2 + ... + depth | -4.52 |
| `rows_with_holes` | rows containing at least one hole | -13.12 |
| `bumpiness` | sum of height differences between neighbouring columns | -2.28 |

The score of a board is the sum of each feature times its weight; higher is
better. A negative weight means "avoid this".

The learned weights tell the AI's strategy:

- **Holes are the enemy.** `col_transitions` is by far the strongest weight,
  and most column transitions are holes and overhangs. Together with
  `rows_with_holes` this covers holes so well that the plain `holes` count
  ended up with a small weight: the other two already punish them.
- **Stay low and flat.** `landing_height` keeps pieces near the floor, and
  `row_transitions`, `bumpiness` and `wells` keep the surface smooth.
- **Clear lines straight away.** `eroded_cells` is positive, so it takes any
  line it can. That makes it a survivor rather than a scorer: it almost never
  builds for a Tetris (see [Results](#6-results)).

### 4. Search

The AI looks at the current piece only. It does not use the next queue or
hold, even though the game shows both: with these features, one piece of
search is already enough to survive indefinitely. Each move takes about
**0.25 ms** in pure Python, including the engine placing the piece.

### 5. Learning the weights: the cross-entropy method

The weights in `weights.json` were not tuned by hand. They were learned with
the **noisy cross-entropy method** (Szita & Lőrincz, 2006), a simple
evolutionary search:

1. Start with a Gaussian over the 8 weights: mean 0, standard deviation 10
   for each.
2. **Sample** 40 weight vectors from it.
3. **Play:** each candidate plays the same 3 games of up to 400 pieces (same
   seeds, so the comparison is fair).
4. **Select** the best 20% by average lines cleared: the "elite".
5. **Refit** the Gaussian to the elite: the new mean is their average, the
   new spread their spread, plus a little extra noise that shrinks every
   generation (from 3.6 down to 0.1) so the search explores early on and
   settles later.
6. Repeat for 15 generations, writing the mean to `weights.json` after each
   one.

The reward is **lines cleared**. Games are capped at 400 pieces because a
good candidate never loses, and an uncapped game would never end. 400 pieces
is at most 160 lines, so once candidates reach about 159 they are all
surviving the whole game.

Training runs on all CPU cores and takes under a minute on two cores. This
is the run that produced the included weights, from the history saved in
`weights.json`:

| Generation | Population average | Elite average | Best |
| --- | --- | --- | --- |
| 1 | 4.8 | 23.8 | 146.3 |
| 2 | 28.9 | 104.5 | 155.3 |
| 3 | 68.5 | 155.3 | 158.3 |
| 4 | 131.9 | 158.1 | 159.0 |
| 5 | 149.7 | 158.5 | 159.0 |
| 10 | 157.8 | 158.6 | 158.7 |
| 15 | 158.0 | 158.9 | 159.0 |

Lines cleared per 400-piece game. In generation 1 a typical random AI clears
about 5 lines before topping out. By generation 4 nearly all of the
population survives the whole game.

### 6. Results

Measured on unseen games with the included weights:

| Test | Score | Lines | Lines from Tetrises | Deaths |
| --- | --- | --- | --- | --- |
| 20 games of 500 pieces | 254,000 | 198 | 0.5% | 0 of 20 |
| 3 games of 10,000 pieces | 92,900,000 | 3,998 | 0.7% | 0 of 3 |

A 500-piece game can clear at most 200 lines, and it averages 198, so it is
close to clearing every line it possibly could. The scores are high because
points are multiplied by the level, which reaches about 400 in the long games.

The low Tetris rate is the cost of training for lines. Because
`eroded_cells` rewards every clear, the AI takes singles and doubles as soon
as they appear. Rustris trained the same kind of AI for score instead, and
its weights learned to stack and save the I piece for Tetrises (89% of its
lines).

### 7. Watching it play

In `watch`, the AI plays through the same moves a human uses: it rotates
clockwise until the piece is in the target rotation, shifts left or right to
the target column, then hard drops. If a rotate or shift is blocked, it hard
drops where it is instead of getting stuck. The speed setting is the time
between those actions, from 0.25 s at speed 1 to no delay at speed 6. Gravity
is not applied in this mode, so the AI is never rushed.

### Training and measuring it yourself

```sh
# train from scratch (rewrites weights.json)
python main.py train
python main.py train --generations 25 --population 60 --games 5 --max-pieces 500

# lines and score over seeded games without a window
python main.py eval --games 5 --max-pieces 5000 --seed 1

# use a different weights file
python main.py --weights my_weights.json train
python main.py --weights my_weights.json watch
```

`train` also takes `--seed N` for a repeatable run and `--workers N` to set
how many CPU cores it uses.

### Limitations and ideas

- **No tucks or spins:** moves are straight drops, so the AI never slides a
  piece under an overhang and cannot do T-spins.
- **Plays for lines, not score:** training on score instead of lines, and
  adding features for Tetrises and small clears, teaches it to build for
  Tetrises. This is what Rustris did.
- **No look-ahead or hold:** judging each move by the best placement of the
  next piece, and considering the held piece, would make it stronger in hard
  positions and at high gravity.
