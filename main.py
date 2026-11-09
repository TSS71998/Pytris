"""Pytris: Tetris plus an AI that learns to play it.

    python main.py play                 play Tetris yourself
    python main.py train                train the AI (saves weights.json)
    python main.py watch                watch the trained AI play
    python main.py eval                 benchmark the AI without graphics
"""

import argparse
import os
import random
import sys

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.json")


def cmd_play(args):
    from pytris.gui import run_human
    run_human()


def cmd_train(args):
    from pytris.agent import FEATURE_NAMES, train
    print(f"Training for {args.generations} generations "
          f"({args.population} candidates x {args.games} games, "
          f"up to {args.max_pieces} pieces each)...")
    print("Each line: lines cleared per game by the best candidate, the elite "
          "and the whole population.\n")
    mean, _ = train(generations=args.generations, population=args.population,
                    games=args.games, max_pieces=args.max_pieces,
                    workers=args.workers, out=args.weights, seed=args.seed)
    print(f"\nSaved learned weights to {args.weights}:")
    for name, w in zip(FEATURE_NAMES, mean):
        print(f"  {name:16s} {w:8.3f}")
    print("\nWatch it play with:  python main.py watch")


def _load_agent(path, allow_random):
    from pytris.agent import Agent, N_FEATURES
    if os.path.exists(path):
        return Agent.load(path), os.path.basename(path)
    if not allow_random:
        sys.exit(f"No weights at {path}. Run `python main.py train` first.")
    print(f"No weights at {path}; using random untrained weights.")
    return Agent([random.gauss(0, 10) for _ in range(N_FEATURES)]), "random (untrained)"


def cmd_watch(args):
    from pytris.gui import run_ai
    agent, label = _load_agent(args.weights, allow_random=True)
    run_ai(agent, seed=args.seed, speed=args.speed - 1, label=label)


def cmd_eval(args):
    import time
    from pytris.game import Tetris
    agent, _ = _load_agent(args.weights, allow_random=False)
    rng = random.Random(args.seed)
    results = []
    for i in range(args.games):
        game = Tetris(seed=rng.randrange(1 << 30))
        t0 = time.time()
        while not game.game_over and game.pieces < args.max_pieces:
            mv = agent.choose(game)
            if mv is None:
                break
            game.place(*mv)
        results.append(game.lines)
        end = "game over" if game.game_over else f"stopped at {args.max_pieces} pieces"
        print(f"game {i + 1}: {game.lines:6d} lines, score {game.score:,} "
              f"({end}, {time.time() - t0:.1f}s)")
    print(f"\naverage {sum(results) / len(results):.1f} lines over {len(results)} games")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="weights file")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("play", help="play Tetris yourself")

    t = sub.add_parser("train", help="train the AI")
    t.add_argument("--generations", type=int, default=15)
    t.add_argument("--population", type=int, default=40)
    t.add_argument("--games", type=int, default=3, help="games per candidate")
    t.add_argument("--max-pieces", type=int, default=400,
                   help="cap per training game (good players never lose)")
    t.add_argument("--workers", type=int, default=None, help="CPU processes")
    t.add_argument("--seed", type=int, default=None)

    w = sub.add_parser("watch", help="watch the AI play")
    w.add_argument("--speed", type=int, default=3, help="1 (slow) to 6 (instant)")
    w.add_argument("--seed", type=int, default=None)

    e = sub.add_parser("eval", help="benchmark the AI without graphics")
    e.add_argument("--games", type=int, default=5)
    e.add_argument("--max-pieces", type=int, default=5000)
    e.add_argument("--seed", type=int, default=None)

    args = p.parse_args()
    commands = {"play": cmd_play, "train": cmd_train, "watch": cmd_watch, "eval": cmd_eval}
    commands[args.cmd or "play"](args)


if __name__ == "__main__":
    main()