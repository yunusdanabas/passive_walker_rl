"""
Environment throughput benchmark.

Measures control steps per second of PassiveWalkerEnv in FSM mode and
(optionally) prints a cProfile breakdown. Used to track performance work
against the numbers recorded in docs/baseline.md.

Usage:
    python scripts/bench_env.py [--steps 3000] [--repeats 3] [--profile]
"""
import argparse
import cProfile
import pstats
import time

import numpy as np

from passive_walker.core.env import PassiveWalkerEnv


def bench(steps: int) -> float:
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    env.reset(seed=0)
    env.simend = 1e9
    action = np.zeros(3, dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(action)
    elapsed = time.perf_counter() - t0
    env.close()
    return steps / elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    rates = [bench(args.steps) for _ in range(args.repeats)]
    best = max(rates)
    print(f"control steps/s: best={best:,.0f}  median={np.median(rates):,.0f}  "
          f"({best * 10:,.0f} physics steps/s at 100 Hz control, 1 kHz physics)")

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        bench(args.steps)
        profiler.disable()
        pstats.Stats(profiler).sort_stats("tottime").print_stats(10)


if __name__ == "__main__":
    main()
