#!/usr/bin/env python3
"""
Performance benchmark script for passive walker environment.
"""

import time
import numpy as np
import csv
from pathlib import Path
from passive_walker.core.io import load_walker_config
from passive_walker.core.env import PassiveWalkerEnv


def benchmark_config(mode, preset, jax_enable, duration_seconds=60):
    """Benchmark a specific configuration."""
    print(f"Benchmarking: mode={mode}, preset={preset}, jax={jax_enable}")

    # Load config
    cfg = load_walker_config("passive_walker/configs/walker.yaml")
    cfg.mode = mode
    cfg.jax.enable = jax_enable
    if mode == "research":
        cfg.reward.preset = preset

    # Create environment
    env = PassiveWalkerEnv(cfg, use_gui=False)

    # Warmup
    obs, _ = env.reset()
    for _ in range(10):
        obs, _, done, _ = env.step(np.zeros(3, dtype=np.float32))
        if done:
            obs, _ = env.reset()

    # Benchmark
    start_time = time.time()
    steps = 0
    episodes = 0

    while time.time() - start_time < duration_seconds:
        obs, _, done, _ = env.step(np.zeros(3, dtype=np.float32))
        steps += 1
        if done:
            episodes += 1
            obs, _ = env.reset()

    elapsed = time.time() - start_time
    steps_per_sec = steps / elapsed
    episodes_per_sec = episodes / elapsed

    env.close()

    print(f"  Steps: {steps}, Episodes: {episodes}")
    print(f"  Steps/sec: {steps_per_sec:.1f}, Episodes/sec: {episodes_per_sec:.1f}")

    return {
        "mode": mode,
        "preset": preset,
        "jax_enable": jax_enable,
        "steps": steps,
        "episodes": episodes,
        "duration": elapsed,
        "steps_per_sec": steps_per_sec,
        "episodes_per_sec": episodes_per_sec,
    }


def main():
    """Run performance benchmarks."""
    print("Passive Walker Performance Benchmark")
    print("=" * 40)

    # Create results directory
    results_dir = Path("results/bench")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Benchmark configurations
    configs = [
        ("fsm", "minimal", False),
        ("fsm", "minimal", True),
        ("research", "minimal", False),
        ("research", "minimal", True),
        ("research", "default", False),
        ("research", "default", True),
        ("research", "aggressive", False),
        ("research", "aggressive", True),
    ]

    results = []

    for mode, preset, jax_enable in configs:
        try:
            result = benchmark_config(mode, preset, jax_enable, duration_seconds=30)
            results.append(result)
        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Save results
    csv_path = results_dir / "bench.csv"
    with open(csv_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\nResults saved to: {csv_path}")

    # Print summary
    if results:
        print("\nSummary:")
        print("Mode      Preset      JAX    Steps/sec")
        print("-" * 40)
        for r in results:
            print(
                f"{r['mode']:8s} {r['preset']:11s} {r['jax_enable']:5s} {r['steps_per_sec']:8.1f}"
            )


if __name__ == "__main__":
    main()
