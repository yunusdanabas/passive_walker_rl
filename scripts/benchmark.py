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

# JAX imports for PD benchmark
try:
    import jax
    import jax.numpy as jnp
    from passive_walker.jax.controller_jax import pd_step_broadcast_jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


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


def benchmark_pd_controller(batch_sizes=[1, 64, 1024], iterations=10000):
    """Benchmark JAX vs NumPy PD controller."""
    if not JAX_AVAILABLE:
        print("JAX not available, skipping PD benchmark")
        return
    
    print("PD Controller Benchmark (JAX vs NumPy)")
    print("=" * 50)
    print("Batch Size    NumPy (ms)    JAX (ms)    Speedup")
    print("-" * 50)
    
    # PD parameters
    kp = np.array([5.0, 1000.0, 1000.0], dtype=np.float32)
    kv = np.array([1.0, 100.0, 100.0], dtype=np.float32)
    umin = np.array([-50.0, -800.0, -800.0], dtype=np.float32)
    umax = np.array([50.0, 800.0, 800.0], dtype=np.float32)
    
    for batch_size in batch_sizes:
        # Generate random data
        np.random.seed(42)
        q = np.random.randn(batch_size, 3).astype(np.float32)
        qd = np.random.randn(batch_size, 3).astype(np.float32)
        q_des = np.random.randn(batch_size, 3).astype(np.float32)
        
        # NumPy benchmark
        start_time = time.time()
        for _ in range(iterations):
            u_np = kp[None, :] * (q_des - q) - kv[None, :] * qd
            u_np = np.clip(u_np, umin[None, :], umax[None, :])
        numpy_time = (time.time() - start_time) * 1000  # ms
        
        # JAX benchmark (with JIT compilation)
        q_jax = jnp.array(q)
        qd_jax = jnp.array(qd)
        q_des_jax = jnp.array(q_des)
        kp_jax = jnp.array(kp)
        kv_jax = jnp.array(kv)
        umin_jax = jnp.array(umin)
        umax_jax = jnp.array(umax)
        
        # Warmup JIT
        _ = pd_step_broadcast_jit(q_jax, qd_jax, q_des_jax, kp_jax, kv_jax, umin_jax, umax_jax)
        
        start_time = time.time()
        for _ in range(iterations):
            u_jax = pd_step_broadcast_jit(q_jax, qd_jax, q_des_jax, kp_jax, kv_jax, umin_jax, umax_jax)
        jax_time = (time.time() - start_time) * 1000  # ms
        
        speedup = numpy_time / jax_time if jax_time > 0 else float('inf')
        print(f"{batch_size:8d}    {numpy_time:8.2f}    {jax_time:8.2f}    {speedup:6.2f}x")


def benchmark_vecenv():
    """Benchmark vectorized environment performance."""
    from passive_walker.core.io import load_walker_config
    from passive_walker.core.env import PassiveWalkerEnv
    from passive_walker.core.vec import NumpySubprocVecEnv
    import time
    
    cfg = load_walker_config("passive_walker/fsm/fsm_collect.yaml")
    
    def make():
        return PassiveWalkerEnv(cfg, use_gui=False)
    
    print("VecEnv Benchmark")
    print("=" * 30)
    print("Batch Size    Steps/sec")
    print("-" * 30)
    
    for batch_size in [1, 2, 4, 8]:
        try:
            vec = NumpySubprocVecEnv([make for _ in range(batch_size)])
            obs, _ = vec.reset(seed=0)
            acts = np.zeros((batch_size, 3), dtype=np.float32)
            
            t0 = time.time()
            steps = 0
            while time.time() - t0 < 1.0:  # Run for 1 second
                vec.step(acts)
                steps += 1
            
            steps_per_sec = steps * batch_size
            print(f"{batch_size:8d}    {steps_per_sec:8d}")
            vec.close()
        except Exception as e:
            print(f"{batch_size:8d}    ERROR: {e}")


def main():
    """Run performance benchmarks."""
    print("Passive Walker Performance Benchmark")
    print("=" * 40)
    
    # Run PD controller benchmark first
    benchmark_pd_controller()
    print()
    
    # Run VecEnv benchmark
    benchmark_vecenv()
    print()

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
                f"{r['mode']:8s} {r['preset']:11s} {str(r['jax_enable']):5s} {r['steps_per_sec']:8.1f}"
            )


if __name__ == "__main__":
    main()
