"""
FSM Data Collection

Collects FSM walking episodes for BC training.
Supports both duration-based and continuous collection modes.
"""
import os
import json
import time
import argparse
import numpy as np
from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.core.perturbations import create_perturbation_manager

# =============================================================================
# Configuration
# =============================================================================

# Collection defaults
DEFAULT_EPISODES = 80
DEFAULT_DURATION_SEC = 25.0
DEFAULT_OUTDIR = "data/fsm_runs"
DEFAULT_SEED = 123
DEFAULT_MODE = "fsm"

# Performance settings
PRINT_EVERY_SEC = 0.5
SAVE_META = True
COMPRESS_NPZ = True

# Early abort safety
EARLY_ABORT_THRESHOLD = 0.30  # 30% fall rate
EARLY_ABORT_EPISODES = 10     # Check after first 10 episodes

# Observation noise parameters
OBS_NOISE_POSITION_STD = 0.015  # ±1.5% noise on positions (x, z, angles)
OBS_NOISE_VELOCITY_STD = 0.025  # ±2.5% noise on velocities

# Duration presets for different gait cycle targets
DURATION_PRESETS = {
    "short": 15.0,    # ~6-8 cycles
    "medium": 20.0,   # ~8-10 cycles  
    "long": 25.0,     # ~10-12 cycles
    "extended": 30.0  # ~12-15 cycles
}

# Physics condition presets for structured diversity
PHYSICS_PRESETS = {
    # Baseline conditions
    "nominal": {"ramp_deg": 10.0, "friction": 0.9, "randomize": False},
    "gentle": {"ramp_deg": 8.0, "friction": 0.9, "randomize": False},
    "very_gentle": {"ramp_deg": 7.0, "friction": 0.9, "randomize": False},
    "moderate": {"ramp_deg": 9.0, "friction": 0.9, "randomize": False},
    
    # Friction variations
    "low_friction": {"ramp_deg": 10.0, "friction": 0.6, "randomize": False},
    "very_low_friction": {"ramp_deg": 10.0, "friction": 0.4, "randomize": False},
    "medium_friction": {"ramp_deg": 10.0, "friction": 0.7, "randomize": False},
    "high_friction": {"ramp_deg": 10.0, "friction": 1.0, "randomize": False},
    
    # Steep slopes
    "steep": {"ramp_deg": 12.0, "friction": 0.9, "randomize": False},
    "very_steep": {"ramp_deg": 13.0, "friction": 0.9, "randomize": False},
    "extreme_steep": {"ramp_deg": 14.0, "friction": 0.9, "randomize": False},
    
    # Combined conditions
    "gentle_high": {"ramp_deg": 8.0, "friction": 1.0, "randomize": False},
    "sweep_gentle_low": {"ramp_deg": 8.0, "friction": 0.6, "randomize": False},
    "steep_low_friction": {"ramp_deg": 12.0, "friction": 0.6, "randomize": False},
    "gentle_very_low": {"ramp_deg": 8.0, "friction": 0.4, "randomize": False},
    
    # Randomization
    "mass_jitter": {"ramp_deg": 10.0, "friction": 0.9, "randomize": True},
}


# =============================================================================
# Utility Functions
# =============================================================================

def _save_npz(path, payload, compress=True):
    """Save NPZ file with optional compression."""
    if compress:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def _add_observation_noise(obs, noise_level, rng):
    """
    Add Gaussian noise to observation for robustness.
    
    Args:
        obs: Observation array (11,) - [x, z, pitch, ẋ, ż, hip, lk, rk, hiṗ, lk̇, rk̇]
        noise_level: Noise level multiplier (0.0 = no noise, 1.0 = full noise)
        rng: Random number generator
        
    Returns:
        Noisy observation
    """
    if noise_level <= 0:
        return obs
    
    noisy_obs = obs.copy()
    
    # Position noise (indices 0-2 and 5-7: x, z, pitch, hip, lk, rk)
    position_indices = [0, 1, 2, 5, 6, 7]
    for i in position_indices:
        noise = rng.normal(0, OBS_NOISE_POSITION_STD * noise_level)
        noisy_obs[i] += noise * abs(obs[i]) if obs[i] != 0 else noise * 0.01
    
    # Velocity noise (indices 3-4 and 8-10: ẋ, ż, hiṗ, lk̇, rk̇)
    velocity_indices = [3, 4, 8, 9, 10]
    for i in velocity_indices:
        noise = rng.normal(0, OBS_NOISE_VELOCITY_STD * noise_level)
        noisy_obs[i] += noise * abs(obs[i]) if obs[i] != 0 else noise * 0.01
    
    return noisy_obs


def _get_descriptive_dirname(condition, physics_params):
    """Generate descriptive directory name from physics parameters."""
    ramp = int(physics_params["ramp_deg"])
    friction = physics_params["friction"]
    randomize = physics_params["randomize"]
    
    if randomize:
        return f"slope{ramp}_fric{friction:.1f}_jitter5"
    else:
        return f"slope{ramp}_fric{friction:.1f}_nomass"


def _create_metadata(episodes, duration_sec, steps, env, physics_condition, physics, seed_base=None, obs_noise=0.0):
    """Create collection metadata dictionary."""
    return {
            "episodes": int(episodes),
        "target_duration_sec": float(duration_sec),
        "target_steps": int(steps),
        "ctrl_hz": float(env.ctrl_hz),
        "seed_base": None if seed_base is None else int(seed_base),
            "env": "PassiveWalkerEnv",
        "mode": env.mode,
            "pd_backend": env.pd.backend_name,
        "design": "duration_based",
        "gait_cycle_target": "8-10+ per episode",
        "physics_condition": physics_condition or "nominal",
        "physics_params": {
            "ramp_deg": float(physics["ramp_deg"]),
            "friction": float(physics["friction"]),
            "randomize_physics": bool(physics["randomize"])
        },
        "observation_noise": float(obs_noise),
            "schema": {
                "obs": "(T+1,11)",
                "act": "(T,3)",
                "rew": "(T,)",
                "done": "(T,)",
                "info_pitch": "(T,)",
                "info_torso_z": "(T,)",
                "info_dx": "(T,)",
            "info_qdes": "(T,3)",
            "info_fsm_hip": "(T,)",
            "info_fsm_k1": "(T,)",
            "info_fsm_k2": "(T,)",
            "info_u_abs_sum": "(T,)"
        },
    }


def _create_quality_report(episode_lengths, episode_gait_cycles, episode_distances, 
                          episode_fell_flags, episode_pitch_max, steps):
    """Create quality analysis report."""
    if not episode_lengths:
        return None
    
    # Calculate quality flags
    short_episodes = sum(1 for l in episode_lengths if l < steps * 0.8)
    low_cycles = sum(1 for c in episode_gait_cycles if c < 6)
    high_pitch = sum(1 for p in episode_pitch_max if p > 1.0)
    
    return {
        "episode_count": len(episode_lengths),
        "length_stats": {
            "mean": float(np.mean(episode_lengths)),
            "median": float(np.median(episode_lengths)),
            "std": float(np.std(episode_lengths)),
            "min": int(np.min(episode_lengths)),
            "max": int(np.max(episode_lengths))
        },
        "gait_cycle_stats": {
            "mean": float(np.mean(episode_gait_cycles)),
            "min": int(np.min(episode_gait_cycles)),
            "max": int(np.max(episode_gait_cycles))
        },
        "distance_stats": {
            "mean": float(np.mean(episode_distances)),
            "max": float(np.max(episode_distances))
        },
        "quality_flags": {
            "fall_rate": float(np.mean(episode_fell_flags)),
            "max_pitch": float(np.max(episode_pitch_max)),
            "short_episodes": short_episodes,
            "low_cycle_episodes": low_cycles,
            "high_pitch_episodes": high_pitch
        }
    }


# =============================================================================
# Main Collection Functions
# =============================================================================

def collect(episodes, duration_sec, outdir, seed=None, physics_condition=None, mode="fsm", obs_noise=0.0, randomization_profile=None, perturbation_mode="none", perturbation_strength=1.0, perturbation_freq=0.5):
    """Collect FSM episodes with duration-based design and gait cycle validation."""
    # Setup
    os.makedirs(outdir, exist_ok=True)
    
    # Create RNG for observation noise
    noise_rng = np.random.RandomState(seed + 9999 if seed is not None else None) if obs_noise > 0 else None

    # Get physics parameters
    if physics_condition and physics_condition in PHYSICS_PRESETS:
        physics = PHYSICS_PRESETS[physics_condition]
        print(f"Physics condition: {physics_condition} (ramp={physics['ramp_deg']}°, friction={physics['friction']}, randomize={physics['randomize']})")
    else:
        physics = PHYSICS_PRESETS["nominal"]
        print(f"Using nominal physics (ramp={physics['ramp_deg']}°, friction={physics['friction']})")

    # Create environment
    env = PassiveWalkerEnv(
        mode=mode, 
        use_gui=False,
        ramp_deg=physics["ramp_deg"],
        friction=physics["friction"],
        randomize_physics=physics["randomize"],
        randomization_profile=randomization_profile
    )
    
    # Initialize perturbation manager
    perturbation_manager = create_perturbation_manager(
        mode=perturbation_mode,
        strength=perturbation_strength
    )
    
    if perturbation_mode != "none":
        print(f"Perturbation mode: {perturbation_mode}, strength: {perturbation_strength}, freq: {perturbation_freq} Hz")
    
    # Calculate steps based on environment control rate
    steps = int(round(duration_sec * env.ctrl_hz))
    print(f"Target duration: {duration_sec:.1f}s → {steps} steps at {env.ctrl_hz:.0f} Hz")
    
    # Ensure 'done' only means FALL; let the step cap end the episode
    env.simend = 1e9  # huge horizon so only falls set done=True

    # Save metadata
    if SAVE_META:
        meta = _create_metadata(episodes, duration_sec, steps, env, physics_condition, physics, seed, obs_noise)
        with open(os.path.join(outdir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # Quality tracking
    episode_lengths = []
    episode_gait_cycles = []
    episode_distances = []
    episode_fell_flags = []  # Per-episode boolean fall flags
    episode_pitch_max = []

    # Collect episodes
    for ep in range(episodes):
        # Reset with deterministic seed
        ep_seed = None if seed is None else (int(seed) + ep)
        obs, _ = env.reset(seed=ep_seed)
        
        # Reset perturbation manager for new episode
        if perturbation_mode != "none":
            perturbation_manager.reset()

        # Pre-allocate episode buffers
        T = int(steps)
        obs_buf = np.empty((T + 1, 17), dtype=np.float32)  # Updated for 17D observations
        obs_buf[0] = obs
        act_buf = np.zeros((T, 3), dtype=np.float32)
        rew_buf = np.empty((T,), dtype=np.float32)
        done_buf = np.empty((T,), dtype=np.bool_)
        ipitch = np.empty((T,), dtype=np.float32)
        itorso = np.empty((T,), dtype=np.float32)
        idx = np.empty((T,), dtype=np.float32)
        qdes_buf = np.empty((T, 3), dtype=np.float32)
        
        # Perturbation tracking arrays
        perturbation_applied = np.zeros((T,), dtype=np.bool_)
        perturbation_type = np.zeros((T,), dtype=np.int32)
        
        # FSM states and control effort for QA
        fsm_hip_arr = np.empty((T,), dtype=np.int32)
        fsm_k1_arr = np.empty((T,), dtype=np.int32)
        fsm_k2_arr = np.empty((T,), dtype=np.int32)
        uabs_arr = np.empty((T,), dtype=np.float32)

        # Track gait cycles via hip state changes
        gait_cycles = 0
        last_hip_state = None
        hip_state_changes = 0
        
        # Track fall for this episode
        fell_this_ep = False

        # Episode loop
        last_print = time.time()
        t_used = 0

        for t in range(T):
            # Apply perturbations if enabled
            if perturbation_mode != "none":
                perturbation_info = perturbation_manager.update(env, env.data.time)
                if perturbation_info['perturbations_applied']:
                    perturbation_applied[t] = True
                    # Store perturbation type (simplified: 1 for any perturbation)
                    perturbation_type[t] = 1
            
            # Get FSM joint targets and step environment
            if env.mode == "fsm":
                # In FSM mode the env ignores the action; send zeros for clarity/speed
                action = np.zeros(3, dtype=np.float32)
            elif env.mode == "hybrid_hip":
                # Hybrid mode: NN controls hip, FSM controls knees
                # Send zeros for hip (will be overridden by FSM), FSM handles knees
                action = np.zeros(3, dtype=np.float32)
            elif env.mode == "hybrid_knees":
                # Hybrid mode: FSM controls hip, NN controls knees
                # Send zeros for knees (will be overridden by FSM), FSM handles hip
                action = np.zeros(3, dtype=np.float32)
            else:
                # If you *really* want to run non-FSM collection, you must send normalized [-1,1] actions
                # (Consider adding PDController.norm() and using that here.)
                raise RuntimeError("Collector currently supports mode='fsm', 'hybrid_hip', 'hybrid_knees' only for safe demos.")
            obs, r, done, info = env.step(action)
            
            # Add observation noise for robustness if enabled
            if obs_noise > 0:
                obs = _add_observation_noise(obs, obs_noise, noise_rng)

            # Count gait cycles via hip state transitions
            current_hip_state = env.fsm.fsm_hip
            if last_hip_state is not None and current_hip_state != last_hip_state:
                hip_state_changes += 1
                gait_cycles = hip_state_changes // 2  # Complete cycle = 2 transitions
            last_hip_state = current_hip_state

            # Store step data
            obs_buf[t + 1] = obs
            act_buf[t] = action
            rew_buf[t] = r
            done_buf[t] = done
            ipitch[t] = info.get("pitch_abs", 0.0)
            itorso[t] = info.get("torso_z", 0.0)
            idx[t] = info.get("dx", 0.0)
            qdes_buf[t] = info.get("qdes", [0.0, 0.0, 0.0])
            
            # Store FSM states and control effort for QA
            fsm_hip_arr[t] = info.get("fsm_hip", -1)
            fsm_k1_arr[t] = info.get("fsm_k1", -1)
            fsm_k2_arr[t] = info.get("fsm_k2", -1)
            uabs_arr[t] = info.get("u_abs_sum", 0.0)
            
            # Track fall for this episode
            fell_this_ep = fell_this_ep or bool(info.get("fell", False))
            
            t_used = t + 1

            # Progress monitoring
            now = time.time()
            if (now - last_print) >= PRINT_EVERY_SEC:
                print(f"[ep {ep+1}/{episodes}] t={t_used:4d}/{T} "
                      f"dx={idx[t]:+.4f} pitch|rad|={ipitch[t]:.3f} "
                      f"torso_z={itorso[t]:.3f} cycles={gait_cycles} done={bool(done)}")
                last_print = now

            if done:
                break

        # Validate gait cycle count
        actual_duration = t_used / env.ctrl_hz
        if gait_cycles < 8:
            print(f"WARNING: Episode {ep+1} only has {gait_cycles} gait cycles "
                  f"(target: 8-10+). Consider longer duration.")
        
        # Quality tracking for this episode
        episode_lengths.append(t_used)
        episode_gait_cycles.append(gait_cycles)
        episode_distances.append(float(np.sum(idx[:t_used])))  # Sum dx for explicit distance
        episode_fell_flags.append(float(fell_this_ep))  # Per-episode boolean fall flag
        episode_pitch_max.append(float(np.max(ipitch[:t_used])) if t_used > 0 else 0.0)
        
        # Save episode data
        payload = dict(
            obs=obs_buf[:t_used + 1],
            act=act_buf[:t_used],
            rew=rew_buf[:t_used],
            done=done_buf[:t_used],
            info_pitch=ipitch[:t_used],
            info_torso_z=itorso[:t_used],
            info_dx=idx[:t_used],
            info_qdes=qdes_buf[:t_used],
            info_fsm_hip=fsm_hip_arr[:t_used],
            info_fsm_k1=fsm_k1_arr[:t_used],
            info_fsm_k2=fsm_k2_arr[:t_used],
            info_u_abs_sum=uabs_arr[:t_used],
            **({} if ep_seed is None else {"seed": np.int64(ep_seed)}),
        )
        
        # Add perturbation data if enabled
        if perturbation_mode != "none":
            payload["info_perturbations"] = perturbation_applied[:t_used]
            payload["info_perturbation_type"] = perturbation_type[:t_used]
        
        # Add realized physics for audit trail
        if physics["randomize"]:
            payload["torso_mass"] = float(env.model.body_mass[env.b_torso])
        else:
            payload["ramp_deg"] = float(env.ramp_deg)
            payload["friction"] = float(env.friction)
        out_path = os.path.join(outdir, f"episode_{ep:06d}.npz")
        _save_npz(out_path, payload, compress=COMPRESS_NPZ)
        print(f"saved: {out_path} (T={t_used}, {actual_duration:.1f}s, {gait_cycles} cycles)")
        
        # Early abort check for high fall rate
        if (ep + 1) == EARLY_ABORT_EPISODES and len(episode_fell_flags) >= EARLY_ABORT_EPISODES:
            avg_fall = float(np.mean(episode_fell_flags))
            if avg_fall > EARLY_ABORT_THRESHOLD:
                print(f"\nEARLY ABORT: High fall rate detected!")
                print(f"Fall rate: {avg_fall:.1%} (threshold: {EARLY_ABORT_THRESHOLD:.1%})")
                print(f"Stopping collection after {ep + 1} episodes to prevent bad dataset.")
                print("Check physics parameters or FSM configuration.")
                env.close()
                return

    # Quality analysis and reporting
    if episode_lengths:
        print(f"\n=== QUALITY ANALYSIS ===")
        print(f"Episode length stats: mean={np.mean(episode_lengths):.1f}, median={np.median(episode_lengths):.1f}")
        print(f"Gait cycles stats: mean={np.mean(episode_gait_cycles):.1f}, min={np.min(episode_gait_cycles)}")
        print(f"Distance stats: mean={np.mean(episode_distances):.1f}m, max={np.max(episode_distances):.1f}m")
        print(f"Fall rate: {np.mean(episode_fell_flags):.1%} ({np.sum(episode_fell_flags)}/{len(episode_fell_flags)} episodes)")
        print(f"Max pitch: {np.max(episode_pitch_max):.3f} rad")
        
        # Quality warnings
        short_episodes = sum(1 for l in episode_lengths if l < steps * 0.8)
        low_cycles = sum(1 for c in episode_gait_cycles if c < 6)
        high_pitch = sum(1 for p in episode_pitch_max if p > 1.0)
        
        if short_episodes > 0:
            print(f"WARNING: {short_episodes} episodes shorter than 80% of target duration")
        if low_cycles > 0:
            print(f"WARNING: {low_cycles} episodes with <6 gait cycles")
        if high_pitch > 0:
            print(f"WARNING: {high_pitch} episodes with max pitch >1.0 rad")
        
        # Save quality report
        quality_report = _create_quality_report(
            episode_lengths, episode_gait_cycles, episode_distances, 
            episode_fell_flags, episode_pitch_max, steps
        )
        
        if quality_report:
            with open(os.path.join(outdir, "README.json"), "w") as f:
                json.dump(quality_report, f, indent=2)
            print(f"Quality report saved to: {os.path.join(outdir, 'README.json')}")

    env.close()


def collect_physics_sweep(episodes_per_condition, duration_sec, base_outdir, seed=None, conditions=None, mode="fsm", obs_noise=0.0, randomization_profile=None):
    """Collect FSM data across multiple physics conditions with organized directory structure."""
    if conditions is None:
        # Default: comprehensive sweep across major variations
        conditions = [
            "nominal", "gentle", "steep",  # Ramp variations
            "low_friction", "high_friction",  # Friction variations
            "sweep_gentle_low", "steep_low_friction",  # Combined extremes
            "mass_jitter"  # Randomization
        ]
    
    print(f"Physics Diversity Collection - {len(conditions)} conditions")
    print(f"Episodes per condition: {episodes_per_condition}")
    print(f"Duration per episode: {duration_sec:.1f}s")
    if obs_noise > 0:
        print(f"Observation noise: {obs_noise:.2f}")
    if randomization_profile:
        print(f"Randomization profile: {randomization_profile}")
    
    for i, condition in enumerate(conditions):
        print(f"\n--- Condition {i+1}/{len(conditions)}: {condition} ---")
        
        # Get physics parameters for descriptive naming
        physics = PHYSICS_PRESETS[condition]
        dirname = _get_descriptive_dirname(condition, physics)
        condition_dir = os.path.join(base_outdir, dirname)
        
        # Collect data for this condition
        collect(episodes_per_condition, duration_sec, condition_dir, seed, condition, mode, obs_noise, randomization_profile, "none", 1.0, 0.5)
        
        print(f"Completed {condition}: {episodes_per_condition} episodes in {condition_dir}")
    
    print(f"\nPhysics sweep complete! Data saved in {base_outdir}/")
    print("Directory structure:")
    for condition in conditions:
        physics = PHYSICS_PRESETS[condition]
        dirname = _get_descriptive_dirname(condition, physics)
        print(f"  {dirname}/ - {episodes_per_condition} episodes")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI entry point for FSM data collection."""
    parser = argparse.ArgumentParser("FSM data collection (duration-based)")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="Number of episodes to collect")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SEC,
                        help="Episode duration in seconds")
    parser.add_argument("--mode", type=str, default="fsm",
                        choices=["fsm", "research", "hybrid_hip", "hybrid_knees"],
                        help="Control mode")
    parser.add_argument("--preset", type=str, choices=list(DURATION_PRESETS.keys()),
                        help=f"Duration preset: {', '.join(DURATION_PRESETS.keys())}")
    parser.add_argument("--physics", type=str, choices=list(PHYSICS_PRESETS.keys()),
                        help=f"Physics condition: {', '.join(PHYSICS_PRESETS.keys())}")
    parser.add_argument("--physics-sweep", action="store_true",
                        help="Collect data across multiple physics conditions")
    parser.add_argument("--out", type=str, default=DEFAULT_OUTDIR,
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed")
    parser.add_argument("--obs-noise", type=float, default=0.0,
                        help="Observation noise level (0.0=no noise, 1.0=full noise)")
    parser.add_argument("--randomization-profile", type=str, choices=["none", "basic", "moderate", "aggressive", "temporal"],
                        help="Advanced randomization profile for data collection")
    parser.add_argument("--perturbation-mode", type=str, default="none", 
                        choices=["none", "random", "scheduled", "curriculum"],
                        help="Perturbation mode: none, random, scheduled, or curriculum")
    parser.add_argument("--perturbation-strength", type=float, default=1.0,
                        help="Perturbation strength multiplier (0.0 to 1.0)")
    parser.add_argument("--perturbation-freq", type=float, default=0.5,
                        help="Perturbation frequency in Hz")
    args = parser.parse_args()
    
    # Use preset if specified, otherwise use duration argument
    duration = DURATION_PRESETS[args.preset] if args.preset else args.duration
    
    if args.physics_sweep:
        # Physics diversity collection
        print(f"FSM Physics Diversity Collection")
        print(f"Episodes per condition: {args.episodes}, Duration: {duration:.1f}s")
        collect_physics_sweep(args.episodes, duration, args.out, args.seed, mode=args.mode, obs_noise=args.obs_noise, randomization_profile=args.randomization_profile)
    else:
        # Single condition collection
        print(f"FSM Data Collection - Duration-based Design")
        print(f"Episodes: {args.episodes}, Duration: {duration:.1f}s")
        print(f"Expected gait cycles: ~{int(duration * 0.4):.0f}-{int(duration * 0.5):.0f} (8-10+ recommended)")
        if args.obs_noise > 0:
            print(f"Observation noise: {args.obs_noise:.2f}")
        
        collect(args.episodes, duration, args.out, args.seed, args.physics, args.mode, args.obs_noise, args.randomization_profile, args.perturbation_mode, args.perturbation_strength, args.perturbation_freq)


if __name__ == "__main__":
    main()