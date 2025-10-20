#!/usr/bin/env python3
"""
Compare model vs FSM frame-by-frame to identify exact divergence point.
Helps understand where and why the model fails vs FSM baseline.
"""
import argparse
import json
import numpy as np
import time
from pathlib import Path

def compare_model_vs_fsm(ckpt_path: str, meta_path: str, duration_sec: float = 5.0):
    """
    Run both FSM baseline and model inference, comparing outputs frame-by-frame.
    
    Args:
        ckpt_path: Path to model checkpoint
        meta_path: Path to model metadata
        duration_sec: Duration to run comparison (seconds)
    """
    import torch
    from passive_walker.bc.models.models_torch import TorchMLP, TorchMLPLarge
    from passive_walker.bc.utils import load_checkpoint, Normalizer
    from passive_walker.core.env import PassiveWalkerEnv
    from passive_walker.core.controller import FSMStateMachine
    
    print(f"🔄 Starting model vs FSM comparison")
    print(f"⏱️  Duration: {duration_sec} seconds")
    
    # Load model
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    try:
        model = TorchMLPLarge(in_dim=meta["input_dim"], out_dim=meta["output_dim"], hidden=512, dropout=0.1)
        model = load_checkpoint(ckpt_path, model)
    except Exception:
        model = TorchMLP(in_dim=meta["input_dim"], out_dim=meta["output_dim"])
        model = load_checkpoint(ckpt_path, model)
    
    model.eval()
    
    # Setup normalizer
    normalizer = Normalizer(
        mean=np.array(meta["normalizer_mean"], dtype=np.float32),
        std=np.array(meta["normalizer_std"], dtype=np.float32)
    )
    
    # Setup environments
    if meta["section"] == "hip":
        fsm_mode = "fsm"
        model_mode = "hybrid_hip"
    elif meta["section"] == "knees":
        fsm_mode = "fsm" 
        model_mode = "hybrid_knees"
    else:
        fsm_mode = "fsm"
        model_mode = "research"
    
    print(f"🎛️  FSM mode: {fsm_mode}, Model mode: {model_mode}")
    
    # Create environments (headless for speed)
    fsm_env = PassiveWalkerEnv(mode=fsm_mode, use_gui=False)
    model_env = PassiveWalkerEnv(mode=model_mode, use_gui=False)
    
    # Reset both environments with same seed
    seed = 42
    fsm_obs, _ = fsm_env.reset(seed=seed)
    model_obs, _ = model_env.reset(seed=seed)
    
    # Data collection
    timesteps = []
    fsm_data = {
        "qdes": [],
        "qpos": [],
        "qvel": [],
        "rewards": [],
        "dx": [],
        "pitch": []
    }
    model_data = {
        "raw_outputs": [],
        "qdes": [],
        "qpos": [],
        "qvel": [],
        "rewards": [],
        "dx": [],
        "pitch": []
    }
    
    max_steps = int(duration_sec * fsm_env.ctrl_hz)
    
    print(f"🏃 Running comparison for {max_steps} steps...")
    
    # Run FSM baseline
    print(f"\n📊 Running FSM baseline...")
    for step in range(max_steps):
        # FSM step (ignores action)
        fsm_obs, fsm_reward, fsm_done, fsm_info = fsm_env.step(np.zeros(3))
        
        fsm_qdes = fsm_info.get("qdes", [0, 0, 0])
        fsm_qpos = [fsm_env.data.qpos[fsm_env.qpos_hip], 
                   fsm_env.data.qpos[fsm_env.qpos_lk],
                   fsm_env.data.qpos[fsm_env.qpos_rk]]
        
        fsm_data["qdes"].append(fsm_qdes)
        fsm_data["qpos"].append(fsm_qpos)
        fsm_data["rewards"].append(fsm_reward)
        fsm_data["dx"].append(fsm_info.get("dx", 0))
        fsm_data["pitch"].append(fsm_info.get("pitch_abs", 0))
        
        if fsm_done:
            print(f"FSM fell at step {step}")
            break
    
    # Reset for model run
    model_obs, _ = model_env.reset(seed=seed)
    
    # Run model inference
    print(f"\n🧠 Running model inference...")
    fsm_for_comparison = FSMStateMachine()
    fsm_for_comparison.bind_indices(
        qpos_hip=model_env.qpos_hip, qpos_lk=model_env.qpos_lk, qpos_rk=model_env.qpos_rk,
        qvel_hip=model_env.qvel_hip, qvel_lk=model_env.qvel_lk, qvel_rk=model_env.qvel_rk,
        b_lfoot=model_env.b_lfoot, b_rfoot=model_env.b_rfoot,
        b_lleg=model_env.b_lleg, b_rleg=model_env.b_rleg
    )
    
    model_fell_step = None
    
    for step in range(max_steps):
        # Get model prediction
        x = normalizer.apply(model_obs[None, :]).astype(np.float32)
        with torch.no_grad():
            model_raw = model(torch.tensor(x, dtype=torch.float32))[0].numpy()
        
        # Assemble action based on section
        if meta["section"] == "hip":
            action = np.array([float(model_raw[0]), 0.0, 0.0], dtype=np.float32)
        elif meta["section"] == "knees":
            action = np.array([float(model_raw[0]), float(model_raw[1]), 0.0], dtype=np.float32)
        else:  # both
            action = np.array([float(model_raw[0]), float(model_raw[1]), float(model_raw[2])], dtype=np.float32)
        
        # Model step
        model_obs, model_reward, model_done, model_info = model_env.step(action)
        
        model_qdes = model_info.get("qdes", [0, 0, 0])
        model_qpos = [model_env.data.qpos[model_env.qpos_hip],
                     model_env.data.qpos[model_env.qpos_lk], 
                     model_env.data.qpos[model_env.qpos_rk]]
        
        model_data["raw_outputs"].append(model_raw)
        model_data["qdes"].append(model_qdes)
        model_data["qpos"].append(model_qpos)
        model_data["rewards"].append(model_reward)
        model_data["dx"].append(model_info.get("dx", 0))
        model_data["pitch"].append(model_info.get("pitch_abs", 0))
        
        timesteps.append(step)
        
        if model_done and model_fell_step is None:
            model_fell_step = step
            print(f"Model fell at step {step}")
            break
    
    # Close environments
    fsm_env.close()
    model_env.close()
    
    # Analysis
    print(f"\n📈 ANALYSIS RESULTS:")
    print(f"=" * 50)
    
    print(f"FSM run: {len(fsm_data['qdes'])} steps")
    print(f"Model run: {len(model_data['raw_outputs'])} steps")
    if model_fell_step:
        print(f"Model fell at step: {model_fell_step}")
    
    # Convert to numpy arrays
    for key in fsm_data:
        fsm_data[key] = np.array(fsm_data[key])
    for key in model_data:
        model_data[key] = np.array(model_data[key])
    
    # Compare joint targets
    print(f"\n🎯 JOINT TARGET COMPARISON:")
    joint_names = ["hip", "left_knee", "right_knee"]
    
    for i, joint_name in enumerate(joint_names):
        if i < len(fsm_data["qdes"][0]) and i < len(model_data["qdes"][0]):
            fsm_vals = fsm_data["qdes"][:, i]
            model_vals = model_data["qdes"][:, i]
            
            min_len = min(len(fsm_vals), len(model_vals))
            if min_len > 0:
                fsm_vals = fsm_vals[:min_len]
                model_vals = model_vals[:min_len]
                
                diff = np.abs(fsm_vals - model_vals)
                print(f"{joint_name:>12}: FSM={fsm_vals.mean():+.3f}, Model={model_vals.mean():+.3f}, Diff={diff.mean():.3f}")
    
    # Check model raw outputs
    print(f"\n🧠 MODEL RAW OUTPUTS:")
    if len(model_data["raw_outputs"]) > 0:
        raw_outputs = np.array(model_data["raw_outputs"])
        print(f"   Shape: {raw_outputs.shape}")
        print(f"   Range: [{raw_outputs.min():.3f}, {raw_outputs.max():.3f}]")
        print(f"   Mean:  {raw_outputs.mean(axis=0)}")
        print(f"   All zeros? {np.allclose(raw_outputs, 0.0, atol=1e-6)}")
        
        # Show first few outputs
        print(f"   First 5 raw outputs:")
        for i in range(min(5, len(raw_outputs))):
            print(f"     step {i}: {raw_outputs[i]}")
    
    # Performance metrics
    print(f"\n📊 PERFORMANCE METRICS:")
    fsm_total_reward = fsm_data["rewards"].sum()
    model_total_reward = model_data["rewards"].sum() if len(model_data["rewards"]) > 0 else 0
    
    fsm_distance = fsm_data["dx"].sum()
    model_distance = model_data["dx"].sum() if len(model_data["dx"]) > 0 else 0
    
    print(f"Total reward:  FSM={fsm_total_reward:+.1f}, Model={model_total_reward:+.1f}")
    print(f"Distance (m):  FSM={fsm_distance:+.1f}, Model={model_distance:+.1f}")
    
    if len(model_data["pitch"]) > 0:
        print(f"Max pitch:     FSM={fsm_data['pitch'].max():.3f}, Model={model_data['pitch'].max():.3f}")
    
    # Find divergence point
    print(f"\n🔍 DIVERGENCE ANALYSIS:")
    if len(fsm_data["qdes"]) > 0 and len(model_data["qdes"]) > 0:
        min_len = min(len(fsm_data["qdes"]), len(model_data["qdes"]))
        if min_len > 10:  # Need enough data
            fsm_qdes = fsm_data["qdes"][:min_len]
            model_qdes = model_data["qdes"][:min_len]
            
            # Find where joint targets start diverging significantly
            for i, joint_name in enumerate(joint_names):
                if i < fsm_qdes.shape[1] and i < model_qdes.shape[1]:
                    fsm_joint = fsm_qdes[:, i]
                    model_joint = model_qdes[:, i]
                    diff = np.abs(fsm_joint - model_joint)
                    
                    # Find first step where difference > threshold
                    threshold = 0.1  # 10% of joint range
                    divergence_steps = np.where(diff > threshold)[0]
                    
                    if len(divergence_steps) > 0:
                        first_div = divergence_steps[0]
                        print(f"{joint_name:>12}: First divergence at step {first_div} (diff={diff[first_div]:.3f})")
                    else:
                        print(f"{joint_name:>12}: No significant divergence detected")
    
    # Final diagnosis
    print(f"\n💡 DIAGNOSIS:")
    print(f"=" * 30)
    
    if len(model_data["raw_outputs"]) > 0 and np.allclose(model_data["raw_outputs"], 0.0, atol=1e-6):
        print(f"❌ Model learned to output zeros")
        print(f"   → This explains immediate falling")
        print(f"   → Fix: Retrain with --label-type qdes")
    elif model_fell_step and model_fell_step < 100:
        print(f"❌ Model falls very early (step {model_fell_step})")
        print(f"   → Check if model outputs are too small or wrong direction")
        print(f"   → May need proper normalization/denormalization")
    else:
        print(f"✅ Model runs for reasonable duration")
        print(f"   → Check quality of joint targets and reward")


def main():
    parser = argparse.ArgumentParser(description="Diagnose model vs FSM behavior")
    parser.add_argument("--ckpt", required=True, help="Model checkpoint path")
    parser.add_argument("--meta", required=True, help="Model metadata path")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    args = parser.parse_args()
    
    compare_model_vs_fsm(args.ckpt, args.meta, args.duration)


if __name__ == "__main__":
    main()
