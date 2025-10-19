#!/usr/bin/env python3
"""
Analyze what trained BC models actually output during inference.
Diagnoses if models learned zeros vs proper joint positions.
"""
import argparse
import json
import numpy as np
from pathlib import Path

def analyze_model(ckpt_path: str, meta_path: str, data_dir: str = "data/fsm_demos"):
    """
    Analyze model outputs and compare with expected FSM targets.
    
    Args:
        ckpt_path: Path to model checkpoint
        meta_path: Path to model metadata
        data_dir: Path to training data for sample observations
    """
    import torch
    from .models.models_torch import TorchMLP, TorchMLPLarge
    from .utils import load_checkpoint, Normalizer
    from passive_walker.core.controller import PDController
    from .dataset import discover_npzs, load_xy
    
    print(f"🔍 Analyzing model: {ckpt_path}")
    
    # Load model metadata
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    print(f"📋 Model info:")
    print(f"   Section: {meta['section']}")
    print(f"   Input dim: {meta['input_dim']}")
    print(f"   Output dim: {meta['output_dim']}")
    print(f"   Label type: {meta.get('label_type', 'unknown')}")
    
    # Load model
    try:
        model = TorchMLPLarge(in_dim=meta["input_dim"], out_dim=meta["output_dim"], hidden=512, dropout=0.1)
        model = load_checkpoint(ckpt_path, model)
        print(f"✅ Loaded TorchMLPLarge model")
    except Exception:
        model = TorchMLP(in_dim=meta["input_dim"], out_dim=meta["output_dim"])
        model = load_checkpoint(ckpt_path, model)
        print(f"✅ Loaded TorchMLP model")
    
    model.eval()
    
    # Setup normalizer
    normalizer = Normalizer(
        mean=np.array(meta["normalizer_mean"], dtype=np.float32),
        std=np.array(meta["normalizer_std"], dtype=np.float32)
    )
    
    # Load sample observations from training data
    try:
        files = discover_npzs(data_dir)
        train_files = files[:2]  # Use first 2 episodes for testing
        X_sample, _ = load_xy(train_files, meta["section"], meta.get("label_type", "act"), frame_stack=1)
        print(f"📊 Loaded {X_sample.shape[0]} sample observations from training data")
    except Exception as e:
        print(f"❌ Failed to load training data: {e}")
        # Create dummy observations
        X_sample = np.random.randn(100, meta["input_dim"]).astype(np.float32)
        print(f"📊 Using {X_sample.shape[0]} random observations for testing")
    
    # Test model on sample observations
    print(f"\n🧪 TESTING MODEL OUTPUTS:")
    print(f"=" * 40)
    
    model_outputs = []
    
    with torch.no_grad():
        for i in range(min(50, len(X_sample))):  # Test on first 50 samples
            obs = X_sample[i:i+1]  # Keep batch dimension
            obs_norm = normalizer.apply(obs)
            
            # Get model prediction
            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32)
            output = model(obs_tensor)[0].numpy()
            model_outputs.append(output)
    
    model_outputs = np.array(model_outputs)
    
    print(f"📈 Model output statistics:")
    print(f"   Shape: {model_outputs.shape}")
    print(f"   Min:   {model_outputs.min(axis=0)}")
    print(f"   Max:   {model_outputs.max(axis=0)}")
    print(f"   Mean:  {model_outputs.mean(axis=0)}")
    print(f"   Std:   {model_outputs.std(axis=0)}")
    
    # Check if outputs are in expected range [-1, 1] (Tanh activation)
    in_tanh_range = np.all(model_outputs >= -1.0) and np.all(model_outputs <= 1.0)
    print(f"   In Tanh range [-1,1]? {'✅ Yes' if in_tanh_range else '❌ No'}")
    
    # Check if outputs are near zero (learned zeros)
    near_zero = np.allclose(model_outputs, 0.0, atol=1e-6)
    almost_zero = np.all(np.abs(model_outputs) < 0.01)
    print(f"   Near zero? {'✅ Yes' if near_zero else '❌ No'}")
    print(f"   Almost zero (<0.01)? {'⚠️  Yes' if almost_zero else '✅ No'}")
    
    # Show first few outputs
    print(f"\n📋 First 10 model outputs:")
    for i in range(min(10, len(model_outputs))):
        print(f"   obs[{i:2d}]: {model_outputs[i]}")
    
    # Test denormalization
    print(f"\n🔄 TESTING DENORMALIZATION:")
    print(f"=" * 35)
    
    pd_controller = PDController()
    
    # Test denormalization of model outputs to physical joint positions
    if meta["section"] == "hip":
        joint_indices = [0]
        joint_names = ["hip"]
    elif meta["section"] == "knees":
        joint_indices = [1, 2]
        joint_names = ["left_knee", "right_knee"]
    elif meta["section"] == "both":
        joint_indices = [0, 1, 2]
        joint_names = ["hip", "left_knee", "right_knee"]
    
    print(f"   Joint indices: {joint_indices}")
    print(f"   Joint names: {joint_names}")
    
    # Test denormalization for a few outputs
    print(f"\n   Sample denormalizations:")
    for i in range(min(5, len(model_outputs))):
        output = model_outputs[i]
        print(f"   Model output[{i}]: {output}")
        
        denorm_values = []
        for j, joint_idx in enumerate(joint_indices):
            denorm_val = pd_controller.denorm(joint_idx, float(output[j]))
            denorm_values.append(denorm_val)
            print(f"     Joint {joint_names[j]} (idx {joint_idx}): {output[j]:+.3f} → {denorm_val:+.3f}")
    
    # Load actual FSM targets from training data for comparison
    print(f"\n🎯 COMPARING WITH FSM TARGETS:")
    print(f"=" * 40)
    
    try:
        files = discover_npzs(data_dir)
        train_files = files[:1]  # Use first episode
        _, y_fsm_act = load_xy(train_files, meta["section"], "act", frame_stack=1)
        _, y_fsm_qdes = load_xy(train_files, meta["section"], "qdes", frame_stack=1)
        
        print(f"📊 FSM target statistics (from training data):")
        print(f"   'act' labels shape: {y_fsm_act.shape}")
        print(f"   'act' range: [{y_fsm_act.min():.3f}, {y_fsm_act.max():.3f}]")
        print(f"   'act' mean: {y_fsm_act.mean(axis=0)}")
        
        print(f"   'qdes' labels shape: {y_fsm_qdes.shape}")
        print(f"   'qdes' range: [{y_fsm_qdes.min():.3f}, {y_fsm_qdes.max():.3f}]")
        print(f"   'qdes' mean: {y_fsm_qdes.mean(axis=0)}")
        
        # Compare model outputs with training targets
        model_mean = model_outputs.mean(axis=0)
        act_mean = y_fsm_act.mean(axis=0) if len(y_fsm_act) > 0 else np.zeros(model_outputs.shape[1])
        qdes_mean = y_fsm_qdes.mean(axis=0) if len(y_fsm_qdes) > 0 else np.zeros(model_outputs.shape[1])
        
        print(f"\n📈 COMPARISON SUMMARY:")
        print(f"   Model outputs mean: {model_mean}")
        print(f"   Training 'act' mean: {act_mean}")
        print(f"   Training 'qdes' mean: {qdes_mean}")
        
        matches_act = np.allclose(model_mean, act_mean, atol=1e-3)
        matches_qdes = np.allclose(model_mean, qdes_mean, atol=1e-3)
        
        print(f"\n🎯 DIAGNOSIS:")
        if matches_act:
            print(f"   ✅ Model learned to match 'act' targets (but these are zeros!)")
            print(f"   ❌ This explains why model falls - it outputs zeros")
        elif matches_qdes:
            print(f"   ✅ Model learned to match 'qdes' targets (ideal!)")
            print(f"   ✅ This would work with proper denormalization")
        else:
            print(f"   ❓ Model doesn't match either target type")
            print(f"   🔍 Check training process or data loading")
            
    except Exception as e:
        print(f"❌ Failed to load FSM targets for comparison: {e}")
    
    # Final diagnosis
    print(f"\n💡 FINAL DIAGNOSIS:")
    print(f"=" * 30)
    
    if np.allclose(model_outputs, 0.0, atol=1e-6):
        print(f"❌ CRITICAL ISSUE: Model learned to output zeros")
        print(f"   → Training on 'act' field which contains all zeros")
        print(f"   → Model achieved low loss by learning useless constant")
        print(f"   → Fix: Retrain with '--label-type qdes'")
    elif np.all(np.abs(model_outputs) < 0.01):
        print(f"⚠️  Model outputs are very small values")
        print(f"   → Likely learned near-zero outputs")
        print(f"   → Check if training targets were properly normalized")
    else:
        print(f"✅ Model outputs show variation")
        print(f"   → Check denormalization and action assembly")


def main():
    parser = argparse.ArgumentParser(description="Analyze BC model outputs")
    parser.add_argument("--ckpt", required=True, help="Model checkpoint path")
    parser.add_argument("--meta", required=True, help="Model metadata path")
    parser.add_argument("--data", default="data/fsm_demos", help="Training data directory")
    args = parser.parse_args()
    
    analyze_model(args.ckpt, args.meta, args.data)


if __name__ == "__main__":
    main()
