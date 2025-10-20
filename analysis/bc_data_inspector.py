#!/usr/bin/env python3
"""
Inspect and analyze training data to understand act vs qdes fields.
Helps diagnose why models learned zeros instead of proper joint targets.
"""
import os
import sys
import argparse
import numpy as np
import json
from pathlib import Path

def inspect_dataset(data_dir: str, sample_size: int = 5):
    """
    Inspect training dataset and analyze the mismatch between 'act' and 'info_qdes'.
    
    Args:
        data_dir: Directory containing episode_*.npz files
        sample_size: Number of episodes to sample for detailed analysis
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
        
    # Find episode files
    episode_files = sorted(data_path.glob("episode_*.npz"))
    if not episode_files:
        print(f"❌ No episode files found in {data_dir}")
        return
        
    print(f"📊 Analyzing dataset: {data_dir}")
    print(f"📁 Found {len(episode_files)} episode files")
    
    # Load sample episodes for detailed analysis
    sample_files = episode_files[:sample_size]
    
    print(f"\n🔍 Detailed analysis of {len(sample_files)} sample episodes:")
    
    all_act_stats = {"min": [], "max": [], "mean": [], "std": []}
    all_qdes_stats = {"min": [], "max": [], "mean": [], "std": []}
    
    for i, ep_file in enumerate(sample_files):
        print(f"\n--- Episode {i+1}: {ep_file.name} ---")
        
        with np.load(ep_file) as data:
            # Check what fields are available
            available_keys = list(data.keys())
            print(f"Available keys: {available_keys}")
            
            if "act" in data:
                act_data = data["act"]
                act_stats = {
                    "shape": act_data.shape,
                    "min": act_data.min(axis=0),
                    "max": act_data.max(axis=0), 
                    "mean": act_data.mean(axis=0),
                    "std": act_data.std(axis=0)
                }
                print(f"📋 'act' field:")
                print(f"    Shape: {act_stats['shape']}")
                print(f"    Min:   {act_stats['min']}")
                print(f"    Max:   {act_stats['max']}")
                print(f"    Mean:  {act_stats['mean']}")
                print(f"    Std:   {act_stats['std']}")
                print(f"    All zeros? {np.allclose(act_data, 0.0)}")
                
                all_act_stats["min"].append(act_stats["min"])
                all_act_stats["max"].append(act_stats["max"])
                all_act_stats["mean"].append(act_stats["mean"])
                all_act_stats["std"].append(act_stats["std"])
            
            if "info_qdes" in data:
                qdes_data = data["info_qdes"]
                qdes_stats = {
                    "shape": qdes_data.shape,
                    "min": qdes_data.min(axis=0),
                    "max": qdes_data.max(axis=0),
                    "mean": qdes_data.mean(axis=0), 
                    "std": qdes_data.std(axis=0)
                }
                print(f"🎯 'info_qdes' field:")
                print(f"    Shape: {qdes_stats['shape']}")
                print(f"    Min:   {qdes_stats['min']}")
                print(f"    Max:   {qdes_stats['max']}")
                print(f"    Mean:  {qdes_stats['mean']}")
                print(f"    Std:   {qdes_stats['std']}")
                print(f"    All zeros? {np.allclose(qdes_data, 0.0)}")
                
                all_qdes_stats["min"].append(qdes_stats["min"])
                all_qdes_stats["max"].append(qdes_stats["max"])
                all_qdes_stats["mean"].append(qdes_stats["mean"])
                all_qdes_stats["std"].append(qdes_stats["std"])
                
                # Show first few timesteps
                print(f"    First 5 timesteps:")
                for t in range(min(5, len(qdes_data))):
                    print(f"      t={t}: qdes={qdes_data[t]}")
    
    # Summary statistics across all episodes
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"=" * 50)
    
    if all_act_stats["min"]:
        act_min_global = np.min(all_act_stats["min"], axis=0)
        act_max_global = np.max(all_act_stats["max"], axis=0)
        act_mean_global = np.mean(all_act_stats["mean"], axis=0)
        print(f"🔹 'act' field (what model trained on):")
        print(f"   Range: [{act_min_global}, {act_max_global}]")
        print(f"   Mean:  {act_mean_global}")
        print(f"   Status: {'❌ ALL ZEROS - MODEL LEARNED NOTHING USEFUL' if np.allclose(act_mean_global, 0.0) else '✅ Has variation'}")
    
    if all_qdes_stats["min"]:
        qdes_min_global = np.min(all_qdes_stats["min"], axis=0)
        qdes_max_global = np.max(all_qdes_stats["max"], axis=0)
        qdes_mean_global = np.mean(all_qdes_stats["mean"], axis=0)
        print(f"🔹 'info_qdes' field (what FSM actually outputs):")
        print(f"   Range: [{qdes_min_global}, {qdes_max_global}]")
        print(f"   Mean:  {qdes_mean_global}")
        print(f"   Status: {'✅ HAS VARIATION - THESE ARE THE REAL TARGETS' if not np.allclose(qdes_mean_global, 0.0) else '❌ Unexpected'}")
    
    # Expected joint ranges (from controller.py)
    JOINT_MIN = np.array([-0.5, -0.5, -0.5])
    JOINT_MAX = np.array([+0.5, +0.5, +0.5])
    print(f"\n🎛️  Expected joint ranges (from controller.py):")
    print(f"   Hip:    [{JOINT_MIN[0]:.1f}, {JOINT_MAX[0]:.1f}] rad")
    print(f"   L-Knee: [{JOINT_MIN[1]:.1f}, {JOINT_MAX[1]:.1f}] m")
    print(f"   R-Knee: [{JOINT_MAX[2]:.1f}, {JOINT_MAX[2]:.1f}] m")
    
    # Diagnosis
    print(f"\n🔬 DIAGNOSIS:")
    print(f"=" * 50)
    
    if all_qdes_stats["min"] and all_act_stats["min"]:
        qdes_in_range = np.all(qdes_min_global >= JOINT_MIN) and np.all(qdes_max_global <= JOINT_MAX)
        
        print(f"✅ 'info_qdes' contains valid FSM joint targets")
        print(f"   → Range verification: {'✅ In expected range' if qdes_in_range else '⚠️  Outside expected range'}")
        print(f"   → These are the actual joint positions FSM computes")
        
        act_all_zeros = np.allclose(act_mean_global, 0.0)
        print(f"❌ 'act' field is {'all zeros' if act_all_zeros else 'mostly constant'}")
        print(f"   → This is what the model was trained to predict")
        print(f"   → Model learned to output zeros regardless of input")
        print(f"   → Explains low training loss but immediate falling")
        
        print(f"\n💡 SOLUTION:")
        print(f"   Retrain with: --label-type qdes")
        print(f"   This will use 'info_qdes' as training targets instead of 'act'")

def compare_sections(data_dir: str):
    """Compare act vs qdes for different control sections."""
    print(f"\n🎯 SECTION COMPARISON")
    print(f"=" * 30)
    
    # Load one episode
    data_path = Path(data_dir)
    episode_files = sorted(data_path.glob("episode_*.npz"))
    if not episode_files:
        return
        
    with np.load(episode_files[0]) as data:
        if "act" in data and "info_qdes" in data:
            act = data["act"]
            qdes = data["info_qdes"] 
            
            sections = {
                "hip": [0],
                "knees": [1, 2], 
                "both": [0, 1, 2]
            }
            
            for section, indices in sections.items():
                print(f"\n{section.upper()} section (indices {indices}):")
                act_section = act[:, indices]
                qdes_section = qdes[:, indices]
                
                print(f"  act range:   [{act_section.min():.3f}, {act_section.max():.3f}]")
                print(f"  qdes range:  [{qdes_section.min():.3f}, {qdes_section.max():.3f}]")
                print(f"  act mean:    {act_section.mean(axis=0)}")
                print(f"  qdes mean:   {qdes_section.mean(axis=0)}")


def main():
    parser = argparse.ArgumentParser(description="Inspect BC training data")
    parser.add_argument("data_dir", help="Directory containing episode_*.npz files")
    parser.add_argument("--samples", type=int, default=5, help="Number of episodes to sample")
    args = parser.parse_args()
    
    inspect_dataset(args.data_dir, args.samples)
    compare_sections(args.data_dir)


if __name__ == "__main__":
    main()
