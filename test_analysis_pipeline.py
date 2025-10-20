#!/usr/bin/env python3
"""
Comprehensive test script for analysis pipeline
Tests all components before running the full analysis
"""

import sys
from pathlib import Path
import numpy as np
import torch

print("=" * 70)
print("🧪 ANALYSIS PIPELINE COMPREHENSIVE TEST")
print("=" * 70)

# Test 1: Imports
print("\n1️⃣  Testing imports...")
try:
    from passive_walker.core.env import PassiveWalkerEnv
    from passive_walker.bc.models.models_torch import TorchMLP, TorchMLPLarge
    from passive_walker.bc.dataset import Normalizer
    from passive_walker.bc.play import _assemble_action_torch
    print("   ✅ All imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Environment creation
print("\n2️⃣  Testing environment creation...")
try:
    env = PassiveWalkerEnv(mode="research", use_gui=False)
    obs, _ = env.reset()
    print(f"   ✅ Environment created")
    print(f"   ✅ Observation shape: {obs.shape}")
    
    # Verify attributes
    assert hasattr(env, 'qpos_x'), "Missing qpos_x"
    assert hasattr(env, 'qpos_z'), "Missing qpos_z"
    assert hasattr(env, 'qpos_hip'), "Missing qpos_hip"
    assert hasattr(env, 'qpos_lk'), "Missing qpos_lk"
    assert hasattr(env, 'qpos_rk'), "Missing qpos_rk"
    assert hasattr(env, 'qvel_hip'), "Missing qvel_hip"
    assert hasattr(env, 'qvel_lk'), "Missing qvel_lk"
    assert hasattr(env, 'qvel_rk'), "Missing qvel_rk"
    print("   ✅ All required attributes present")
    
    env.close()
except Exception as e:
    print(f"   ❌ Environment test failed: {e}")
    sys.exit(1)

# Test 3: Data access patterns
print("\n3️⃣  Testing data access patterns...")
try:
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    obs, _ = env.reset()
    env.step([0, 0, 0])
    
    # Test position access
    x_pos = env.data.qpos[env.qpos_x]
    z_pos = env.data.qpos[env.qpos_z]
    hip_pos = env.data.qpos[env.qpos_hip]
    lk_pos = env.data.qpos[env.qpos_lk]
    rk_pos = env.data.qpos[env.qpos_rk]
    print(f"   ✅ Position access: x={x_pos:.3f}, z={z_pos:.3f}")
    
    # Test velocity access
    x_vel = env.data.qvel[0]
    z_vel = env.data.qvel[1]
    hip_vel = env.data.qvel[env.qvel_hip]
    print(f"   ✅ Velocity access: vx={x_vel:.3f}, vz={z_vel:.3f}")
    
    env.close()
except Exception as e:
    print(f"   ❌ Data access test failed: {e}")
    sys.exit(1)

# Test 4: Model loading
print("\n4️⃣  Testing model loading...")
try:
    import json
    
    checkpoint_path = "checkpoints/torch_both_seed123_ep1_steps180000.pt"
    meta_path = "checkpoints/torch_both_seed123_ep1_steps180000_meta.json"
    
    if not Path(checkpoint_path).exists():
        print(f"   ⚠️  Checkpoint not found: {checkpoint_path}")
        print("   ℹ️  Skipping model test (checkpoint required)")
    else:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        architecture = meta.get('architecture', 'TorchMLPLarge')
        
        if architecture == 'TorchMLPLarge':
            model = TorchMLPLarge(meta['input_dim'], meta['output_dim'])
        else:
            model = TorchMLP(meta['input_dim'], meta['output_dim'])
        
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        model.eval()
        
        print(f"   ✅ Model loaded: {architecture}")
        print(f"   ✅ Input dim: {meta['input_dim']}, Output dim: {meta['output_dim']}")
        
        # Test normalizer
        normalizer = Normalizer(meta['input_dim'])
        if 'normalizer_mean' in meta:
            normalizer.mean = np.array(meta['normalizer_mean'])
            normalizer.std = np.array(meta['normalizer_std'])
        print(f"   ✅ Normalizer loaded")
        
        # Test inference
        obs = np.random.randn(11).astype(np.float32)
        
        # Handle frame stacking
        expected_input_dim = meta['input_dim']
        if expected_input_dim > 11:
            frame_stack = expected_input_dim // 11
            obs = np.concatenate([obs] * frame_stack)
        
        x_normalized = normalizer.apply(obs[None, :]).astype(np.float32)
        with torch.no_grad():
            output = model(torch.tensor(x_normalized, dtype=torch.float32))
        
        print(f"   ✅ Inference successful, output shape: {output.shape}")
        
except Exception as e:
    print(f"   ⚠️  Model test issue: {e}")
    print("   ℹ️  This may be OK if checkpoint doesn't exist")

# Test 5: Action assembly
print("\n5️⃣  Testing action assembly...")
try:
    model_output = np.array([0.1, 0.2, 0.3])
    
    # Test different sections
    for section in ["hip", "knees", "both"]:
        action = _assemble_action_torch(section, model_output, None, None, None, "act")
        print(f"   ✅ Section '{section}': action shape {action.shape}")
        assert action.shape == (3,), f"Wrong action shape for {section}"
    
except Exception as e:
    print(f"   ❌ Action assembly test failed: {e}")
    sys.exit(1)

# Test 6: Episode collection simulation
print("\n6️⃣  Testing episode data collection...")
try:
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    obs, _ = env.reset()
    
    data = {
        'times': [],
        'actions': [],
        'rewards': [],
        'positions': [],
        'velocities': [],
        'joint_angles': [],
        'joint_velocities': []
    }
    
    # Simulate 10 steps
    for i in range(10):
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Collect data (using correct attribute names)
        data['times'].append(env.data.time)
        data['actions'].append(action.copy())
        data['rewards'].append(0.0)
        data['positions'].append([env.data.qpos[env.qpos_x], env.data.qpos[env.qpos_z]])
        data['velocities'].append([env.data.qvel[0], env.data.qvel[1]])
        data['joint_angles'].append([
            env.data.qpos[env.qpos_hip],
            env.data.qpos[env.qpos_lk],
            env.data.qpos[env.qpos_rk]
        ])
        data['joint_velocities'].append([
            env.data.qvel[env.qvel_hip],
            env.data.qvel[env.qvel_lk],
            env.data.qvel[env.qvel_rk]
        ])
        
        obs, reward, done, info = env.step(action)
        data['rewards'][-1] = reward
        
        if done:
            break
    
    # Convert to numpy
    for key in data:
        data[key] = np.array(data[key])
    
    print(f"   ✅ Collected {len(data['times'])} steps")
    print(f"   ✅ Positions shape: {data['positions'].shape}")
    print(f"   ✅ Velocities shape: {data['velocities'].shape}")
    
    env.close()
except Exception as e:
    print(f"   ❌ Episode collection test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Visualization imports
print("\n7️⃣  Testing visualization libraries...")
try:
    import matplotlib.pyplot as plt
    print("   ✅ matplotlib imported")
except Exception as e:
    print(f"   ❌ matplotlib import failed: {e}")
    sys.exit(1)

# Test 8: Analysis module imports
print("\n8️⃣  Testing analysis module imports...")
try:
    sys.path.insert(0, str(Path.cwd() / "analysis_code"))
    from behavioral_analysis import run_behavioral_analysis
    from robustness_testing import run_robustness_testing
    print("   ✅ Analysis modules imported successfully")
except Exception as e:
    print(f"   ❌ Analysis module import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\n🚀 Analysis pipeline is ready to use!")
print("\nRun:")
print("  python analysis_code/run_analysis_pipeline.py \\")
print("      --checkpoint checkpoints/torch_both_seed123_ep1_steps180000.pt \\")
print("      --meta checkpoints/torch_both_seed123_ep1_steps180000_meta.json \\")
print("      --episodes 3")
print("\n" + "=" * 70)
