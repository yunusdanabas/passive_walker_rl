"""
Plot analysis results for behavior cloning models.

This script loads trained behavior cloning models and generates analysis plots
using the plotting utilities from walker_plotter.py.

Usage:
    python -m passive_walker.bc.plot_results [--steps N] [--secs S]
                                           [--out DIR] [--gpu]

The script will automatically process all model types (hip, knee, hip_knee, alternatives)
and save their plots in separate directories.
"""

import argparse
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import pickle

from passive_walker.envs.mujoco_fsm_env import PassiveWalkerEnv
from passive_walker.bc.hip_mse import load_model as load_hip_model
from passive_walker.bc.knee_mse import load_model as load_knee_model
from passive_walker.bc.hip_knee_mse import load_model as load_hip_knee_model
from passive_walker.bc.hip_knee_alternatives import load_model as load_alternatives_model
from passive_walker.utils.walker_plotter import (
    plot_joint_kinematics,
    plot_body_motion,
    plot_energetics,
    plot_com_path,
    plot_feet_contact
)
from passive_walker.constants import XML_PATH, RESULTS

def run_episode(model, model_type, sim_secs=30.0, hz=200):
    """Run episode with trained model and collect data."""
    env = PassiveWalkerEnv(
        xml_path=str(XML_PATH),
        simend=sim_secs,
        use_nn_for_hip=model_type in ["hip", "hip_knee", "alternatives"],
        use_nn_for_knees=model_type in ["knee", "hip_knee", "alternatives"],
        use_gui=False
    )
    
    obs = env.reset()
    log = {
        "t": [],
        "q": [],       # joint positions
        "qd": [],      # joint velocities
        "actions": [], # policy outputs
        "x_torso": [], # COM x
        "z_torso": [], # COM z
        "pitch": [],
        "xdot": [],
        "rewards": [],
        "l_foot_z": [],
        "r_foot_z": []
    }
    
    done = False
    while not done:
        # Get action from model
        obs_j = jnp.array(obs, dtype=jnp.float32)
        action = np.array(model(obs_j))
        
        # Step environment
        obs, reward, done, _ = env.step(action)
        
        # Unpack observation
        (x, z, pitch,
         xdot, _,
         hip_q, lk_q, rk_q,
         hip_qd, lk_qd, rk_qd) = obs
        
        # Get foot heights
        lz = float(env.data.xpos[env.left_foot_body_id, 2])
        rz = float(env.data.xpos[env.right_foot_body_id, 2])
        
        # Log data
        log["t"].append(env.data.time)
        log["q"].append([hip_q, lk_q, rk_q])
        log["qd"].append([hip_qd, lk_qd, rk_qd])
        log["actions"].append(action)
        log["x_torso"].append(x)
        log["z_torso"].append(z)
        log["pitch"].append(pitch)
        log["xdot"].append(xdot)
        log["rewards"].append(reward)
        log["l_foot_z"].append(lz)
        log["r_foot_z"].append(rz)
    
    env.close()
    # Convert lists to numpy arrays
    for k in log:
        log[k] = np.asarray(log[k])
    return log

def get_model_dir(model_type, base_dir):
    """Get the directory for a specific model type's plots."""
    model_dirs = {
        "hip": "hip_mse",
        "knee": "knee_mse",
        "hip_knee": "hip_knee_mse",
        "alternatives": "hip_knee_alternatives"
    }
    return base_dir / model_dirs[model_type]

def process_model(model_type, steps, sim_secs, base_dir):
    """Process a single model type and generate its plots."""
    print(f"\nProcessing {model_type.upper()} model...")
    
    # Create model-specific directory
    model_dir = get_model_dir(model_type, base_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to {model_dir.resolve()}")

    # Load appropriate model
    try:
        if model_type == "hip":
            model = load_hip_model(20_000)
        elif model_type == "knee":
            model = load_knee_model(steps)
        elif model_type == "hip_knee":
            model = load_hip_knee_model(steps)
        else:  # alternatives
            model = load_alternatives_model(steps)

        # Run episode and collect data
        print(f"Running episode with {model_type} model...")
        log = run_episode(model, model_type, sim_secs)

        # Generate plots
        title_prefix = f"{model_type.upper()} BC - "
        plot_joint_kinematics(log, model_dir, title_prefix)
        plot_body_motion(log, model_dir, title_prefix)
        plot_energetics(log, model_dir, title_prefix)
        plot_com_path(log, model_dir, title_prefix)
        plot_feet_contact(log, model_dir, title_prefix)
        
        print(f"Successfully generated plots for {model_type.upper()} model")
        return True
        
    except Exception as e:
        print(f"Error processing {model_type.upper()} model: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Plot BC model analysis results")
    parser.add_argument("--steps", type=int, default=20_000,
                       help="Number of steps used for training")
    parser.add_argument("--secs", type=float, default=30.0,
                       help="Simulation duration in seconds")
    parser.add_argument("--out", type=str, default=str(RESULTS / "figs"),
                       help="Output directory for plots")
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU if available")
    args = parser.parse_args()

    # Create base output directory
    base_dir = Path(args.out)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all model types
    model_types = ["hip", "knee", "hip_knee", "alternatives"]
    success_count = 0
    
    for model_type in model_types:
        if process_model(model_type, args.steps, args.secs, base_dir):
            success_count += 1
    
    print(f"\nProcessing complete. Successfully processed {success_count}/{len(model_types)} models.")

if __name__ == "__main__":
    main() 