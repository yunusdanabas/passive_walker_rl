"""
walker_plotter.py
Generate analysis plots for Passive Walker simulations and PPO training results.

Usage
-----
# For FSM analysis:
python walker_plotter.py               # 30 s demo (FSM) → PNGs in ./figs
python walker_plotter.py --secs 15     # shorter rollout
python walker_plotter.py --out results # custom output directory

# For PPO analysis:
python walker_plotter.py --ppo --policy path/to/policy.eqx  # analyze PPO policy
python walker_plotter.py --ppo --log path/to/training_log.pkl  # plot training curves
"""
import argparse, os, math, pathlib
import numpy as np
import matplotlib.pyplot as plt
import pickle
import jax.numpy as jnp
from pathlib import Path

from passive_walker.envs.mujoco_env import PassiveWalkerEnv
from passive_walker.envs.mujoco_fsm_env import PassiveWalkerEnv as FSMEnv
from passive_walker.constants import XML_PATH, RESULTS
from passive_walker.ppo.scratch import load_model

# ––– Defaults ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
SAVE_DIR = RESULTS / "figs"   # override with --out
SIM_SECS = 30.0                   # override with --secs
CTRL_HZ  = 1000                   # PPO default
DT       = 1.0 / CTRL_HZ
FOOT_H_THRESH = 0.05              # contact if z < thresh

# ––– Data collection ––––––––––––––––––––––––––––––––––––––––––––––––––
def run_fsm_episode(sim_secs=SIM_SECS):
    """Run episode with FSM controller."""
    env = FSMEnv(XML_PATH,
                 simend=sim_secs,
                 use_nn_for_hip=False,
                 use_nn_for_knees=False,
                 use_gui=False)
    obs = env.reset()
    log = {
        "t": [],
        "q": [],       # joint positions hip, lk, rk
        "qd": [],      # joint velocities
        "ctrl": [],    # actuator commands
        "x_torso": [], # COM x
        "z_torso": [], # COM z
        "pitch": [],
        "xdot": [],
        "energy": [],
        "l_foot_z": [],
        "r_foot_z": [],
        "fsm_hip": [],
        "fsm_k1": [],
        "fsm_k2": []
    }
    cum_energy = 0.0
    done = False
    while not done:
        # zero external action – FSM controls everything
        obs, _, done, _ = env.step(np.zeros(3, dtype=np.float32))

        # unpack obs array (see env._get_obs)
        (x, z, pitch,
         xdot, _,
         hip_q, lk_q, rk_q,
         hip_qd, lk_qd, rk_qd) = obs

        # actuator commands currently set in env.data.ctrl
        ctrl_vec = env.data.ctrl[[env.hip_pos_actuator_id,
                                  env.left_knee_pos_actuator_id,
                                  env.right_knee_pos_actuator_id]]
        # power = τ·ω ; here control signal acts like desired pos,
        # so we treat |ctrl * qd| as a proxy
        inst_e = np.sum(np.abs(ctrl_vec * np.array([hip_qd, lk_qd, rk_qd])))
        cum_energy += inst_e * DT

        # foot heights (z of geom's CoM)
        lz = float(env.data.xpos[env.left_foot_body_id, 2])
        rz = float(env.data.xpos[env.right_foot_body_id, 2])

        # log everything
        log["t"].append(env.data.time)
        log["q"].append([hip_q, lk_q, rk_q])
        log["qd"].append([hip_qd, lk_qd, rk_qd])
        log["ctrl"].append(ctrl_vec.copy())
        log["x_torso"].append(x)
        log["z_torso"].append(z)
        log["pitch"].append(pitch)
        log["xdot"].append(xdot)
        log["energy"].append(cum_energy)
        log["l_foot_z"].append(lz)
        log["r_foot_z"].append(rz)
        log["fsm_hip"].append(env.fsm_hip)
        log["fsm_k1"].append(env.fsm_knee1)
        log["fsm_k2"].append(env.fsm_knee2)

    env.close()
    # convert lists→numpy for convenience
    for k in log:
        log[k] = np.asarray(log[k])
    return log

def run_ppo_episode(policy_path, sim_secs=SIM_SECS, hz=CTRL_HZ):
    """Run episode with PPO policy."""
    env = PassiveWalkerEnv(
        xml_path=str(XML_PATH),
        simend=sim_secs,
        use_nn_for_hip=True,
        use_nn_for_knees=True,
        use_gui=False
    )
    
    # Load policy
    policy, _ = load_model(policy_path)
    
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
        # Get action from policy
        obs_j = jnp.array(obs, dtype=jnp.float32)
        action = np.array(policy(obs_j))
        
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

# ––– Plot helpers ––––––––––––––––––––––––––––––––––––––––––––––––––––––
def plot_joint_kinematics(log, outdir, title_prefix=""):
    t = log["t"]
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    ax[0].plot(t, log["q"])
    ax[0].set_ylabel("angle [rad]")
    ax[0].legend(["hip", "left knee", "right knee"])
    ax[0].set_title(f"{title_prefix}Joint Positions")
    ax[0].grid(True)
    
    ax[1].plot(t, log["qd"])
    ax[1].set_ylabel("velocity [rad/s]")
    ax[1].set_xlabel("time [s]")
    ax[1].set_title(f"{title_prefix}Joint Velocities")
    ax[1].grid(True)
    
    fig.suptitle(f"{title_prefix}Joint Kinematics Over Time", y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "joint_kinematics.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_body_motion(log, outdir, title_prefix=""):
    t = log["t"]
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    ax[0].plot(t, log["pitch"])
    ax[0].set_ylabel("pitch [rad]")
    ax[0].set_title(f"{title_prefix}Body Pitch Angle")
    ax[0].grid(True)
    
    ax[1].plot(t, log["xdot"])
    ax[1].set_ylabel("forward speed [m/s]")
    ax[1].set_xlabel("time [s]")
    ax[1].set_title(f"{title_prefix}Forward Velocity")
    ax[1].grid(True)
    
    fig.suptitle(f"{title_prefix}Body Motion Characteristics", y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "body_motion.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_energetics(log, outdir, title_prefix=""):
    t = log["t"]
    fig, ax = plt.subplots(figsize=(10, 4))
    if "energy" in log:
        ax.plot(t, log["energy"])
        ax.set_ylabel("cumulative |τ·ω|")
    else:
        ax.plot(t, np.cumsum(log["rewards"]))
        ax.set_ylabel("cumulative reward")
    ax.set_xlabel("time [s]")
    ax.set_title(f"{title_prefix}Cumulative Energy/Reward")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(outdir / "energy_proxy.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_com_path(log, outdir, title_prefix=""):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(log["x_torso"], log["z_torso"])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"{title_prefix}Center of Mass Trajectory")
    # Set y-axis limits to focus on the relevant range
    y_min = np.min(log["z_torso"])
    y_max = np.max(log["z_torso"])
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(outdir / "com_path.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_feet_contact(log, outdir, title_prefix=""):
    t = log["t"]
    lz = log["l_foot_z"]
    rz = log["r_foot_z"]
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    ax[0].plot(t, lz, t, np.full_like(t, FOOT_H_THRESH, dtype=float), linestyle="--")
    ax[0].set_ylabel("L-foot z")
    ax[0].set_title(f"{title_prefix}Left Foot Height")
    ax[0].grid(True)
    
    ax[1].plot(t, rz, t, np.full_like(t, FOOT_H_THRESH, dtype=float), linestyle="--")
    ax[1].set_ylabel("R-foot z")
    ax[1].set_xlabel("time [s]")
    ax[1].set_title(f"{title_prefix}Right Foot Height")
    ax[1].grid(True)
    
    fig.suptitle(f"{title_prefix}Foot Contact Analysis", y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "foot_height.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_ppo_actions(log, outdir, title_prefix=""):
    """Plot PPO policy actions over time."""
    t = log["t"]
    actions = log["actions"]
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    
    for i, joint in enumerate(["Hip", "Left Knee", "Right Knee"]):
        ax[i].plot(t, actions[:, i])
        ax[i].set_ylabel(f"{joint} Action")
        ax[i].set_title(f"{title_prefix}{joint} Control Signal")
        ax[i].grid(True)
    
    ax[-1].set_xlabel("time [s]")
    fig.suptitle(f"{title_prefix}Policy Actions Over Time", y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "policy_actions.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_training_curves(log_path, outdir):
    """Plot PPO training curves."""
    with open(log_path, "rb") as f:
        log = pickle.load(f)
    
    rewards = log["rewards"]
    iterations = range(1, len(rewards) + 1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(iterations, rewards, 'b-', label='Average Reward')
    
    # Add moving average
    window = min(20, len(rewards) // 10)
    if window > 1:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(rewards) + 1), moving_avg, 'r--', 
                label=f'{window}-episode Moving Average')
    
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Average Reward')
    ax.set_title('PPO Training Progress')
    ax.grid(True)
    ax.legend()
    
    # Add statistics
    stats_text = (
        f"Final Reward: {rewards[-1]:.2f}\n"
        f"Best Reward: {np.max(rewards):.2f}\n"
        f"Mean Reward: {np.mean(rewards):.2f}\n"
        f"Total Iterations: {len(rewards)}"
    )
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.tight_layout()
    fig.savefig(outdir / "training_curves.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# ––– Main ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--secs", type=float, default=SIM_SECS,
                       help="simulation length [s]")
    parser.add_argument("--out", type=str, default=str(SAVE_DIR),
                       help="output folder")
    parser.add_argument("--ppo", action="store_true",
                       help="analyze PPO policy instead of FSM")
    parser.add_argument("--policy", type=str,
                       help="path to PPO policy file")
    parser.add_argument("--log", type=str,
                       help="path to PPO training log")
    parser.add_argument("--hz", type=int, default=CTRL_HZ,
                       help="control frequency [Hz]")
    args = parser.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to {outdir.resolve()}")

    if args.ppo:
        if args.policy:
            # Analyze PPO policy
            log = run_ppo_episode(args.policy, args.secs, args.hz)
            title_prefix = "PPO Policy - "
        elif args.log:
            # Plot training curves
            plot_training_curves(args.log, outdir)
            print("Done.")
            return
        else:
            raise ValueError("Must provide either --policy or --log for PPO analysis")
    else:
        # Analyze FSM
        log = run_fsm_episode(sim_secs=args.secs)
        title_prefix = "FSM - "

    # Generate plots
    plot_joint_kinematics(log, outdir, title_prefix)
    plot_body_motion(log, outdir, title_prefix)
    plot_energetics(log, outdir, title_prefix)
    plot_com_path(log, outdir, title_prefix)
    plot_feet_contact(log, outdir, title_prefix)
    
    if args.ppo:
        plot_ppo_actions(log, outdir, title_prefix)
    
    print("Done.")

if __name__ == "__main__":
    main()
