"""
walker_plotter.py
Generate analysis plots for Passive Walker simulations.

Usage
-----
python walker_plotter.py               # 30 s demo (FSM) → PNGs in ./figs
python walker_plotter.py --secs 15     # shorter rollout
python walker_plotter.py --out results # custom output directory
"""
import argparse, os, math, pathlib
import numpy as np
import matplotlib.pyplot as plt

from passive_walker.envs.mujoco_fsm_env import PassiveWalkerEnv
from passive_walker.constants import XML_PATH, DATA_DIR

# ––– Defaults ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
SAVE_DIR = DATA_DIR / "walker_figs"   # override with --out
SIM_SECS = 30.0                   # override with --secs
CTRL_HZ  = 60                     # env default
DT       = 1.0 / CTRL_HZ
FOOT_H_THRESH = 0.05              # contact if z < thresh

# ––– Data collection ––––––––––––––––––––––––––––––––––––––––––––––––––
def run_episode(sim_secs=SIM_SECS):
    env = PassiveWalkerEnv(XML_PATH,
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

# ––– Plot helpers ––––––––––––––––––––––––––––––––––––––––––––––––––––––
def plot_joint_kinematics(log, outdir):
    t = log["t"]
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(t, log["q"])
    ax[0].set_ylabel("angle [rad]")
    ax[0].legend(["hip", "left knee", "right knee"])
    ax[0].set_title("Joint Positions")
    ax[1].plot(t, log["qd"])
    ax[1].set_ylabel("velocity [rad/s]")
    ax[1].set_xlabel("time [s]")
    ax[1].set_title("Joint Velocities")
    fig.suptitle("Joint Kinematics Over Time", y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "joint_kinematics.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_body_motion(log, outdir):
    t = log["t"]
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(t, log["pitch"])
    ax[0].set_ylabel("pitch [rad]")
    ax[0].set_title("Body Pitch Angle")
    ax[1].plot(t, log["xdot"])
    ax[1].set_ylabel("forward speed [m/s]")
    ax[1].set_xlabel("time [s]")
    ax[1].set_title("Forward Velocity")
    fig.suptitle("Body Motion Characteristics", y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "body_motion.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_energetics(log, outdir):
    t = log["t"]
    fig, ax = plt.subplots()
    ax.plot(t, log["energy"])
    ax.set_ylabel("cumulative |τ·ω|")
    ax.set_xlabel("time [s]")
    ax.set_title("Cumulative Energy Consumption")
    fig.tight_layout()
    fig.savefig(outdir / "energy_proxy.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_com_path(log, outdir):
    fig, ax = plt.subplots()
    ax.plot(log["x_torso"], log["z_torso"])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Center of Mass Trajectory")
    # Set y-axis limits to focus on the relevant range
    y_min = np.min(log["z_torso"])
    y_max = np.max(log["z_torso"])
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    fig.tight_layout()
    fig.savefig(outdir / "com_path.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_feet_contact(log, outdir):
    t = log["t"]
    lz = log["l_foot_z"]
    rz = log["r_foot_z"]
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(t, lz, t, np.full_like(t, FOOT_H_THRESH, dtype=float), linestyle="--")
    ax[0].set_ylabel("L-foot z")
    ax[0].set_title("Left Foot Height")
    ax[1].plot(t, rz, t, np.full_like(t, FOOT_H_THRESH, dtype=float), linestyle="--")
    ax[1].set_ylabel("R-foot z")
    ax[1].set_xlabel("time [s]")
    ax[1].set_title("Right Foot Height")
    fig.suptitle("Foot Contact Analysis", y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "foot_height.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_fsm_timeline(log, outdir):
    t = log["t"]
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6,4))
    ax[0].step(t, log["fsm_hip"], where="post")
    ax[0].set_ylabel("hip FSM")
    ax[0].set_title("Hip State Machine")
    ax[1].step(t, log["fsm_k1"], where="post")
    ax[1].set_ylabel("left knee")
    ax[1].set_title("Left Knee State Machine")
    ax[2].step(t, log["fsm_k2"], where="post")
    ax[2].set_ylabel("right knee")
    ax[2].set_xlabel("time [s]")
    ax[2].set_title("Right Knee State Machine")
    fig.suptitle("Finite State Machine Timeline", y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "fsm_timeline.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# ––– Main ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--secs", type=float, default=SIM_SECS, help="simulation length [s]")
    parser.add_argument("--out",  type=str,   default=str(SAVE_DIR), help="output folder")
    args = parser.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to {outdir.resolve()}")

    log = run_episode(sim_secs=args.secs)

    plot_joint_kinematics(log, outdir)
    plot_body_motion(log, outdir)
    plot_energetics(log, outdir)
    plot_com_path(log, outdir)
    plot_feet_contact(log, outdir)
    plot_fsm_timeline(log, outdir)
    print("Done.")

if __name__ == "__main__":
    main()
