"""
One-shot pipeline: collect demos → train MLP → GUI rollout.

Usage:
    python -m passive_walker.bc.hip_knee_mse.run_pipeline [--steps N] [--epochs E]
                                                    [--batch B] [--hidden-size H]
                                                    [--lr LR] [--sim-duration S]
                                                    [--seed SEED] [--gpu] [--plot]
"""

import argparse, pickle, numpy as np, jax, jax.numpy as jnp, optax
from mujoco.glfw import glfw

from passive_walker.envs.mujoco_fsm_env import PassiveWalkerEnv
from passive_walker.controllers.nn.hip_knee_nn import HipKneeController
from passive_walker.bc.hip_knee_mse.collect import collect_demo_data
from passive_walker.bc.hip_knee_mse.train   import train_nn_controller
from . import DATA_DIR, XML_PATH, set_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps",        type=int,   default=20_000)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch",        type=int,   default=32)
    p.add_argument("--hidden-size",  type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--sim-duration", type=float, default=30.0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--gpu",          action="store_true")
    p.add_argument("--plot",         action="store_true")
    args = p.parse_args()

    set_device(args.gpu)

    # 1) collect demos
    env_demo = PassiveWalkerEnv(
        xml_path=str(XML_PATH),
        simend=args.steps / 60.0,
        use_nn_for_hip=False,
        use_nn_for_knees=False,
        use_gui=False
    )
    obs, labels = collect_demo_data(env_demo, args.steps)
    env_demo.close()
    print(f"[collect] obs={obs.shape}, labels={labels.shape}")

    # 2) train
    model     = HipKneeController(input_size=obs.shape[1], hidden_size=args.hidden_size,
                                   key=jax.random.PRNGKey(args.seed))
    optimizer = optax.adam(args.lr)
    model, _  = train_nn_controller(model, optimizer, obs, labels,
                                     epochs=args.epochs, batch=args.batch,
                                     plot_loss=args.plot)

    out = DATA_DIR / "hip_knee_mse_controller.pkl"
    pickle.dump(model, open(out, "wb"))
    print(f"[train] saved → {out}")

    # 3) GUI rollout
    env_test = PassiveWalkerEnv(
        xml_path=str(XML_PATH),
        simend=args.sim_duration,
        use_nn_for_hip=True,
        use_nn_for_knees=True,
        use_gui=True
    )
    obs, done, total = env_test.reset(), False, 0.0
    while not done and not glfw.window_should_close(env_test.window):
        act = np.array(model(jnp.array(obs)))  # (3,)
        obs, r, done, _ = env_test.step(act)
        total += r
        env_test.render()
    env_test.close()
    print(f"[rollout] total reward = {total:.3f}")


if __name__ == "__main__":
    main()
