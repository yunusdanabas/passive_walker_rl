# passive_walker/bc/hip_knee_alternatives/run_pipeline.py
import argparse, pickle
from passive_walker.envs.mujoco_fsm_env import PassiveWalkerEnv
from . import DATA_DIR, XML_PATH, set_device
from .collect import collect_demo_data
import jax.numpy as jnp

def main():
    import optax, jax
    from passive_walker.controllers.nn.hip_knee_nn import HipKneeController

    p=argparse.ArgumentParser()
    p.add_argument("--steps",type=int,default=20000)
    p.add_argument("--gpu",  action="store_true")
    args=p.parse_args()
    set_device(args.gpu)

    # 1) collect
    env=PassiveWalkerEnv(XML_PATH,simend=args.steps/60, use_nn_for_hip=False,use_nn_for_knees=False,use_gui=False)
    obs,labels=collect_demo_data(env,args.steps); env.close()
    pickle.dump({"obs":obs,"labels":labels}, open(DATA_DIR/"hip_knee_alternatives_demos.pkl","wb"))

    # 2) for each loss
    for variant in ["mse","huber","l1"]:
        script = f"passive_walker.bc.hip_knee_alternatives.train_{variant}"
        print("→ training",variant)
        __import__(script, fromlist=["main"]).main_args = ["--gpu"] if args.gpu else []
        __import__(script, fromlist=["main"]).main()

if __name__=="__main__":
    main()
