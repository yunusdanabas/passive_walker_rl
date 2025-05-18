# passive_walker/bc/hip_knee_alternatives/train_mse.py
"""
Train with MSE loss.
"""
import argparse, pickle, numpy as np, jax, jax.numpy as jnp, optax, equinox as eqx
from passive_walker.controllers.nn.hip_knee_nn import HipKneeController
from passive_walker.bc.plotters import plot_loss_curve
from . import DATA_DIR, set_device

def train(model, opt, obs, labels, epochs, batch):
    opt_st = opt.init(eqx.filter(model, eqx.is_array))
    hist = []
    
    @jax.jit
    def loss_fn(m,o,y):
        pred = jax.vmap(m)(o)
        return jnp.mean(jnp.abs(pred - y))

    def step(m,st,o,y):
        g= jax.grad(loss_fn)(m,o,y)
        u,st=opt.update(g,st); return eqx.apply_updates(m,u), st
    n=obs.shape[0]
    for e in range(epochs):
        perm=np.random.permutation(n)
        for i in range(0,n,batch):
            idx=perm[i:i+batch]
            model,opt_st=step(model,opt_st,obs[idx],labels[idx])
        l=float(loss_fn(model,obs,labels)); hist.append(l)
        print(f"[MSE] epoch {e+1}/{epochs} loss={l:.4f}")
    return model,hist

def main():
    p=argparse.ArgumentParser(); 
    p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--batch", type=int,default=32)
    p.add_argument("--lr",    type=float,default=1e-4)
    p.add_argument("--gpu",   action="store_true")
    args=p.parse_args()

    set_device(args.gpu)
    import pickle; demos=pickle.load(open(DATA_DIR/"hip_knee_alternatives_demos.pkl","rb"))
    obs,joint_labels = jnp.array(demos["obs"]), jnp.array(demos["labels"])

    model=HipKneeController(input_size=obs.shape[1],hidden_size=128)
    opt=optax.adam(args.lr)
    model,loss_hist = train(model,opt,obs,joint_labels,args.epochs,args.batch)
    plot_loss_curve(loss_hist)
    pickle.dump(model, open(DATA_DIR/"controller_mse.pkl","wb"))

if __name__=="__main__":
    main()
