"""
Train a BC-seeded PPO policy with a learned critic in one script.

This module implements Proximal Policy Optimization (PPO) starting from
a Behavioral Cloning (BC) policy. It includes:
- A critic network for value estimation
- PPO with clipped objective and BC regularization
- Automatic decay of BC coefficient over training

Usage:
    python -m passive_walker.ppo.bc_init.train \
        --bc-model DATA/bc/hip_knee_mse_controller.pkl \
        [--iters I] [--rollout R] [--epochs E] [--batch B] \
        [--gamma G] [--lam L] [--clip C] [--sigma S] \
        [--lr-policy LP] [--lr-critic LC] [--bc-coef BC] [--anneal AN] \
        [--seed SEED] [--gpu]
"""

import argparse, pickle
import numpy as np
import jax, jax.numpy as jnp
import equinox as eqx
import optax
from mujoco.glfw import glfw

from passive_walker.ppo.bc_init.utils import initialize_policy, compute_advantages
from . import set_device, DATA_DIR, XML_PATH

class Critic(eqx.Module):
    """
    Value function critic network.
    
    Implements a 2-layer MLP that estimates state values.
    
    Args:
        obs_dim: Dimension of observation space
        hidden: Number of hidden units per layer
        key: JAX PRNG key
    """
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    l3: eqx.nn.Linear
    
    def __init__(self, obs_dim, hidden=64, key=None):
        if key is None: key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        self.l1 = eqx.nn.Linear(obs_dim, hidden,  key=k1)
        self.l2 = eqx.nn.Linear(hidden,  hidden,  key=k2)
        self.l3 = eqx.nn.Linear(hidden,  1,       key=k3)
    def __call__(self, x):
        x = jax.nn.relu(self.l1(x))
        x = jax.nn.relu(self.l2(x))
        return self.l3(x).squeeze()

def policy_log_prob(policy, obs, acts, sigma):
    """
    Compute log probability of actions under the policy.
    
    Args:
        policy: Policy model
        obs: Observations
        acts: Actions
        sigma: Fixed standard deviation
        
    Returns:
        Log probabilities of actions
    """
    mean = jax.vmap(policy)(obs)
    var  = sigma**2
    log_std = jnp.log(sigma)
    lp = -0.5 * (((acts - mean)**2)/var + 2*log_std + jnp.log(2*jnp.pi))
    return jnp.sum(lp, axis=-1)

def ppo_loss_fn(policy, obs, acts, old_lp, adv, clip_eps, bc_labels, bc_coef, sigma):
    """
    Compute PPO loss with BC regularization.
    
    Args:
        policy: Policy model
        obs: Observations
        acts: Actions 
        old_lp: Old log probabilities
        adv: Advantages
        clip_eps: PPO clipping parameter
        bc_labels: BC target actions
        bc_coef: BC regularization coefficient
        sigma: Policy standard deviation
        
    Returns:
        Combined PPO and BC loss
    """
    new_lp = policy_log_prob(policy, obs, acts, sigma)
    ratio  = jnp.exp(new_lp - old_lp)
    unclipped = ratio * adv
    clipped   = jnp.clip(ratio, 1-clip_eps, 1+clip_eps) * adv
    ppo_obj   = -jnp.mean(jnp.minimum(unclipped, clipped))
    # always include BC term (zero when bc_coef=0)
    im_loss = jnp.mean((jax.vmap(policy)(obs) - bc_labels)**2)
    return ppo_obj + bc_coef * im_loss

@jax.jit
def ppo_step(policy, opt_state, obs, acts, old_lp, adv, clip_eps, bc_labels, bc_coef, sigma):
    """
    Perform one PPO update step.
    
    Args:
        policy: Policy model
        opt_state: Optimizer state
        obs: Observations
        acts: Actions
        old_lp: Old log probabilities
        adv: Advantages
        clip_eps: PPO clipping parameter
        bc_labels: BC target actions
        bc_coef: BC regularization coefficient
        sigma: Policy standard deviation
        
    Returns:
        Updated policy and optimizer state
    """
    grads = jax.grad(ppo_loss_fn)(
        policy, obs, acts, old_lp, adv, clip_eps, bc_labels, bc_coef, sigma
    )
    updates, opt_state = policy_optimizer.update(grads, opt_state)
    return eqx.apply_updates(policy, updates), opt_state

@jax.jit
def critic_step(critic, opt_state, obs, returns):
    """
    Perform one critic update step.
    
    Args:
        critic: Critic model
        opt_state: Optimizer state
        obs: Observations
        returns: Returns
        
    Returns:
        Updated critic and optimizer state
    """
    def vf_loss(c, o, r): return jnp.mean((jax.vmap(c)(o) - r)**2)
    grads = jax.grad(vf_loss)(critic, obs, returns)
    updates, opt_state = critic_optimizer.update(grads, opt_state)
    return eqx.apply_updates(critic, updates), opt_state

def main():
    # Parse command line arguments
    p = argparse.ArgumentParser()
    p.add_argument("--bc-model", type=str, required=True)
    p.add_argument("--iters",    type=int,   default=200)
    p.add_argument("--rollout",  type=int,   default=2048)
    p.add_argument("--epochs",   type=int,   default=20)
    p.add_argument("--batch",    type=int,   default=256)
    p.add_argument("--gamma",    type=float, default=0.99)
    p.add_argument("--lam",      type=float, default=0.95)
    p.add_argument("--clip",     type=float, default=0.2)
    p.add_argument("--sigma",    type=float, default=0.1)
    p.add_argument("--lr-policy",type=float, default=3e-4)
    p.add_argument("--lr-critic",type=float, default=1e-4)
    p.add_argument("--bc-coef",  type=float, default=1.0)
    p.add_argument("--anneal",   type=int,   default=200_000)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--gpu",      action="store_true")
    args = p.parse_args()

    # Configure GPU/CPU
    set_device(args.gpu)
    
    # Define global optimizers
    global policy_optimizer, critic_optimizer
    
    # Initialize environment and BC policy
    env, get_scaled, get_env_act, policy = initialize_policy(
        model_path=args.bc_model,
        xml_path=str(XML_PATH),
        simend=args.rollout/60.0,
        sigma=args.sigma,
        use_gui=False
    )

    # Build critic network and optimizers
    obs_dim = env.observation_space.shape[0]
    critic = Critic(obs_dim, key=jax.random.PRNGKey(args.seed))
    policy_optimizer = optax.adam(args.lr_policy)
    critic_optimizer = optax.adam(args.lr_critic)
    pol_state   = policy_optimizer.init(eqx.filter(policy,   eqx.is_array))
    crt_state   = critic_optimizer.init(eqx.filter(critic,   eqx.is_array))

    # Initialize rewards log
    rewards_log = []

    # Main training loop
    total_steps = 0
    for it in range(1, args.iters+1):
        # Collect rollout data
        buf = {'obs':[], 'scaled':[], 'acts':[],
               'rews':[], 'dones':[]}
        o = env.reset()
        for _ in range(args.rollout):
            oj = jnp.array(o, dtype=jnp.float32)
            scaled = np.array(get_scaled(oj), dtype=np.float32)
            act    = get_env_act(oj)
            no, r, done, _ = env.step(act)
            buf['obs'].append(o)
            buf['scaled'].append(scaled)
            buf['acts'].append(act)
            buf['rews'].append(r)
            buf['dones'].append(done)
            o = no if not done else env.reset()

        # Convert to arrays
        obs   = jnp.array(buf['obs'],   dtype=jnp.float32)
        acts  = jnp.array(buf['scaled'],dtype=jnp.float32)
        rews  = np.array(buf['rews'])
        dones = np.array(buf['dones'],  dtype=np.float32)

        # BC labels = scaled actions
        bc_labels = acts

        # Compute advantages using Generalized Advantage Estimation (GAE)
        vals = np.array(jax.vmap(critic)(obs))
        adv, ret = compute_advantages(rews, dones, vals,
                                      gamma=args.gamma, lam=args.lam)
        adv_j = jnp.array(adv, dtype=jnp.float32)
        ret_j = jnp.array(ret, dtype=jnp.float32)

        # Compute old log-probabilities for PPO ratio
        old_lp = policy_log_prob(policy, obs, acts, args.sigma)

        # Anneal BC coefficient
        bc_coef = args.bc_coef * max(0.0, 1 - total_steps/args.anneal)

        # Perform PPO and critic updates
        idxs = np.arange(obs.shape[0])
        for _ in range(args.epochs):
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), args.batch):
                b = idxs[start:start+args.batch]
                policy, pol_state = ppo_step(
                    policy, pol_state,
                    obs[b], acts[b], old_lp[b], adv_j[b],
                    args.clip, bc_labels[b], bc_coef,
                    args.sigma
                )
                critic, crt_state = critic_step(
                    critic, crt_state,
                    obs[b], ret_j[b]
                )

        # Log average reward for this iteration
        avg_reward = rews.mean()
        rewards_log.append(avg_reward)

        total_steps += args.rollout
        print(f"[PPO] iter {it}/{args.iters}  avg_rew={avg_reward:.2f}  bc_coef={bc_coef:.3f}")

    env.close()
    
    # Save trained policy and critic
    out_path = DATA_DIR / "trained_policy_with_critic.pkl"
    with open(out_path, "wb") as f:
        pickle.dump((policy, critic), f)
    print(f"Saved policy+critic → {out_path}")

    # Save training rewards log
    from passive_walker.ppo.bc_init.utils import save_pickle
    log_path = DATA_DIR / "ppo_training_log.pkl"
    save_pickle({"rewards": rewards_log}, log_path)
    print(f"Saved reward log → {log_path}")

if __name__ == "__main__":
    main()
