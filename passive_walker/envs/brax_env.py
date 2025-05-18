import jax
import jax.numpy as jnp
import brax
from brax.envs.base    import PipelineEnv, State


class BraxPassiveWalker(PipelineEnv):
    """
    Simple PD‑controlled biped walker (hip + two knees) converted from MuJoCo XML.
    * Action ∈ [‑1,1]^3, scaled to physical ranges inside the class.
    * Reward = forward progress (Δx).  Episode ends if torso falls below 0.5 m
      or pitch exceeds ±0.8 rad.
    """
    def __init__(self,
                 sys: brax.System,
                 kp: jnp.ndarray = jnp.array([  5.0, 1000.0, 1000.0]),
                 kd: jnp.ndarray = jnp.array([  0.5,   50.0,   50.0]),
                 reset_noise: float = 0.1):
        super().__init__(sys=sys, backend='positional')

        self.act_idx      = sys.actuator.q_id            # [hip, knees]
        self.kp, self.kd  = kp, kd
        self.action_scale = jnp.array([0.5, 0.3, 0.3])   # rad, m, m
        self.reset_noise  = reset_noise

        self.obs_size     = 2 + 1 + 2 + 3 + 3   # (see _get_obs)
        self.act_size     = 3

    # ---------- helpers ---------------------------------------------------
    def _get_obs(self, ps: brax.base.State) -> jnp.ndarray:
        torso_euler = brax.math.quat_to_euler(ps.x.rot[0])  # xyz order
        return jnp.concatenate([
            ps.x.pos[0, (0, 2)],          # x, z
            torso_euler[1:2],             # pitch
            ps.xd.vel[0, (0, 2)],         # ẋ, ż
            ps.q[self.act_idx],           # joint pos
            ps.qd[self.act_idx],          # joint vel
        ])

    # ---------- API -------------------------------------------------------
    def reset(self, rng: jax.Array):
        qd_dim = int(self.sys.qd_size())
        q  = self.sys.init_q.copy()
        qd = jnp.zeros(qd_dim)

        # small uniform noise on the three actuated joints
        rng, sub = jax.random.split(rng)
        q_noise  = self.reset_noise * (2.*jax.random.uniform(sub, (3,)) - 1.)
        q = q.at[self.act_idx].set(q_noise * self.action_scale)

        ps = self.pipeline_init(q, qd)          # now uses generalized pipeline
        return brax.envs.base.State(            # same namedtuple as before
            pipeline_state = ps,
            obs            = self._get_obs(ps),
            reward         = jnp.array(0., jnp.float32),
            done           = jnp.array(0., jnp.float32),
            metrics        = {
                'prev_x': ps.x.pos[0, 0],
                'reward': jnp.array(0., jnp.float32),
            },
        )        

    def step(self, state, action):
        action   = jnp.clip(action, -1., 1.)
        targets  = action * self.action_scale

        q, qd    = state.pipeline_state.q[self.act_idx], state.pipeline_state.qd[self.act_idx]
        τ        = self.kp * (targets - q) - self.kd * qd  # PD torque

        ps_next  = self.pipeline_step(state.pipeline_state, τ)
        reward   = ps_next.x.pos[0, 0] - state.metrics['prev_x']

        height_ok = ps_next.x.pos[0, 2] > 0.5
        pitch_ok  = jnp.abs(brax.math.quat_to_euler(ps_next.x.rot[0])[1]) < 0.8
        done      = jnp.logical_not(jnp.logical_and(height_ok, pitch_ok))

        return state.replace(
            pipeline_state = ps_next,
            obs            = self._get_obs(ps_next),
            reward         = reward,
            done           = done.astype(jnp.float32),
            metrics        = {
                'prev_x': ps_next.x.pos[0, 0],
                'reward': reward,
            },
        )