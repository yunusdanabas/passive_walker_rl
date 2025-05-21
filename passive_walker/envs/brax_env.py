# NEW / CHANGED – final tidy-up: we now lazy-load the cached System once,
# expose .observation_size / .action_size and remove v1-only imports.

from __future__ import annotations
import jax, jax.numpy as jnp
from brax.envs.base import PipelineEnv, State
from brax.math      import quat_to_euler
import brax


class BraxPassiveWalker(PipelineEnv):
    """
    PD-controlled biped walker (hip + two knees) converted from MuJoCo.

    * **Action** ∈ [-1, 1]^3 – scaled to physical ranges internally  
    * **Reward** = forward progress Δx (per step)  
    * **Episode terminates** if torso height < 0.5 m **or** pitch |θ| > 0.8 rad
    """

    def __init__(
        self,
        sys: brax.System,               # override for unit-tests if needed
        kp: jnp.ndarray  = jnp.array([ 5.0, 1000.0, 1000.0]),
        kd: jnp.ndarray  = jnp.array([ 0.5,   50.0,   50.0]),
        reset_noise: float = 0.10,
    ):
        super().__init__(sys=sys, backend="positional")

        self.act_idx       = sys.actuator.q_id        # hip, kneeL, kneeR
        self.kp, self.kd   = kp, kd
        self.action_scale  = jnp.array([0.5, 0.3, 0.3])   # rad, m, m
        self.reset_noise   = reset_noise

        # required by brax.training
        self._observation_size = 2 + 1 + 2 + 3 + 3    # keep in sync with _get_obs()
        self._action_size      = 3

    # ------------------------------------------------------------------ utils
    def _get_obs(self, ps: brax.base.State) -> jnp.ndarray:
        torso_euler = quat_to_euler(ps.x.rot[0])       # xyz
        return jnp.concatenate(
            [
                ps.x.pos[0, (0, 2)],                  # x, z
                torso_euler[1:2],                     # pitch
                ps.xd.vel[0, (0, 2)],                 # ẋ, ż
                ps.q[self.act_idx],                   # joint position
                ps.qd[self.act_idx],                  # joint velocity
            ]
        )

    # ------------------------------------------------------------------ API
    @property
    def observation_size(self) -> int:   # <- brax.training uses these
        return self._observation_size

    @property
    def action_size(self) -> int:
        return self._action_size

    # -------------- reset --------------------------------------------------
    def reset(self, rng: jax.Array):
        q      = self.sys.init_q.copy()
        qd_dim = int(self.sys.qd_size())
        qd     = jnp.zeros(qd_dim)

        # randomise initial joint positions a bit
        rng, sub = jax.random.split(rng)
        noise    = self.reset_noise * (2.0 * jax.random.uniform(sub, (3,)) - 1.0)
        q        = q.at[self.act_idx].set(noise * self.action_scale)

        ps = self.pipeline_init(q, qd)
        return State(
            pipeline_state=ps,
            obs=self._get_obs(ps),
            reward=jnp.array(0.0, jnp.float32),
            done=jnp.array(0.0, jnp.float32),
            metrics=dict(prev_x=ps.x.pos[0, 0], 
                         reward=jnp.array(0.0, jnp.float32)),
        )

    # -------------- step ---------------------------------------------------
    def step(self, state: State, action: jnp.ndarray):
        action  = jnp.clip(action, -1.0, 1.0)
        targets = action * self.action_scale

        q  = state.pipeline_state.q[self.act_idx]
        qd = state.pipeline_state.qd[self.act_idx]
        tau = self.kp * (targets - q) - self.kd * qd     # PD torque

        ps_next = self.pipeline_step(state.pipeline_state, tau)
        reward  = ps_next.x.pos[0, 0] - state.metrics["prev_x"]

        height_ok = ps_next.x.pos[0, 2] > 0.5
        pitch_ok  = jnp.abs(quat_to_euler(ps_next.x.rot[0])[1]) < 0.8
        done      = ~(height_ok & pitch_ok)

        return state.replace(
            pipeline_state=ps_next,
            obs=self._get_obs(ps_next),
            reward=reward,
            done=done.astype(jnp.float32),
            metrics=dict(prev_x=ps_next.x.pos[0, 0], reward=reward),
        )
