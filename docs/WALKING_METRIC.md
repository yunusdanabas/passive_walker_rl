# Walking Metric (specification v0, proposal)

Status: **proposal, not yet implemented or scored.** It is implemented in Phase 6 of `docs/REVIEW_AND_PLAN.md`. Its open parameters are frozen as **v1** after the Phase 2 FSM reference measurement. Any later change to formulas, weights, scenarios or thresholds creates a new version.

The benchmark measures physical walking, not the shaped training reward. FSM, BC (Torch and JAX) and PPO policies are all scored with the same task, scenarios, seeds and code.

## Principles

1. Measure behavior in physical units: simulated time, net displacement, actuator work.
2. Use a fixed, versioned task shared by every policy.
3. Score robustness on a prescribed scenario matrix, not on whatever a randomization flag happened to produce.
4. Don't let one strong component hide zero progress or a missing gait (geometric means, reliability cap).
5. Always publish the raw per-episode components and uncertainty next to the headline numbers.
6. Report compute cost (training time, inference latency, memory) separately from the score.

## Benchmark configuration (recorded with every result)

| Item | Initial value / rule |
|---|---|
| Horizon `H` | 25 s simulated (2500 control steps at 100 Hz) |
| Target speed schedule `v_target(t)` | **Open.** The research reward uses 0.15 m/s, while the nominal FSM walks at 1.438 m/s (`docs/baseline.md`). Choose after the Phase 2 report; consider several named speed tasks. |
| Target distance | `D_target = ∫ v_target dt` |
| Tracking scale `σ_v`, startup exclusion | Open, fixed per metric version |
| Contact, gait, clearance and posture thresholds | Defined after the Phase 2 contact fix |
| Scenarios, condition weights, seed list | See "Scenario matrix" |
| Energy reference `CoT_ref`, component weights | Fixed from the FSM reference, never retuned per policy |
| Provenance | metric version, obs/action schema version, model-XML hash, git SHA, dependency versions |

## Per-episode raw measurements

**Survival / validity.** `s_e = 1` only if all of the following hold, otherwise `0`:
- the full horizon completes without a fall;
- all observations, actions and rewards are finite;
- declared actuator limits are respected;
- the simulation passes its validity checks.

Also record partial survival `C_e = min(t_end / H, 1)` and the failure class: physical fall, numerical failure, or infrastructure failure. An infrastructure failure is rerun or reported, never scored as a fall.

**Net distance.** `D_e = x_end − x_start` along the downhill plane axis, with `d_e = clip(D_e / D_target, 0, 1)`. Never sum positions or positive-only displacements.

**Speed tracking.**
```
RMSE_v = sqrt( ∫ (v_x(t) − v_target(t))² dt / evaluated_duration )
v_e    = exp(−0.5 · (RMSE_v / σ_v)²)
```
Also report mean speed and signed bias. Overspeed is penalized like stagnation.

**Actuation energy / cost of transport.** Use MuJoCo `data.actuator_force × data.actuator_velocity`, so torque × angular velocity and force × linear velocity both become watts:
```
P⁺(t)   = Σ_j max(actuator_force_j · actuator_velocity_j, 0)
E⁺      = ∫ P⁺ dt
CoT_act = E⁺ / (m_ref · |g| · max(D_e, ε_D))
e_e     = 1 / (1 + CoT_act / CoT_ref)
```
Record positive work, net work and saturation time separately.

Caveat: the slope is tilted gravity, so gravity supplies part of the locomotion energy. This is **actuation** cost of transport in a downhill task, not total energy and not battery consumption. Always report the slope.

**Gait quality.** `g_e = (p_e · a_e · f_e)^(1/3)`:
- `p_e = exp(−0.5 · (pitch_RMS / σ_pitch)²)` (posture);
- `a_e`: alternating stance/swing quality from valid left/right contact transitions, cycle count, and the absence of dragging, hopping or stalls;
- `f_e`: adequate swing-foot clearance relative to the ground; extra height earns nothing.

`a_e` and `f_e` are defined only after the contact observation is fixed (Phase 2) and the FSM's gait has been measured. They must respect this model: sliding knees, an asymmetric relative hip, and alternating knee retraction. They must not reward instantaneous left/right symmetry.

## Condition score

For condition `c`, averaging `D, V, E, G` over **successful** episodes only:
```
R_c = mean(s_e)            over all valid episodes
Q_c = R_c^0.30 · D_c^0.20 · V_c^0.20 · E_c^0.20 · G_c^0.10
```
`Q_c = 0` if no episode succeeds; partial components are still published for diagnosis. The weights are provisional and are fixed in v1.

## Headline scores

- **Nominal Walker Score** = `100 · Q_nominal`, capped by nominal reliability.
- **Robust Walker Score** over the scenario matrix with condition weights `π_c` (equal by default):
  ```
  A     = Π_c Q_c^π_c
  T     = mean of the lowest max(1, ceil(0.2 · n_conditions)) condition scores
  R_min = min_c R_c
  Robust Walker Score = 100 · min(A^0.8 · T^0.2, R_min)
  ```
  A policy that finishes only half its episodes in any one condition can't score above 50.

The two scores are different benchmarks and are never compared with each other.

## Scenario matrix (to be fixed in v1)

1. Nominal: 10° slope, friction 0.9, nominal mass.
2. Gentler and steeper slopes.
3. Lower and higher friction.
4. Mass, damping and actuator-gain variation.
5. A fixed, measured torso push or impulse with a recovery window. This needs the Phase 3 push module.
6. Observation noise and action latency.
7. Combined conditions held out from training.

The existing `PHYSICS_PRESETS` (`fsm/collect.py`) and randomization profiles are candidates, but not yet physically validated ones. Policies are evaluated on paired seeds and identical realized parameters, and every realized condition is saved.

## Qualification badge (provisional)

"Qualified walking" requires:
- ≥ 95 % full-horizon completion in the nominal condition;
- ≥ 90 % in every declared robustness condition;
- the net-distance and speed-tracking tolerances are met;
- a valid alternating gait, with no unreported safety-limit violations.

## Result card (kept for every compared policy)

- Nominal and Robust Walker Scores, the metric version, and confidence intervals (bootstrap at the training-run level; duplicate deterministic episodes don't count as independent evidence).
- Per condition: `R, D, V, E, G, Q`.
- Full and partial survival, net distance, mean speed, RMSE and signed bias.
- Positive and net work, actuation CoT, slope, saturation.
- Gait cycles, stance/swing timing, swing clearance, pitch RMS.
- Push recovery time and failure reasons.
- Training and evaluation seeds, raw trajectories, provenance.
- Engineering metrics, kept separate: training wall-clock, inference p50/p95, throughput, peak memory.
