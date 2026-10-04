# Machines and environment setup

| Machine | Role |
|---|---|
| Ubuntu 24.04 PC | Development, tests, GUI playback, short runs |
| UT Dallas **Juno** servers | Training and evaluation sweeps (details pending) |
| Windows laptop | Editing and planning only; no Python environment |

## Ubuntu 24.04 (development)

Ubuntu 24.04 ships Python 3.12. The pinned versions in `requirements-dev.txt` were recorded on Python 3.11. On the first setup, confirm they install on 3.12; if any pin fails, update `requirements-dev.txt` and re-record `docs/baseline.md`.

```bash
sudo apt install python3-venv python3-dev libgl1 libglfw3   # GLFW/OpenGL for the MuJoCo viewer
git clone https://github.com/yunusdanabas/passive_walker_rl
cd passive_walker_rl
git checkout stabilization
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e ".[all]"            # extras: torch, jax, analysis, dev
pytest -q                          # 15 passed at the end of Phase 1
walker-demo --seconds 5            # opens the MuJoCo viewer
python scripts/bench_env.py        # compare with docs/baseline.md
```

A mamba environment works the same way (`mamba create -n walker python=3.12`, then the same `pip` lines).

Optional: `export PASSIVE_WALKER_HOME=/path/with/space` puts `experiments/` outside the checkout.

## Headless operation

Training and evaluation never need a display.
- Leave `MUJOCO_GL` unset unless you render; nothing on the training path imports the viewer.
- When offscreen rendering is needed for videos on a server, use `MUJOCO_GL=egl` if the node has EGL, or `MUJOCO_GL=osmesa` otherwise.

## UT Dallas Juno (pending)

To be filled in when the cluster details are available. Needed:
- Scheduler (e.g. Slurm), partitions, and time and core limits.
- Module system or container policy (Apptainer/Singularity?), plus the Python/conda availability.
- Storage: home quota, scratch path, and purge policy.
- CPU model and cores per node, and GPU availability.
- Outbound network access from compute nodes (needed for `pip install`).

Planned usage is described in Phase 9 of `docs/REVIEW_AND_PLAN.md`:
- `PASSIVE_WALKER_HOME` on scratch;
- one run directory per job, with full provenance;
- array jobs for seeds;
- thread pinning;
- resumable checkpoints.

## Windows (editing only)

The repository's `.gitattributes` forces LF line endings, so files edited on Windows run unchanged on Linux. Don't run sweeps or training from the Windows checkout.

## Historical archive

The pre-rewrite git history, including datasets, Brax sweep results, the 18 historical PPO checkpoints and their TensorBoard logs, is archived outside the repository. It's a verified git bundle plus a SHA-256 manifest (`passive_walker_rl_ARCHIVE` folder). Restore it with `git clone <bundle>`. Those checkpoints use old observation formats and a PPO trainer with known bugs, so they are historical reference only.
