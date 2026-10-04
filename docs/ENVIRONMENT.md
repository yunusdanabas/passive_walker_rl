# Machines and environment setup

| Machine | Role |
|---|---|
| Ubuntu 24.04 PC | Development, tests, GUI playback, short runs |
| UT Dallas **Juno** cluster | Training and evaluation sweeps (CPU partitions for now) |
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

## UT Dallas Juno (training)

Juno is a Slurm cluster. Confirm the numbers live with `sinfo`; they change as hardware arrives. Official guide: https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/

**Partitions**

| Partition | Limit | Hardware | Use for this project |
|---|---|---|---|
| `normal` (default) | 2 days, ≤ 8 nodes/job | 64 CPU cores, 384 GB per node | **All current training and evaluation** (CPU MuJoCo, PPO with vectorized envs, BC, sweeps) |
| `dev` | 2 h, ≤ 4 nodes | same nodes | Short tests and benchmarks |
| `h200` | 2 days | 2× H200 NVL 141 GB per node | Future MJX/JAX GPU work only |
| `h100`, `a30` | 2 days | H100 (pin the GPU type), A30 24 GB | Future GPU work only |

**Limits and etiquette**
- At most 4 running and 100 submitted jobs per user.
- Always set `--mem` (the default is 64 GB). Fairshare is charged for what is **allocated**, not what is used, so request only what a job needs.
- Login nodes are for editing, `git`, `pip install` and submitting; all computation goes through `sbatch`/`srun`.
- CPU-only jobs don't go on GPU partitions, and long GPU jobs must actually use the GPU.

**Storage and software**
- Code and the conda env go in `~/work` (backed up).
- Run outputs go in `~/scratch` (fast, never backed up, purged after 45 days idle) via `PASSIVE_WALKER_HOME`. Copy final results back to `~/work`.
- `module load miniconda`, then a prefix env with Python 3.12; install with `pip install --no-cache-dir` (home quotas are small).
- Headless rendering: `MUJOCO_GL=egl`.

**Future GPU work (MJX/JAX)**
- `nvidia-smi` utilization is not real load; measure DCGM SM active and power.
- A single small MJX run uses a fraction of an H200, so pack several processes per GPU under MPS, with a per-process `XLA_PYTHON_CLIENT_MEM_FRACTION`.
- Whole-node two-GPU jobs can wait hours; a two-task, one-GPU-per-task shape on 1–2 nodes usually starts sooner. Compare with `sbatch --test-only`.
- Load no `cuda` module for `jax[cuda12]`. GPU results are not bit-reproducible, so compare them with a tolerance.

Planned usage is described in Phase 9 of `docs/REVIEW_AND_PLAN.md`:
- `PASSIVE_WALKER_HOME` on scratch;
- one run directory per job, with full provenance;
- array jobs for seeds;
- thread pinning;
- resumable checkpoints.

Job submission always needs the owner's approval.

## Windows (editing only)

The repository's `.gitattributes` forces LF line endings, so files edited on Windows run unchanged on Linux. Don't run sweeps or training from the Windows checkout.

## Historical archive

The pre-rewrite git history, including datasets, Brax sweep results, the 18 historical PPO checkpoints and their TensorBoard logs, is archived outside the repository. It's a verified git bundle plus a SHA-256 manifest (`passive_walker_rl_ARCHIVE` folder). Restore it with `git clone <bundle>`. Those checkpoints use old observation formats and a PPO trainer with known bugs, so they are historical reference only.
