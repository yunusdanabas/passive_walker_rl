#!/bin/bash
#
# Overnight BC Model Training Sweep
# - Collects FSM data
# - Trains multiple BC models across hyperparameters
# - Evaluates models and generates plots only (no JSON/MD)
# - Saves EVERYTHING into experiments/overnights/<timestamp>/bc/
#

set -e

PROJECT_DIR="/home/yunusdanabas/passive_walker_rl"
BASE_OUT="experiments/overnights"
TS=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$BASE_OUT/$TS/bc"

mkdir -p "$RUN_DIR"

echo ""
echo "======================================================================"
echo "  OVERNIGHT BC SWEEP"
echo "======================================================================"
echo "Output: $RUN_DIR"
echo ""

cd "$PROJECT_DIR"

# Activate environment (preferred)
if command -v mamba >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null || true)"
  mamba activate main || true
fi
PY=python
ENABLE_EXTENDED_REPORTS=${ENABLE_EXTENDED_REPORTS:-0}

# Optional extended outputs (markdown, extra viz). Default OFF to stay PNG-only.
ENABLE_EXTENDED_REPORTS=${ENABLE_EXTENDED_REPORTS:-0}

# 1) Collect FSM data
echo "[1/4] Collecting FSM data..."
DATA_DIR="$RUN_DIR/fsm_data"
mkdir -p "$DATA_DIR"

$PY -m passive_walker.fsm.collect \
  --episodes 100 \
  --duration 25.0 \
  --out "$DATA_DIR" \
  --mode fsm

# Drop metadata JSONs if present (user requested no JSON/MD)
rm -f "$DATA_DIR"/README.json "$DATA_DIR"/meta.json 2>/dev/null || true

# 2) Train BC models across sections and seeds
echo "[2/4] Training BC models (Torch + JAX, multiple sizes)..."
BC_MODELS_DIR="$RUN_DIR/models"
mkdir -p "$BC_MODELS_DIR"

SECTIONS=("hip" "knees" "both")
SEEDS=(123 456)
EPOCHS=40

# Track times (TXT, not JSON/MD)
TIMES_FILE="$RUN_DIR/bc_times.tsv"
echo -e "model\tbackend\tsection\tseed\ttrain_sec" > "$TIMES_FILE"

# Torch: MLPs — Phase 1 small (mlp_small@128), Phase 2 large (mlp_large@512)
model=mlp_small; hidden=128; dropout=0.0
for section in "${SECTIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "  -> torch $model hidden=$hidden section=$section seed=$seed"
    t0=$(date +%s)
    $PY -m passive_walker.bc.training.train \
      --backend torch \
      --section "$section" \
      --data "$DATA_DIR" \
      --epochs $EPOCHS \
      --batch 512 \
      --lr 1e-3 \
      --seed $seed \
      --save-dir "$BC_MODELS_DIR" \
      --model "$model" \
      --hidden $hidden \
      --dropout $dropout \
      --gpu || true
    t1=$(date +%s)
    echo -e "torch_${section}_${model}_h${hidden}_seed${seed}\ttorch\t${section}\t${seed}\t$((t1 - t0))" >> "$TIMES_FILE"
  done
done
model=mlp_large; hidden=512; dropout=0.1
for section in "${SECTIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "  -> torch $model hidden=$hidden section=$section seed=$seed"
    t0=$(date +%s)
    $PY -m passive_walker.bc.training.train \
      --backend torch \
      --section "$section" \
      --data "$DATA_DIR" \
      --epochs $EPOCHS \
      --batch 512 \
      --lr 1e-3 \
      --seed $seed \
      --save-dir "$BC_MODELS_DIR" \
      --model "$model" \
      --hidden $hidden \
      --dropout $dropout \
      --gpu || true
    t1=$(date +%s)
    echo -e "torch_${section}_${model}_h${hidden}_seed${seed}\ttorch\t${section}\t${seed}\t$((t1 - t0))" >> "$TIMES_FILE"
  done
done

# Torch Temporal: train ALL small first (128), then large (256)
# Phase 1: hidden=128
for tmodel in lstm gru; do
  hsz=128
  for section in "${SECTIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "  -> torch temporal ${tmodel} hidden=${hsz} section=$section seed=$seed"
      t0=$(date +%s)
      $PY - <<PYTMP
from passive_walker.bc.config import TemporalTrainingConfig
from passive_walker.bc.training import train_temporal_torch

cfg = TemporalTrainingConfig(
    backend='torch',
    section='${section}',
    data_dir='${DATA_DIR}',
    epochs=${EPOCHS},
    batch_size=256,
    learning_rate=1e-3,
    seed=${seed},
    model_type='${tmodel}',
    hidden_size=${hsz},
    num_layers=1,
    dropout=0.1,
    checkpoint_dir='${BC_MODELS_DIR}',
)
train_temporal_torch(cfg)
PYTMP
      t1=$(date +%s)
      echo -e "torch_temporal_${tmodel}_h${hsz}_${section}_seed${seed}\ttorch-temporal\t${section}\t${seed}\t$((t1 - t0))" >> "$TIMES_FILE"
    done
  done
done
# Phase 2: hidden=256
for tmodel in lstm gru; do
  hsz=256
  for section in "${SECTIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "  -> torch temporal ${tmodel} hidden=${hsz} section=$section seed=$seed"
      t0=$(date +%s)
      $PY - <<PYTMP
from passive_walker.bc.config import TemporalTrainingConfig
from passive_walker.bc.training import train_temporal_torch

cfg = TemporalTrainingConfig(
    backend='torch',
    section='${section}',
    data_dir='${DATA_DIR}',
    epochs=${EPOCHS},
    batch_size=256,
    learning_rate=1e-3,
    seed=${seed},
    model_type='${tmodel}',
    hidden_size=${hsz},
    num_layers=1,
    dropout=0.1,
    checkpoint_dir='${BC_MODELS_DIR}',
)
train_temporal_torch(cfg)
PYTMP
      t1=$(date +%s)
      echo -e "torch_temporal_${tmodel}_h${hsz}_${section}_seed${seed}\ttorch-temporal\t${section}\t${seed}\t$((t1 - t0))" >> "$TIMES_FILE"
    done
  done
done

# JAX Temporal: ALL small first (128), then large (256)
# Phase 1: hidden=128
for tmodel in lstm gru; do
  hsz=128
  for section in "${SECTIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "  -> jax temporal ${tmodel} hidden=${hsz} section=$section seed=$seed"
      t0=$(date +%s)
      $PY - <<PYTMP
from passive_walker.bc.config import TemporalTrainingConfig
from passive_walker.bc.training import train_temporal_jax

cfg = TemporalTrainingConfig(
    backend='jax',
    section='${section}',
    data_dir='${DATA_DIR}',
    epochs=${EPOCHS},
    batch_size=256,
    learning_rate=1e-3,
    seed=${seed},
    model_type='${tmodel}',
    hidden_size=${hsz},
    num_layers=1,
    dropout=0.1,
    checkpoint_dir='${BC_MODELS_DIR}',
)
train_temporal_jax(cfg)
PYTMP
      t1=$(date +%s)
      echo -e "jax_temporal_${tmodel}_h${hsz}_${section}_seed${seed}\tjax-temporal\t${section}\t${seed}\t$((t1 - t0))" >> "$TIMES_FILE"
    done
  done
done
# Phase 2: hidden=256
for tmodel in lstm gru; do
  hsz=256
  for section in "${SECTIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "  -> jax temporal ${tmodel} hidden=${hsz} section=$section seed=$seed"
      t0=$(date +%s)
      $PY - <<PYTMP
from passive_walker.bc.config import TemporalTrainingConfig
from passive_walker.bc.training import train_temporal_jax

cfg = TemporalTrainingConfig(
    backend='jax',
    section='${section}',
    data_dir='${DATA_DIR}',
    epochs=${EPOCHS},
    batch_size=256,
    learning_rate=1e-3,
    seed=${seed},
    model_type='${tmodel}',
    hidden_size=${hsz},
    num_layers=1,
    dropout=0.1,
    checkpoint_dir='${BC_MODELS_DIR}',
)
train_temporal_jax(cfg)
PYTMP
      t1=$(date +%s)
      echo -e "jax_temporal_${tmodel}_h${hsz}_${section}_seed${seed}\tjax-temporal\t${section}\t${seed}\t$((t1 - t0))" >> "$TIMES_FILE"
    done
  done
done

# JAX MLP: ALL small widths first (128), then large (256)
# Phase 1: width=128
width=128
for depth in 1 2; do
  for section in "${SECTIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "  -> jax width=$width depth=$depth section=$section seed=$seed"
      t0=$(date +%s)
      $PY -m passive_walker.bc.training.train \
        --backend jax \
        --section "$section" \
        --data "$DATA_DIR" \
        --epochs $EPOCHS \
        --batch 512 \
        --lr 1e-3 \
        --seed $seed \
        --save-dir "$BC_MODELS_DIR" \
        --width $width \
        --depth $depth || true
      t1=$(date +%s)
      echo -e "jax_${section}_w${width}d${depth}_seed${seed}\tjax\t${section}\t${seed}\t$((t1 - t0))" >> "$TIMES_FILE"
    done
  done
done
# Phase 2: width=256
width=256
for depth in 1 2; do
  for section in "${SECTIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "  -> jax width=$width depth=$depth section=$section seed=$seed"
      t0=$(date +%s)
      $PY -m passive_walker.bc.training.train \
        --backend jax \
        --section "$section" \
        --data "$DATA_DIR" \
        --epochs $EPOCHS \
        --batch 512 \
        --lr 1e-3 \
        --seed $seed \
        --save-dir "$BC_MODELS_DIR" \
        --width $width \
        --depth $depth || true
      t1=$(date +%s)
      echo -e "jax_${section}_w${width}d${depth}_seed${seed}\tjax\t${section}\t${seed}\t$((t1 - t0))" >> "$TIMES_FILE"
    done
  done
done

# 3) Evaluate each model and generate plots-only comparison
echo "[3/4] Evaluating models and generating plots..."
EVAL_DIR="$RUN_DIR/eval"
mkdir -p "$EVAL_DIR"

# Inline evaluation runner (plots only, no JSON/MD outputs)
$PY - << 'PY'
import os, json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("experiments/overnights")

# Find latest timestamp dir by scanning upwards from CWD
ts_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()], key=lambda p: p.name)
if not ts_dirs:
    print("No overnights found.")
    sys.exit(0)

ts_dir = ts_dirs[-1] / "bc"
models_dir = ts_dir / "models"
eval_dir = ts_dir / "eval"
eval_dir.mkdir(parents=True, exist_ok=True)

def eval_model(ckpt: Path):
    meta = ckpt.with_suffix("")  # remove .pt
    try:
        from passive_walker.core.env import PassiveWalkerEnv
    except Exception as e:
        print(f"Env error for {ckpt.name}: {e}")
        return None

    if ckpt.suffix == ".pt":
        # Torch path: checkpoint payload has meta and normalizer
        import torch
        from passive_walker.bc.utils import Normalizer
        from passive_walker.bc.models.models_torch import TorchMLP, TorchMLPLarge
        from passive_walker.bc.models.temporal_torch import create_temporal_model
        payload = torch.load(str(ckpt), map_location='cpu')
        meta = payload.get('meta', {})
        in_dim = meta.get('input_dim') or meta.get('in_dim'); out_dim = meta.get('output_dim') or meta.get('out_dim')
        model_kind = meta.get('model')
        if model_kind in ('mlp_small','mlp_large'):
            hidden = meta.get('hidden', 512)
            if model_kind == 'mlp_small':
                model = TorchMLP(in_dim=in_dim, out_dim=out_dim, hidden=hidden)
            else:
                model = TorchMLPLarge(in_dim=in_dim, out_dim=out_dim, hidden=hidden, dropout=meta.get('dropout', 0.1))
            model.load_state_dict(payload['model_state_dict'])
            model.eval()
        elif meta.get('model_type') in ('lstm','gru','bilstm'):
            # Temporal torch model
            ttype = meta['model_type']
            hidden_size = meta.get('hidden_size', 128)
            num_layers = meta.get('num_layers', 1)
            bidirectional = meta.get('bidirectional', False)
            dropout = meta.get('dropout', 0.1)
            model = create_temporal_model(ttype, in_dim, out_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
            model.load_state_dict(payload['model_state_dict'])
            model.eval()
        else:
            return None
        normalizer = Normalizer(mean=np.array(payload.get('normalizer_mean', np.zeros(in_dim))), std=np.array(payload.get('normalizer_std', np.ones(in_dim))))
        section = meta.get('section', 'both')

        def assemble_action(section, model_out):
            try:
                if section == 'hip':
                    # Expect length 1
                    v0 = float(model_out[0]) if len(model_out) >= 1 else 0.0
                    return np.array([v0, 0.0, 0.0], dtype=np.float32)
                elif section == 'knees':
                    # Expect length 2
                    v1 = float(model_out[0]) if len(model_out) >= 1 else 0.0
                    v2 = float(model_out[1]) if len(model_out) >= 2 else 0.0
                    return np.array([0.0, v1, v2], dtype=np.float32)
                else:
                    # both or both-adv expect length 3
                    v0 = float(model_out[0]) if len(model_out) >= 1 else 0.0
                    v1 = float(model_out[1]) if len(model_out) >= 2 else 0.0
                    v2 = float(model_out[2]) if len(model_out) >= 3 else 0.0
                    return np.array([v0, v1, v2], dtype=np.float32)
            except Exception:
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        env = PassiveWalkerEnv(mode='research')
        returns, lengths = [], []
        for _ in range(10):
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0
            while env.data.time < 25.0:
                x = torch.as_tensor(normalizer.encode(torch.as_tensor(obs, dtype=torch.float32)[None, :]), dtype=torch.float32)
                with torch.no_grad():
                    out = model(x)
                    if isinstance(out, tuple):
                        out = out[0]
                    model_out = out.squeeze(0).numpy()
                act = assemble_action(section, model_out)
                obs, r, done, info = env.step(act)
                ep_ret += r
                ep_len += 1
                if done:
                    break
            returns.append(ep_ret)
            lengths.append(ep_len)
        env.close()
        return {
            'name': ckpt.stem,
            'section': section,
            'backend': 'torch',
            'seed': meta.get('seed', -1),
            'avg_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'avg_length': float(np.mean(lengths)),
            'success_rate': float(np.mean([l > 100 for l in lengths]))
        }
    elif ckpt.suffix == ".eqx":
        # JAX path: read meta json (will be deleted after), normalizer from meta
        try:
            import jax
            import jax.numpy as jnp
            import equinox as eqx
            from passive_walker.bc.models.models_jax import load_eqx_with_template
            from passive_walker.bc.models.temporal_jax import load_temporal_model_with_template
            from passive_walker.bc.utils import Normalizer
        except Exception as e:
            print(f"JAX eval skip {ckpt.name}: {e}")
            return None
        meta_json = Path(str(ckpt).replace('.eqx', '_meta.json'))
        if not meta_json.exists():
            return None
        with open(meta_json, 'r') as f:
            meta = json.load(f)
        in_dim = meta.get('in_dim'); out_dim = meta.get('out_dim')
        if meta.get('model_type') in ('lstm','gru'):
            model = load_temporal_model_with_template(str(ckpt), meta['model_type'], in_dim, out_dim, hidden_size=meta.get('hidden_size', 128), dropout_rate=meta.get('dropout', 0.1))
        else:
            model = load_eqx_with_template(str(ckpt), in_dim, out_dim, width=meta.get('hidden', 128), depth=meta.get('depth', 2))
        norm = meta.get('normalizer')
        if norm and 'mean' in norm and 'std' in norm:
            normalizer = Normalizer(mean=np.array(norm['mean']), std=np.array(norm['std']))
        else:
            normalizer = Normalizer(mean=np.zeros(in_dim, dtype=np.float32), std=np.ones(in_dim, dtype=np.float32))
        section = meta.get('section', 'both')

        def assemble_action(section, model_out):
            try:
                if section == 'hip':
                    v0 = float(model_out[0]) if model_out.shape[0] >= 1 else 0.0
                    return np.array([v0, 0.0, 0.0], dtype=np.float32)
                elif section == 'knees':
                    v1 = float(model_out[0]) if model_out.shape[0] >= 1 else 0.0
                    v2 = float(model_out[1]) if model_out.shape[0] >= 2 else 0.0
                    return np.array([0.0, v1, v2], dtype=np.float32)
                else:
                    v0 = float(model_out[0]) if model_out.shape[0] >= 1 else 0.0
                    v1 = float(model_out[1]) if model_out.shape[0] >= 2 else 0.0
                    v2 = float(model_out[2]) if model_out.shape[0] >= 3 else 0.0
                    return np.array([v0, v1, v2], dtype=np.float32)
            except Exception:
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        env = PassiveWalkerEnv(mode='research')
        returns, lengths = [], []
        for _ in range(10):
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0
            while env.data.time < 25.0:
                x = normalizer.apply(obs[None, :]).astype(np.float32)
                xj = jnp.asarray(x)
                out = model(xj)
                if isinstance(out, tuple):
                    out = out[0]
                model_out = np.asarray(out).squeeze(0)
                act = assemble_action(section, model_out)
                obs, r, done, info = env.step(act)
                ep_ret += r
                ep_len += 1
                if done:
                    break
            returns.append(ep_ret)
            lengths.append(ep_len)
        env.close()
        return {
            'name': ckpt.stem,
            'section': section,
            'backend': 'jax',
            'seed': meta.get('seed', -1),
            'avg_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'avg_length': float(np.mean(lengths)),
            'success_rate': float(np.mean([l > 100 for l in lengths]))
        }
    else:
        return None

ckpts = sorted(list(models_dir.glob('*.pt')) + list(models_dir.glob('*.eqx')))
results = []
for ck in ckpts:
    r = eval_model(ck)
    if r:
        print(f"Eval {ck.name}: return={r['avg_return']:.2f} len={r['avg_length']:.1f} succ={r['success_rate']:.2f}")
        results.append(r)

if not results:
    print("No results to plot.")
    sys.exit(0)

# Figures
plt.rcParams.update({'figure.figsize': (12, 8)})

# 1) Average return by backend
backends = ['torch', 'jax']
fig, ax = plt.subplots()
vals = []
errs = []
labels = []
for b in backends:
    group = [r['avg_return'] for r in results if r.get('backend')==b]
    if group:
        labels.append(b)
        vals.append(np.mean(group))
        errs.append(np.std(group))
if vals:
    ax.bar(labels, vals, yerr=errs, capsize=5, alpha=0.8)
    ax.set_title('BC: Average Return by Backend')
    ax.set_ylabel('Average Return')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(eval_dir / 'bc_avg_return_by_backend.png', dpi=150)
    plt.close(fig)

# 2) Average return by section
sections = ['hip', 'knees', 'both']
fig, ax = plt.subplots()
vals = []
errs = []
labels = []
for s in sections:
    group = [r['avg_return'] for r in results if r['section']==s]
    if group:
        labels.append(s)
        vals.append(np.mean(group))
        errs.append(np.std(group))
if vals:
    ax.bar(labels, vals, yerr=errs, capsize=5, alpha=0.8)
    ax.set_title('BC: Average Return by Section')
    ax.set_ylabel('Average Return')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(eval_dir / 'bc_avg_return_by_section.png', dpi=150)
    plt.close(fig)

# 3) Success rate by section
fig, ax = plt.subplots()
vals = []
errs = []
labels = []
for s in sections:
    group = [r['success_rate'] for r in results if r['section']==s]
    if group:
        labels.append(s)
        vals.append(np.mean(group))
        errs.append(np.std(group))
if vals:
    ax.bar(labels, vals, yerr=errs, capsize=5, alpha=0.8)
    ax.set_ylim(0,1)
    ax.set_title('BC: Success Rate by Section')
    ax.set_ylabel('Success Rate')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(eval_dir / 'bc_success_by_section.png', dpi=150)
    plt.close(fig)

# 4) All models return ranking
fig, ax = plt.subplots(figsize=(12, max(4, 0.4*len(results))))
names = [r['name'] for r in results]
rets = [r['avg_return'] for r in results]
ax.barh(names, rets, alpha=0.8)
ax.set_title('BC: Average Return per Model')
ax.set_xlabel('Average Return')
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig(eval_dir / 'bc_models_avg_return.png', dpi=150)
plt.close(fig)

# 5) Training time chart from TSV
times_file = ts_dir / 'bc_times.tsv'
if times_file.exists():
    rows = []
    with open(times_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0: continue
            parts = line.strip().split('\t')
            if len(parts) == 5:
                rows.append((parts[0], parts[1], parts[2], parts[3], float(parts[4])))
    if rows:
        rows = sorted(rows, key=lambda x: x[4], reverse=True)
        labels = [r[0] for r in rows]
        secs = [r[4] for r in rows]
        fig, ax = plt.subplots(figsize=(12, max(4, 0.35*len(rows))))
        ax.barh(labels, secs, alpha=0.8)
        ax.set_title('BC: Training Time per Model (seconds)')
        ax.set_xlabel('Seconds')
        ax.grid(axis='x', alpha=0.3)
        fig.tight_layout()
        fig.savefig(eval_dir / 'bc_training_times.png', dpi=150)
        plt.close(fig)

print(f"Saved plots to {eval_dir}")
PY

# Comprehensive, organized evaluation (multi-panel figures)
echo "[3b/4] Generating comprehensive organized evaluation..."
$PY tools/evaluation/bc_comprehensive_eval.py \
  --models_dir "$BC_MODELS_DIR" \
  --out "$EVAL_DIR" \
  --episodes 5 || true

# Organize evaluation outputs into subdirectories
echo "Organizing evaluation outputs..."
mkdir -p "$EVAL_DIR/figures" "$EVAL_DIR/reports" "$EVAL_DIR/tables" "$EVAL_DIR/metrics"
# Move top-level PNGs into figures (comprehensive evaluator already writes into figures/)
find "$EVAL_DIR" -maxdepth 1 -type f -name "*.png" -exec mv {} "$EVAL_DIR/figures/" \; 2>/dev/null || true
# Move tables and metrics
[ -f "$RUN_DIR/bc_times.tsv" ] && mv "$RUN_DIR/bc_times.tsv" "$EVAL_DIR/tables/bc_times.tsv" || true
find "$EVAL_DIR" -maxdepth 1 -type f \( -name "*.csv" -o -name "*.tsv" \) -exec mv {} "$EVAL_DIR/tables/" \; 2>/dev/null || true
find "$EVAL_DIR" -maxdepth 1 -type f -name "*.json" -exec mv {} "$EVAL_DIR/metrics/" \; 2>/dev/null || true
# Move reports
find "$EVAL_DIR" -maxdepth 1 -type f -name "*.md" -exec mv {} "$EVAL_DIR/reports/" \; 2>/dev/null || true

# Optional extended reports (safe, non-blocking)
if [ "$ENABLE_EXTENDED_REPORTS" -eq 1 ]; then
  echo "[Optional] Generating extended reports (markdown, extra viz)..."
  $PY - << 'PY'
import os, sys, traceback
from pathlib import Path

ROOT = Path("experiments/overnights")
ts_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()], key=lambda p: p.name)
if not ts_dirs:
    sys.exit(0)
ts_dir = ts_dirs[-1] / "bc"
eval_dir = ts_dir / "eval"
md_path = eval_dir / "SUMMARY.md"

try:
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(md_path, 'w') as f:
        f.write("# BC Extended Summary\n\n")
        f.write("This optional report is generated after the core PNG pipeline.\n\n")
        for img in sorted(eval_dir.glob('*.png')):
            f.write(f"![{img.stem}]({img.name})\n\n")
    print("Extended markdown summary written:", md_path)
except Exception:
    traceback.print_exc()
PY
fi

# 4) Final structure and verification
echo "[4/4] Sweep complete. Verifying outputs..."
echo "Directory layout (BC):"
echo "  $RUN_DIR/"
echo "    ├── fsm_data/"
echo "    ├── models/"
echo "    └── eval/"
echo "        ├── figures/  # PNG plots"
echo "        ├── tables/   # CSV/TSV"
echo "        ├── metrics/  # JSON"
echo "        └── reports/  # Markdown"

# Verification summary
echo ""
echo "Output verification:"
N_FIGS=$(find "$EVAL_DIR/figures" -name "*.png" 2>/dev/null | wc -l || echo 0)
N_TABLES=$(find "$EVAL_DIR/tables" -type f 2>/dev/null | wc -l || echo 0)
N_MODELS=$(find "$BC_MODELS_DIR" -type f \( -name "*.pt" -o -name "*.eqx" \) 2>/dev/null | wc -l || echo 0)
echo "  Models: $N_MODELS"
echo "  Figures: $N_FIGS"
echo "  Tables: $N_TABLES"

# Cleanup section (preserve organized file types)
find "$RUN_DIR" -type f ! -name "*.png" ! -name "*.md" ! -name "*.json" ! -name "*.csv" ! -name "*.tsv" -delete 2>/dev/null || true
find "$RUN_DIR" -type d -empty -delete 2>/dev/null || true

echo ""
echo "Done. See results in: $RUN_DIR"


