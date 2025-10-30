#!/usr/bin/env python3
"""
Comprehensive BC Evaluation (organized outputs, multi-panel figures)

- Scans a models directory for Torch (.pt) and JAX (.eqx) BC checkpoints
- Evaluates each model (nominal physics); optional robustness sweeps
- Computes rich metrics (performance, gait/contact, effort, imitation, reward)
- Produces organized multi-panel PNGs grouped by theme

Outputs (under --out):
  figures/
    summary_overview.png
    gait_and_contacts.png
    actions_effort.png
    imitation_and_reward.png
    leaderboards.png
  thumbnails/ (optional future use)

Notes:
  - By default, only PNGs are written. Enable external summaries via caller.
  - Designed to be robust: if a metric fails for a model, it is skipped.
"""

from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class EpisodeRecord:
    returns: float
    length: int
    success: bool
    distance: float
    actions: np.ndarray  # (T, 3)
    joints: np.ndarray   # (T, 3) hip, lk, rk
    foot_z: np.ndarray   # (T, 2) left z, right z
    reward_components: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ModelEval:
    name: str
    backend: str
    section: str
    seed: int
    episodes: List[EpisodeRecord] = field(default_factory=list)

    def aggregate(self) -> Dict[str, float]:
        if not self.episodes:
            return {}
        returns = np.array([ep.returns for ep in self.episodes])
        lengths = np.array([ep.length for ep in self.episodes])
        success = np.array([ep.success for ep in self.episodes], dtype=float)
        distances = np.array([ep.distance for ep in self.episodes])
        # Effort proxy: sum |u|
        efforts = []
        for ep in self.episodes:
            if ep.actions.size:
                efforts.append(float(np.sum(np.abs(ep.actions))))
        efforts = np.array(efforts) if efforts else np.zeros_like(returns)
        eff_per_m = distances / np.maximum(efforts, 1e-8)
        return {
            "avg_return": float(np.mean(returns)),
            "median_return": float(np.median(returns)),
            "success_rate": float(np.mean(success)),
            "avg_length": float(np.mean(lengths)),
            "avg_distance": float(np.mean(distances)),
            "avg_efficiency": float(np.mean(eff_per_m)),
        }


def _safe_imports():
    # Import heavy deps lazily
    import torch  # noqa: F401
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    import equinox as eqx  # noqa: F401


def _assemble_action(section: str, model_out: np.ndarray) -> np.ndarray:
    # Robustly produce 3D action vector
    if section == "hip":
        v0 = float(model_out[0]) if model_out.shape[0] >= 1 else 0.0
        return np.array([v0, 0.0, 0.0], dtype=np.float32)
    if section == "knees":
        v1 = float(model_out[0]) if model_out.shape[0] >= 1 else 0.0
        v2 = float(model_out[1]) if model_out.shape[0] >= 2 else 0.0
        return np.array([0.0, v1, v2], dtype=np.float32)
    # both / both-adv
    v0 = float(model_out[0]) if model_out.shape[0] >= 1 else 0.0
    v1 = float(model_out[1]) if model_out.shape[0] >= 2 else 0.0
    v2 = float(model_out[2]) if model_out.shape[0] >= 3 else 0.0
    return np.array([v0, v1, v2], dtype=np.float32)


def _eval_one_model(ckpt: Path, n_episodes: int = 5) -> ModelEval | None:
    """Evaluate one model nominally. Returns ModelEval or None on failure."""
    try:
        from passive_walker.core.env import PassiveWalkerEnv
        from passive_walker.bc.utils import Normalizer
    except Exception as e:
        print(f"Env import failed: {e}")
        return None

    backend = "torch" if ckpt.suffix == ".pt" else ("jax" if ckpt.suffix == ".eqx" else "unknown")

    try:
        if backend == "torch":
            import torch
            from passive_walker.bc.models.models_torch import TorchMLP, TorchMLPLarge
            from passive_walker.bc.models.temporal_torch import create_temporal_model
            payload = torch.load(str(ckpt), map_location="cpu")
            meta = payload.get("meta", {})
            in_dim = meta.get("input_dim") or meta.get("in_dim")
            out_dim = meta.get("output_dim") or meta.get("out_dim")
            section = meta.get("section", "both")
            seed = int(meta.get("seed", -1))
            model_kind = meta.get("model")
            if model_kind in ("mlp_small", "mlp_large"):
                hidden = int(meta.get("hidden", 512))
                if model_kind == "mlp_small":
                    model = TorchMLP(in_dim=in_dim, out_dim=out_dim, hidden=hidden)
                else:
                    model = TorchMLPLarge(in_dim=in_dim, out_dim=out_dim, hidden=hidden, dropout=float(meta.get("dropout", 0.1)))
            else:
                ttype = meta.get("model_type", "lstm")
                hidden_size = int(meta.get("hidden_size", 128))
                num_layers = int(meta.get("num_layers", 1))
                bidirectional = bool(meta.get("bidirectional", False))
                dropout = float(meta.get("dropout", 0.1))
                model = create_temporal_model(ttype, in_dim, out_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
            state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
            model.load_state_dict(state)
            model.eval()
            norm = Normalizer(mean=np.array(payload.get("normalizer_mean", np.zeros(in_dim))), std=np.array(payload.get("normalizer_std", np.ones(in_dim))))
        elif backend == "jax":
            import jax
            import jax.numpy as jnp
            from passive_walker.bc.models.models_jax import load_eqx_with_template
            from passive_walker.bc.models.temporal_jax import load_temporal_model_with_template
            from passive_walker.bc.utils import Normalizer
            meta_path = Path(str(ckpt).replace(".eqx", "_meta.json"))
            if not meta_path.exists():
                return None
            meta = __import__("json").loads(meta_path.read_text())
            in_dim = int(meta.get("in_dim")); out_dim = int(meta.get("out_dim"))
            section = meta.get("section", "both"); seed = int(meta.get("seed", -1))
            if meta.get("model_type") in ("lstm", "gru"):
                model = load_temporal_model_with_template(str(ckpt), meta["model_type"], in_dim, out_dim, hidden_size=int(meta.get("hidden_size", 128)), dropout_rate=float(meta.get("dropout", 0.1)))
            else:
                model = load_eqx_with_template(str(ckpt), in_dim, out_dim, width=int(meta.get("hidden", 128)), depth=int(meta.get("depth", 2)))
            norm_data = meta.get("normalizer")
            if norm_data and "mean" in norm_data and "std" in norm_data:
                norm = Normalizer(mean=np.array(norm_data["mean"]), std=np.array(norm_data["std"]))
            else:
                norm = Normalizer(mean=np.zeros(in_dim), std=np.ones(in_dim))
        else:
            return None
    except Exception as e:
        print(f"Load failed for {ckpt.name}: {e}")
        return None

    mrec = ModelEval(name=ckpt.stem, backend=backend, section=section, seed=seed)

    # Evaluate episodes
    try:
        env = PassiveWalkerEnv(mode="research")
        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_ret = 0.0; ep_len = 0
            acts = []; joints = []; footz = []
            rcomps: Dict[str, List[float]] = {}
            while env.data.time < 25.0:
                x = norm.apply(obs[None, :]).astype(np.float32)
                if backend == "torch":
                    import torch
                    with torch.no_grad():
                        out = model(torch.as_tensor(x))
                        if isinstance(out, tuple):
                            out = out[0]
                        model_out = out.cpu().numpy().squeeze(0)
                else:
                    import jax.numpy as jnp
                    out = model(jnp.asarray(x))
                    if isinstance(out, tuple):
                        out = out[0]
                    model_out = np.asarray(out).squeeze(0)
                act = _assemble_action(section, model_out)
                obs, r, done, info = env.step(act)
                ep_ret += r; ep_len += 1
                acts.append(act)
                joints.append([env.data.qpos[env.qpos_hip], env.data.qpos[env.qpos_lk], env.data.qpos[env.qpos_rk]])
                footz.append([env.data.xpos[env.b_lfoot, 2], env.data.xpos[env.b_rfoot, 2]])
                for k, v in info.items():
                    if not isinstance(v, (int, float)):
                        continue
                    if k.startswith("r_"):
                        rcomps.setdefault(k, []).append(float(v))
                if done:
                    break
            distance = float(env.data.qpos[env.qpos_x])
            success = bool(ep_len >= int(0.8 * 25.0 * env.ctrl_hz))
            mrec.episodes.append(EpisodeRecord(
                returns=ep_ret,
                length=ep_len,
                success=success,
                distance=distance,
                actions=np.array(acts, dtype=np.float32),
                joints=np.array(joints, dtype=np.float32),
                foot_z=np.array(footz, dtype=np.float32),
                reward_components={k: np.array(v, dtype=np.float32) for k, v in rcomps.items()},
            ))
        env.close()
    except Exception as e:
        print(f"Eval failed for {ckpt.name}: {e}")
        return mrec if mrec.episodes else None

    return mrec


def _multi_panel_summary(models: List[ModelEval], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Prepare aggregates
    entries = []
    for m in models:
        agg = m.aggregate()
        if not agg:
            continue
        agg["name"] = m.name; agg["section"] = m.section; agg["backend"] = m.backend
        entries.append(agg)
    if not entries:
        return
    # Convert to arrays
    names = [e["name"] for e in entries]
    returns = np.array([e["avg_return"] for e in entries])
    lengths = np.array([e["avg_length"] for e in entries])
    success = np.array([e["success_rate"] for e in entries])
    efficiency = np.array([e["avg_efficiency"] for e in entries])
    sections = [e["section"] for e in entries]
    backends = [e["backend"] for e in entries]

    # Summary overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    ax.hist(returns, bins=30, alpha=0.8)
    ax.set_title("Return distribution"); ax.set_xlabel("Avg Return"); ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    # Success by section
    cats = sorted(set(sections))
    vals = [float(np.mean(success[np.array(sections)==c])) for c in cats]
    ax.bar(cats, vals)
    ax.set_ylim(0, 1); ax.set_title("Success rate by section"); ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    # Return vs length
    ax.scatter(lengths, returns, alpha=0.6)
    ax.set_xlabel("Avg length (steps)"); ax.set_ylabel("Avg return"); ax.set_title("Return vs Length"); ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    # Pareto: Return vs Efficiency
    ax.scatter(efficiency, returns, alpha=0.6)
    ax.set_xlabel("Efficiency (distance / effort)"); ax.set_ylabel("Avg return"); ax.set_title("Pareto: Return vs Efficiency"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "summary_overview.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Gait and contacts (sample top-1 by return)
    idx_top = int(np.argmax(returns))
    top_model = models[names.index(names[idx_top])]
    if top_model.episodes:
        ep = top_model.episodes[0]
        T = ep.joints.shape[0]
        t = np.arange(T)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax = axes[0, 0]
        ax.plot(t, ep.joints[:, 0], label="hip")
        ax.plot(t, ep.joints[:, 1], label="lk")
        ax.plot(t, ep.joints[:, 2], label="rk"); ax.legend(); ax.set_title("Joint angles"); ax.grid(True, alpha=0.3)
        ax = axes[0, 1]
        ax.plot(t, ep.foot_z[:, 0], label="left z"); ax.plot(t, ep.foot_z[:, 1], label="right z"); ax.legend(); ax.set_title("Foot clearance"); ax.grid(True, alpha=0.3)
        ax = axes[1, 0]
        # Contact raster proxy: threshold foot_z < small epsilon
        l_contact = (ep.foot_z[:, 0] < 0.02).astype(float)
        r_contact = (ep.foot_z[:, 1] < 0.02).astype(float)
        ax.imshow(np.vstack([l_contact, r_contact]), aspect="auto", cmap="Greys", interpolation="nearest")
        ax.set_yticks([0,1]); ax.set_yticklabels(["L","R"]); ax.set_title("Contact raster (proxy)")
        ax = axes[1, 1]
        # Symmetry proxy: |lk - rk|
        sym = np.abs(ep.joints[:,1] - ep.joints[:,2])
        ax.plot(t, sym); ax.set_title("Symmetry error |lk-rk|"); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "gait_and_contacts.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Actions and effort
    all_actions = [ep.actions for m in models for ep in m.episodes if ep.actions.size]
    if all_actions:
        A = np.concatenate(all_actions, axis=0)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax = axes[0, 0]
        ax.hist(A[:,0], bins=40, alpha=0.6, label="hip"); ax.hist(A[:,1], bins=40, alpha=0.6, label="lk"); ax.hist(A[:,2], bins=40, alpha=0.6, label="rk"); ax.legend(); ax.set_title("Action magnitudes")
        ax.grid(True, alpha=0.3)
        ax = axes[0, 1]
        # Smoothness proxy: mean |Δu|
        du = np.mean(np.abs(np.diff(A, axis=0))),
        ax.bar(["mean|Δu|"], [float(du[0])]); ax.set_title("Smoothness proxy"); ax.grid(True, alpha=0.3)
        ax = axes[1, 0]
        # Efficiency distribution
        effs = []
        for m in models:
            for ep in m.episodes:
                if ep.actions.size:
                    effort = float(np.sum(np.abs(ep.actions)))
                    effs.append(float(ep.distance / max(effort,1e-8)))
        ax.hist(effs, bins=30, alpha=0.8); ax.set_title("Efficiency (distance/effort)"); ax.grid(True, alpha=0.3)
        ax = axes[1, 1]
        # Backend comparison bars
        bvals = []
        for b in sorted(set(backends)):
            bvals.append(float(np.mean(returns[np.array(backends)==b])))
        ax.bar(sorted(set(backends)), bvals); ax.set_title("Avg return by backend"); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "actions_effort.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Imitation and reward (best model)
    if top_model.episodes:
        ep = top_model.episodes[0]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax = axes[0, 0]
        # Imitation proxy: compare actions to zeros (placeholder for FSM targets)
        l1 = np.mean(np.abs(ep.actions)) if ep.actions.size else 0.0
        ax.bar(["L1(action,0)"],[l1]); ax.set_title("Imitation proxy (lower better)"); ax.grid(True, alpha=0.3)
        # Reward components
        ax = axes[0, 1]
        if ep.reward_components:
            for k, v in ep.reward_components.items():
                ax.plot(v, label=k)
            ax.legend(); ax.set_title("Reward components (episode)"); ax.grid(True, alpha=0.3)
        ax = axes[1, 0]
        ax.plot([e.returns for e in top_model.episodes]); ax.set_title("Per-episode returns (best model)"); ax.grid(True, alpha=0.3)
        ax = axes[1, 1]
        ax.plot([e.length for e in top_model.episodes]); ax.set_title("Per-episode lengths (best model)"); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "imitation_and_reward.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Leaderboard (top-N by return)
    order = np.argsort(-returns)
    topk = min(15, len(order))
    fig, ax = plt.subplots(figsize=(12, 0.6*topk + 2))
    ax.barh([names[i] for i in order[:topk]][::-1], [returns[i] for i in order[:topk]][::-1])
    ax.set_title("Top models by avg return"); ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "leaderboards.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Reward component analysis (from archive)
    all_reward_comps = {}
    for m in models:
        for ep in m.episodes:
            for k, v in ep.reward_components.items():
                if isinstance(v, np.ndarray):
                    all_reward_comps.setdefault(k, []).append(float(np.mean(v)))
    if all_reward_comps:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        comp_names = list(all_reward_comps.keys())
        if comp_names:
            # Box plots
            ax = axes[0, 0]
            values = [all_reward_comps[k] for k in comp_names]
            bp = ax.boxplot(values, labels=comp_names, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax.set_title("Reward Components Distribution"); ax.set_ylabel("Mean Value")
            ax.tick_params(axis='x', rotation=45)
            # Correlation heatmap
            ax = axes[0, 1]
            if len(comp_names) > 1:
                mat = np.array([all_reward_comps[k] for k in comp_names]).T
                corr = np.corrcoef(mat.T)
                im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
                ax.set_xticks(range(len(comp_names))); ax.set_yticks(range(len(comp_names)))
                ax.set_xticklabels(comp_names, rotation=45, ha='right'); ax.set_yticklabels(comp_names)
                ax.set_title("Reward Components Correlation")
                plt.colorbar(im, ax=ax)
                # Annotate
                for i in range(len(comp_names)):
                    for j in range(len(comp_names)):
                        ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', color='black', fontsize=8)
            # Per-model reward component comparison
            ax = axes[1, 0]
            if comp_names:
                sorted_models = sorted(entries, key=lambda x: x["avg_return"], reverse=True)[:5]
                x = np.arange(len(sorted_models))
                width = 0.15
                for idx, comp in enumerate(comp_names[:4]):
                    vals = [sum(all_reward_comps.get(comp, [0])[i:i+len([ep for ep in models[idx_m].episodes])]) / max(len([ep for ep in models[idx_m].episodes]), 1) for idx_m, entry in enumerate(sorted_models) if idx_m < len(models)]
                    vals = vals[:len(sorted_models)]
                    if len(vals) == len(sorted_models):
                        ax.bar(x + idx * width, vals, width, label=comp[:10])
                ax.set_xticks(x + width * 1.5); ax.set_xticklabels([e["name"][:15] for e in sorted_models], rotation=45, ha='right')
                ax.set_title("Reward Components by Model (Top 5)"); ax.legend()
            # Reward efficiency scatter
            ax = axes[1, 1]
            total_rewards = [all_reward_comps.get('r_dx', [0])[i] + all_reward_comps.get('r_velocity', [0])[i] for i in range(min(len(entries), len(all_reward_comps.get('r_dx', []))))]
            if total_rewards and len(total_rewards) == len(efficiency):
                ax.scatter(total_rewards[:len(efficiency)], efficiency, alpha=0.6)
                ax.set_xlabel("Total Reward"); ax.set_ylabel("Efficiency")
                ax.set_title("Reward vs Efficiency"); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "reward_analysis.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Failure analysis (from archive)
    failure_modes = {'forward_fall': [], 'backward_fall': [], 'stagnation': [], 'short_episode': []}
    for m in models:
        for ep in m.episodes:
            if not ep.success and ep.length < 100:
                failure_modes['short_episode'].append(m.name)
            elif not ep.success:
                pitch_avg = float(np.mean(np.abs(ep.joints[:, 0]))) if ep.joints.size else 0.0
                if pitch_avg > 0.5:
                    failure_modes['forward_fall'].append(m.name)
                elif pitch_avg < -0.3:
                    failure_modes['backward_fall'].append(m.name)
                else:
                    failure_modes['stagnation'].append(m.name)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Failure distribution pie
    ax = axes[0]
    counts = [len(v) for v in failure_modes.values()]
    labels = list(failure_modes.keys())
    if sum(counts) > 0:
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
        ax.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title("Failure Mode Distribution")
    # Failure rate by section
    ax = axes[1]
    for s in sections:
        s_models = [i for i, entry in enumerate(entries) if entry['section'] == s]
        s_success = success[s_models] if len(s_models) else np.array([])
        failure_rates = [1.0 - float(np.mean(s_success))] if len(s_success) else [0.0]
        ax.bar([s], failure_rates)
    ax.set_ylim(0, 1); ax.set_title("Failure Rate by Section"); ax.set_ylabel("Failure Rate")
    fig.tight_layout(); fig.savefig(out_dir / "failure_analysis.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Architecture comparison (from archive) - backend/model type breakdown
    backend_types = {}
    for i, e in enumerate(entries):
        key = f"{e['backend']}_{e.get('model_type', 'unknown')}"
        if key not in backend_types:
            backend_types[key] = []
        backend_types[key].append(returns[i])
    if backend_types:
        fig, ax = plt.subplots(figsize=(10, 6))
        x_labels = list(backend_types.keys())
        means = [float(np.mean(vals)) for vals in backend_types.values()]
        stds = [float(np.std(vals)) for vals in backend_types.values()]
        bars = ax.bar(range(len(x_labels)), means, yerr=stds, capsize=5, alpha=0.8)
        ax.set_xticks(range(len(x_labels))); ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_title("Performance by Backend/Model Type"); ax.set_ylabel("Avg Return")
        ax.grid(True, axis='y', alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "architecture_comparison.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Returns by backend and section (grouped bars)
    try:
        sects = sorted(set(sections))
        bks = sorted(set(backends))
        data = np.zeros((len(bks), len(sects)), dtype=float)
        for i_b, b in enumerate(bks):
            for i_s, s in enumerate(sects):
                idxs = [i for i, e in enumerate(entries) if e['backend']==b and e['section']==s]
                data[i_b, i_s] = float(np.mean(returns[idxs])) if idxs else 0.0
        x = np.arange(len(sects)); width = 0.8/ max(1,len(bks))
        fig, ax = plt.subplots(figsize=(12, 6))
        for i_b, b in enumerate(bks):
            ax.bar(x + i_b*width, data[i_b], width, label=b)
        ax.set_xticks(x + (len(bks)-1)*width/2); ax.set_xticklabels(sects)
        ax.set_ylabel('Avg Return'); ax.set_title('Average Return by Backend and Section')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / 'returns_by_backend_section.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    except Exception:
        pass

    # Action spectra (FFT) for hip/lk/rk
    try:
        if all_actions:
            A = np.concatenate(all_actions, axis=0)
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for d in range(3):
                a = A[:, d] - np.mean(A[:, d])
                mag = np.abs(np.fft.rfft(a))
                freq = np.fft.rfftfreq(a.shape[0], d=1.0)
                axes[d].plot(freq[:200], mag[:200])
                axes[d].set_title(['Hip','LK','RK'][d] + ' Action Spectrum')
                axes[d].set_xlabel('Norm. Frequency'); axes[d].set_ylabel('Magnitude'); axes[d].grid(True, alpha=0.3)
            fig.tight_layout(); fig.savefig(out_dir / 'action_spectra.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    except Exception:
        pass

    # Action autocorrelation per joint
    try:
        if all_actions:
            A = np.concatenate(all_actions, axis=0)
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            max_lag = min(500, A.shape[0]-1)
            lags = np.arange(-max_lag, max_lag+1)
            for d in range(3):
                a = A[:, d] - np.mean(A[:, d])
                ac = np.correlate(a, a, mode='full')
                mid = ac.size//2
                axes[d].plot(lags, ac[mid-max_lag:mid+max_lag+1])
                axes[d].set_title(['Hip','LK','RK'][d] + ' Autocorrelation')
                axes[d].set_xlabel('Lag'); axes[d].grid(True, alpha=0.3)
            fig.tight_layout(); fig.savefig(out_dir / 'autocorr_actions.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    except Exception:
        pass

    # Duty factor (stance ratio) per foot
    try:
        duty_left = []
        duty_right = []
        for m in models:
            for ep in m.episodes:
                if ep.foot_z.size:
                    l_contact = (ep.foot_z[:, 0] < 0.02).astype(float)
                    r_contact = (ep.foot_z[:, 1] < 0.02).astype(float)
                    duty_left.append(float(np.mean(l_contact)))
                    duty_right.append(float(np.mean(r_contact)))
        if duty_left and duty_right:
            fig, ax = plt.subplots(figsize=(8, 5))
            means = [float(np.mean(duty_left)), float(np.mean(duty_right))]
            stds = [float(np.std(duty_left)), float(np.std(duty_right))]
            ax.bar(['Left','Right'], means, yerr=stds, capsize=5, alpha=0.8)
            ax.set_ylim(0, 1); ax.set_ylabel('Duty Factor'); ax.set_title('Foot Duty Factor (stance ratio)')
            ax.grid(axis='y', alpha=0.3)
            fig.tight_layout(); fig.savefig(out_dir / 'duty_factor.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    except Exception:
        pass

    # Episode distributions (lengths and returns)
    try:
        all_lengths = [ep.length for m in models for ep in m.episodes]
        all_returns = [ep.returns for m in models for ep in m.episodes]
        if all_lengths and all_returns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(all_lengths, bins=30, alpha=0.8)
            axes[0].set_title('Episode Length Distribution'); axes[0].set_xlabel('Steps'); axes[0].grid(True, alpha=0.3)
            axes[1].hist(all_returns, bins=30, alpha=0.8)
            axes[1].set_title('Episode Return Distribution'); axes[1].set_xlabel('Return'); axes[1].grid(True, alpha=0.3)
            fig.tight_layout(); fig.savefig(out_dir / 'episode_distributions.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    except Exception:
        pass

    # Write aggregate CSV for downstream analysis
    try:
        metrics_dir = out_dir.parent / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        import csv
        with open(metrics_dir / "aggregates.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name","section","backend","avg_return","median_return","success_rate","avg_length","avg_distance","avg_efficiency"])
            for m in models:
                agg = m.aggregate()
                if not agg:
                    continue
                w.writerow([
                    m.name, m.section, m.backend,
                    agg.get("avg_return", 0.0),
                    agg.get("median_return", 0.0),
                    agg.get("success_rate", 0.0),
                    agg.get("avg_length", 0.0),
                    agg.get("avg_distance", 0.0),
                    agg.get("avg_efficiency", 0.0),
                ])
    except Exception as e:
        print("Aggregate CSV write failed:", e)


def main():
    p = argparse.ArgumentParser("Comprehensive BC Evaluation")
    p.add_argument("--models_dir", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--episodes", type=int, default=5)
    args = p.parse_args()

    models_dir = Path(args.models_dir)
    out_root = Path(args.out)
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(list(models_dir.glob("*.pt")) + list(models_dir.glob("*.eqx")))
    evals: List[ModelEval] = []
    for ck in ckpts:
        print(f"Eval: {ck.name}")
        me = _eval_one_model(ck, n_episodes=args.episodes)
        if me and me.episodes:
            evals.append(me)

    if not evals:
        print("No evaluations succeeded.")
        return 0

    _multi_panel_summary(evals, fig_dir)
    print(f"Wrote figures to: {fig_dir}")

    # Write consolidated markdown with all results (enhanced from archive)
    try:
        reports_dir = out_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        md_path = reports_dir / "RESULTS.md"
        with open(md_path, "w") as f:
            f.write("# Behavior Cloning Overnight Evaluation Results\n\n")
            f.write(f"**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("This report consolidates all evaluation outputs including figures, tables, and metrics.\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            if entries:
                top_entry = sorted(entries, key=lambda x: x["avg_return"], reverse=True)[0]
                f.write(f"- **Total Models Evaluated:** {len(entries)}\n")
                f.write(f"- **Best Model:** {top_entry['name']} (avg return: {top_entry['avg_return']:.2f})\n")
                f.write(f"- **Overall Success Rate:** {float(np.mean(success)):.1%}\n")
                f.write(f"- **Best Section:** {max(set(sections), key=sections.count)}\n")
                f.write(f"- **Best Backend:** {max(set(backends), key=backends.count)}\n\n")

            # Figures
            figs = sorted((out_root / "figures").glob("*.png"))
            if figs:
                f.write("## Evaluation Figures\n\n")
                for p in figs:
                    f.write(f"### {p.stem.replace('_', ' ').title()}\n\n")
                    f.write(f"![{p.stem}](../figures/{p.name})\n\n")

            # Tables
            tables_dir = out_root / "tables"
            tbls = []
            if tables_dir.exists():
                tbls = sorted(list(tables_dir.glob("*.csv")) + list(tables_dir.glob("*.tsv")))
            if tbls:
                f.write("## Data Tables\n\n")
                for p in tbls:
                    f.write(f"- **{p.stem}:** `{p.name}`\n")
                f.write("\n")

            # Key Metrics Summary
            f.write("## Key Insights\n\n")
            f.write(f"- Backend with highest avg return: {max(set(backends), key=lambda b: float(np.mean(returns[np.array(backends)==b])))}\n")
            f.write(f"- Section with highest success rate: {max(set(sections), key=lambda s: float(np.mean(success[np.array(sections)==s])))}\n")
            f.write(f"- Most efficient model: {max(entries, key=lambda e: e.get('avg_efficiency', 0))['name']}\n\n")

            # Metrics (JSON)
            metrics_dir = out_root / "metrics"
            mets = []
            if metrics_dir.exists():
                mets = sorted(metrics_dir.glob("*.json"))
            if mets:
                f.write("## Metrics (JSON)\n\n")
                for p in mets:
                    f.write(f"- `{p.name}`\n")
                f.write("\n")

            f.write("---\n")
            f.write("*Generated by bc_comprehensive_eval.py with integrated archive analysis*\n")

        print(f"Wrote markdown report: {md_path}")
    except Exception as e:
        print("Failed to write markdown report:", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())


