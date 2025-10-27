"""
Behavior Cloning Training Pipeline

Unified training CLI supporting PyTorch and JAX backends.
Supports different control sections (hip, knees, both).
"""

from __future__ import annotations
import argparse
import sys
import os
import numpy as np
from passive_walker.bc.utils import (
    set_seed, set_global_seed, pick_device, ensure_dir, save_checkpoint, MetricsWriter, Normalizer,
    ckpt_name_for, meta_name_for, metrics_name_for, save_metrics_json
)
from passive_walker.bc.data.dataset import discover_npzs, split_by_episode, load_xy, create_data_loader, create_sequence_loader_from_files
from passive_walker.bc.data.augmentation import create_default_temporal_augmentation, create_light_temporal_augmentation, create_heavy_temporal_augmentation
from passive_walker.config.paths import BC_MODELS_DIR
from passive_walker.config.paths_redirect import redirect_legacy_dir


def compute_advanced_loss(pred, target, w1=1.0, w2=0.0, w3=0.1, w4=0.01):
    """
    Advanced loss function combining multiple terms for robust training.
    
    Args:
        pred: Model predictions
        target: Ground truth targets
        w1: L1 loss weight
        w2: MSE loss weight  
        w3: Smoothness loss weight (penalizes large action changes)
        w4: Bound penalty weight (penalizes actions outside [-1,1])
    
    Returns:
        Total loss and component breakdown
    """
    import torch

    # L1 loss (robust to outliers)
    l1_loss = torch.mean(torch.abs(pred - target))

    # MSE loss (smooth gradients)
    mse_loss = torch.mean((pred - target) ** 2)

    # Smoothness loss (penalize large action changes across time)
    if pred.size(0) > 1:
        action_diff = pred[1:] - pred[:-1]
        smoothness_loss = torch.mean(torch.abs(action_diff))
    else:
        smoothness_loss = pred.new_tensor(0.0)

    # Bound penalty (keep actions in valid range)
    bound_penalty = torch.mean(torch.relu(torch.abs(pred) - 1.0))

    total = w1 * l1_loss + w2 * mse_loss + w3 * smoothness_loss + w4 * bound_penalty
    return total, {
        "l1": float(l1_loss),
        "mse": float(mse_loss),
        "smoothness": float(smoothness_loss),
        "bound_penalty": float(bound_penalty),
    }


def train_torch(args):
    """Train PyTorch BC model with early stopping and checkpointing."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from passive_walker.bc.models.models_torch import TorchMLP, TorchMLPLarge

    # Setup device and data
    device = pick_device(args.gpu)
    print(f"[INFO] Using device: {device}")

    # Discover and split data
    files = discover_npzs(args.data)
    train_files, val_files = split_by_episode(files, val_ratio=0.1)
    print(f"[INFO] Found {len(files)} episodes: {len(train_files)} train, {len(val_files)} val")

    # Load training data
    print("[INFO] Loading training data...")
    X_train, y_train = load_xy(train_files, args.section, args.label_type, args.frame_stack)
    print("[INFO] Loading validation data...")
    X_val, y_val = load_xy(val_files, args.section, args.label_type, args.frame_stack)

    print(f"[INFO] Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")
    print(f"[INFO] Input dim: {X_train.shape[1]}, Output dim: {y_train.shape[1]}")

    # Normalize inputs
    std_values = np.std(X_train, axis=0)
    # Avoid division by zero for features with no variance
    std_values = np.maximum(std_values, 1e-8)
    normalizer = Normalizer(
        mean=np.mean(X_train, axis=0),
        std=std_values
    )
    X_train_norm = normalizer.encode(X_train)
    X_val_norm = normalizer.encode(X_val)

    # Create model (use actual input dimension from data)
    input_dim = X_train.shape[1]
    model = TorchMLPLarge(in_dim=input_dim, out_dim=y_train.shape[1], hidden=512, dropout=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Loss function
    if args.section == "both-adv":
        def criterion(pred, target):
            return compute_advanced_loss(pred, target, args.w1, args.w2, args.w3, args.w4)
    else:
        criterion = nn.L1Loss()

    # Training setup
    metrics_writer = MetricsWriter()
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5

    # Create save directory
    ensure_dir(args.save_dir)

    print(f"[INFO] Starting training for {args.epochs} epochs...")

    # Training loop
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for X_batch, y_batch in create_data_loader(X_train_norm, y_train, args.batch, True, device):
            optimizer.zero_grad()
            preds = model(X_batch)

            if args.section == "both-adv":
                loss, _ = criterion(preds, y_batch)
            else:
                loss = criterion(preds, y_batch)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
    
        avg_train_loss = train_loss / max(1, train_batches)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
    
        with torch.no_grad():
            for X_batch, y_batch in create_data_loader(X_val_norm, y_val, args.batch, False, device):
                preds = model(X_batch)
                if args.section == "both-adv":
                    loss, _ = criterion(preds, y_batch)
                else:
                    loss = criterion(preds, y_batch)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches) if val_batches > 0 else float('nan')

        # Log metrics
        metrics_writer.log_epoch(epoch, avg_train_loss, avg_val_loss)
        print(f"Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")

        # Early stopping and checkpointing
        score = avg_val_loss if not np.isnan(avg_val_loss) else avg_train_loss
        if score + 1e-8 < best_val_loss:
            best_val_loss = score
            patience_counter = 0

            # Save best model
            meta = {
                "input_dim": input_dim,  # Use actual input dim from data
                "output_dim": y_train.shape[1],
                "section": args.section,
                "label_type": args.label_type,
                "dataset_path": args.data,
                "seed": args.seed,
                "epoch": epoch,
                "steps": X_train.shape[0],
                "best_val_loss": best_val_loss,
                "frame_stack": args.frame_stack,
            }
            checkpoint_path, meta_path = save_checkpoint(
                model, normalizer, meta, args.save_dir,
                args.section, args.seed, epoch, X_train.shape[0]
            )
            print(f"[INFO] Saved checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"[INFO] Early stopping at epoch {epoch} (patience={patience})")
            break

    # Save final metrics
    metrics_path = os.path.join(args.save_dir, f"torch_{args.section}_seed{args.seed}_metrics.json")
    metrics_writer.save(metrics_path)
    print(f"[INFO] Training completed. Best val loss: {best_val_loss:.6f}")
    print(f"[INFO] Metrics saved to: {metrics_path}")


# JAX Training Functions
def _choose_device_jax(prefer_gpu: bool) -> None:
    """JAX device selection (automatic)."""
    try:
        import jax
        gpus = jax.devices("gpu")
        if prefer_gpu and not gpus:
            print("[JAX] GPU requested but not found; using CPU.")
    except Exception:
        pass


def _dataloader_numpy(x: np.ndarray, y: np.ndarray, batch: int, shuffle: bool, rng: np.random.RandomState):
    """Simple numpy data loader for JAX training."""
    n = x.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, batch):
        j = min(i + batch, n)
        sel = idx[i:j]
        yield x[sel], y[sel]


def _l1(pred, targ):
    """L1 loss for JAX."""
    import jax.numpy as jnp
    return jnp.mean(jnp.abs(pred - targ))


def _mse(pred, targ):
    """MSE loss for JAX."""
    import jax.numpy as jnp
    return jnp.mean((pred - targ) ** 2)


def _smoothness_term(pred):
    """Smoothness loss term for JAX."""
    import jax.numpy as jnp
    if pred.shape[0] <= 1:
        return jnp.array(0.0, dtype=pred.dtype)
    diffs = pred[1:] - pred[:-1]
    return jnp.mean(jnp.abs(diffs))


def _bound_penalty(pred):
    """Bound penalty term for JAX."""
    import jax.numpy as jnp
    over = jnp.maximum(jnp.abs(pred) - 1.0, 0.0)
    return jnp.mean(over)


def train_jax(args):
    """Train JAX model with Equinox and Optax."""
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax
    from passive_walker.bc.models.models_jax import make_model, save_eqx
    
    # Section to output dimension mapping
    SECTION_TO_OUTDIM = {"hip": 1, "knees": 2, "both": 3, "both-adv": 3}
    
    set_global_seed(args.seed)
    _choose_device_jax(args.gpu)
    
    out_dim = SECTION_TO_OUTDIM[args.section]
    
    # Discover and split data
    files = discover_npzs(args.data)
    train_files, val_files = split_by_episode(files, val_ratio=0.1)
    print(f"[INFO] Found {len(files)} episodes: {len(train_files)} train, {len(val_files)} val")
    
    # Load data
    print("[INFO] Loading training data...")
    X_train, y_train = load_xy(train_files, args.section, args.label_type, args.frame_stack)
    print("[INFO] Loading validation data...")
    X_val, y_val = load_xy(val_files, args.section, args.label_type, args.frame_stack)
    
    print(f"[INFO] Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")
    print(f"[INFO] Input dim: {X_train.shape[1]}, Output dim: {y_train.shape[1]}")
    
    # Normalize inputs
    normalizer = Normalizer().fit(X_train)
    X_train_norm = normalizer.apply(X_train)
    X_val_norm = normalizer.apply(X_val) if X_val.size > 0 else X_val
    
    # Convert to JAX arrays
    X_train_jax = jnp.asarray(X_train_norm)
    y_train_jax = jnp.asarray(y_train)
    X_val_jax = jnp.asarray(X_val_norm) if X_val.size > 0 else X_val
    y_val_jax = jnp.asarray(y_val) if X_val.size > 0 else y_val
    
    # Create model
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)
    input_dim = 11 * args.frame_stack
    model = make_model(in_dim=input_dim, out_dim=out_dim, width=128, depth=2, key=subkey)
    
    # Setup optimizer
    filter_spec = eqx.is_array
    optimizer = optax.adamw(learning_rate=args.lr, weight_decay=1e-5)
    opt_state = optimizer.init(eqx.filter(model, filter_spec))
    
    # Loss function
    def loss_fn(model, xb, yb):
        pred = model(xb)
        if args.section != "both-adv":
            return _l1(pred, yb)
        # Advanced combo: L = w1*L1 + w2*MSE + w3*Smoothness + w4*BoundPenalty
        w1, w2, w3, w4 = args.w1, args.w2, args.w3, args.w4
        return (
            w1 * _l1(pred, yb) +
            w2 * _mse(pred, yb) +
            w3 * _smoothness_term(pred) +
            w4 * _bound_penalty(pred)
        )
    
    # JITed step
    @eqx.filter_jit
    def step(model, opt_state, xb, yb):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, xb, yb)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, filter_spec))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    
    @eqx.filter_jit
    def eval_loss(model, xb, yb):
        return loss_fn(model, xb, yb)
    
    # Training loop
    rng = np.random.RandomState(args.seed)
    best_val_loss = float('inf')
    best_ckpt = None
    patience_counter = 0
    patience = 5
    hist_train, hist_val = [], []
    
    # Create save directory
    ensure_dir(args.save_dir)
    
    print(f"[INFO] Starting JAX training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Training
        model_loss = 0.0
        batches = 0
        shuffle = args.section != "both-adv"  # Keep order for smoothness term
        
        for xb_np, yb_np in _dataloader_numpy(
            np.asarray(X_train_jax), np.asarray(y_train_jax), 
            batch=args.batch, shuffle=shuffle, rng=rng
        ):
            xb = jnp.asarray(xb_np)
            yb = jnp.asarray(yb_np)
            model, opt_state, train_loss = step(model, opt_state, xb, yb)
            model_loss += float(train_loss)
            batches += 1
        
        avg_train_loss = model_loss / max(1, batches)
        hist_train.append(avg_train_loss)
        
        # Validation
        if X_val.size > 0:
            val_loss = float(eval_loss(model, X_val_jax, y_val_jax))
            hist_val.append(val_loss)
        else:
            val_loss = float('nan')
            hist_val.append(val_loss)
        
        print(f"Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Early stopping and checkpointing
        score = val_loss if not np.isnan(val_loss) else avg_train_loss
        if score + 1e-8 < best_val_loss:
            best_val_loss = score
            patience_counter = 0
            
            # Save best model
            meta = {
                "backend": "jax",
                "input_dim": X_train.shape[1],
                "output_dim": out_dim,
                "section": args.section,
                "label_type": args.label_type,
                "dataset_path": args.data,
                "seed": args.seed,
                "epoch": epoch,
                "steps": X_train.shape[0],
                "best_val_loss": best_val_loss,
                "normalizer_mean": normalizer.mean.tolist(),
                "normalizer_std": normalizer.std.tolist()
            }
            
            # Save temporary checkpoint
            tmp_ckpt = os.path.join(args.save_dir, f"tmp_best_{args.section}.eqx")
            save_eqx(tmp_ckpt, model)
            best_ckpt = tmp_ckpt
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"[INFO] Early stopping at epoch {epoch} (patience={patience})")
            break
    
    # Load best model if early stopping occurred
    if best_ckpt and os.path.exists(best_ckpt):
        model = eqx.tree_deserialise_leaves(best_ckpt, model)
        os.remove(best_ckpt)  # Clean up temp file
    
    # Save final artifacts
    steps = X_train.shape[0]
    episodes = len(train_files)
    stem = ckpt_name_for(backend="jax", section=args.section, seed=args.seed, episodes=episodes, steps=steps)
    ckpt_path = os.path.join(args.save_dir, stem + ".eqx")
    meta_path = os.path.join(args.save_dir, meta_name_for(backend="jax", section=args.section, seed=args.seed, episodes=episodes, steps=steps))
    metrics_path = os.path.join(args.save_dir, metrics_name_for(backend="jax", section=args.section, seed=args.seed))
    
    # Save model
    save_eqx(ckpt_path, model)
    
    # Save metadata
    meta_dict = {
        "backend": "jax",
        "section": args.section,
        "seed": args.seed,
        "episodes": episodes,
        "steps": steps,
        "in_dim": int(X_train.shape[1]),
        "out_dim": out_dim,
        "hidden": 128,
        "depth": 2,
        "lr": args.lr,
        "weight_decay": 1e-5,
        "batch": args.batch,
        "label_type": args.label_type,
        "adv_w": [args.w1, args.w2, args.w3, args.w4] if args.section == "both-adv" else None,
        "normalizer": {"mean": normalizer.mean.tolist(), "std": normalizer.std.tolist()},
        "files_train": train_files,
        "files_val": val_files,
    }
    with open(meta_path, "w") as f:
        import json
        json.dump(meta_dict, f, indent=2)
    
    # Save metrics
    save_metrics_json(metrics_path, hist_train, hist_val)
    
    print(f"[INFO] JAX training completed. Best val loss: {best_val_loss:.6f}")
    print(f"[INFO] Model saved to: {ckpt_path}")
    print(f"[INFO] Metrics saved to: {metrics_path}")


def compute_temporal_loss(pred, target, mask, loss_type="l1", smoothness_weight=0.1):
    """
    Compute loss for temporal models with masking and smoothness penalty.
    
    Args:
        pred: Model predictions (batch, seq_len, action_dim)
        target: Ground truth targets (batch, seq_len, action_dim)
        mask: Boolean mask for valid timesteps (batch, seq_len)
        loss_type: Type of base loss ("l1", "mse", "smooth_l1")
        smoothness_weight: Weight for temporal smoothness penalty
    
    Returns:
        Total loss and component breakdown
    """
    import torch
    
    # Apply mask to predictions and targets
    masked_pred = pred * mask.unsqueeze(-1).float()
    masked_target = target * mask.unsqueeze(-1).float()
    
    # Base loss computation
    if loss_type == "l1":
        base_loss = torch.mean(torch.abs(masked_pred - masked_target))
    elif loss_type == "mse":
        base_loss = torch.mean((masked_pred - masked_target) ** 2)
    elif loss_type == "smooth_l1":
        base_loss = torch.nn.functional.smooth_l1_loss(masked_pred, masked_target)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Temporal smoothness penalty
    if smoothness_weight > 0 and pred.size(1) > 1:
        # Compute action differences across time
        action_diff = pred[:, 1:] - pred[:, :-1]
        # Apply mask to differences (only penalize valid transitions)
        valid_transitions = mask[:, 1:] & mask[:, :-1]
        masked_diff = action_diff * valid_transitions.unsqueeze(-1).float()
        smoothness_loss = torch.mean(torch.abs(masked_diff))
    else:
        smoothness_loss = pred.new_tensor(0.0)
    
    total_loss = base_loss + smoothness_weight * smoothness_loss
    
    return total_loss, {
        "base_loss": float(base_loss.detach()),
        "smoothness_loss": float(smoothness_loss.detach()),
        "total_loss": float(total_loss.detach())
    }


def train_temporal_torch(config):
    """Train PyTorch temporal BC model with sequence data."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from passive_walker.bc.models.temporal_torch import create_temporal_model
    
    # Setup device and data
    device = pick_device(config.gpu if hasattr(config, 'gpu') else False)
    print(f"[INFO] Using device: {device}")
    
    # Discover and split data
    files = discover_npzs(config.data_dir)
    train_files, val_files = split_by_episode(files, val_ratio=config.validation_split)
    print(f"[INFO] Found {len(files)} episodes: {len(train_files)} train, {len(val_files)} val")
    
    # Create sequence data loaders
    print("[INFO] Creating sequence data loaders...")
    
    # Determine augmentation
    augmentation = None
    if config.temporal_augmentation:
        if config.augmentation_type == "light":
            augmentation = create_light_temporal_augmentation()
        elif config.augmentation_type == "heavy":
            augmentation = create_heavy_temporal_augmentation()
        else:
            augmentation = create_default_temporal_augmentation()
        print(f"[INFO] Using {config.augmentation_type} temporal augmentation")
    
    # Create training loader
    train_loader = create_sequence_loader_from_files(
        files=train_files,
        section=config.section,
        batch_size=config.batch_size,
        label_type="act",
        shuffle=True,
        max_length=config.sequence_length,
        window_size=config.window_size,
        stride=config.stride,
        padding_strategy=config.padding_strategy,
        num_workers=0
    )
    
    # Create validation loader
    val_loader = create_sequence_loader_from_files(
        files=val_files,
        section=config.section,
        batch_size=config.batch_size,
        label_type="act",
        shuffle=False,
        max_length=config.sequence_length,
        window_size=config.window_size,
        stride=config.stride,
        padding_strategy=config.padding_strategy,
        num_workers=0
    )
    
    # Get input/output dimensions from first batch
    for obs_batch, action_batch, mask_batch in train_loader:
        input_dim = obs_batch.shape[-1]
        output_dim = action_batch.shape[-1]
        break
    
    print(f"[INFO] Input dim: {input_dim}, Output dim: {output_dim}")
    
    # Create temporal model
    model = create_temporal_model(
        config.model_type,
        input_dim,
        output_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        bidirectional=config.bidirectional,
        dropout=config.dropout
    ).to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    
    # Setup scheduler
    scheduler = None
    if config.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    elif config.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Training setup
    metrics_writer = MetricsWriter()
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Create save directory
    ensure_dir(config.checkpoint_dir)
    
    print(f"[INFO] Starting temporal training for {config.epochs} epochs...")
    print(f"[INFO] Model: {config.model_type}, Hidden: {config.hidden_size}, Layers: {config.num_layers}")
    
    # Training loop
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        train_metrics = {"base_loss": 0.0, "smoothness_loss": 0.0}
        
        for obs_batch, action_batch, mask_batch in train_loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            # Apply temporal augmentation if enabled
            if augmentation is not None:
                obs_batch, action_batch, mask_batch = augmentation(obs_batch, action_batch, mask_batch)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_batch, _ = model(obs_batch)  # Only take the output, ignore hidden state
            
            # Compute loss with masking
            loss, loss_components = compute_temporal_loss(
                pred_batch, action_batch, mask_batch,
                loss_type=config.loss_type,
                smoothness_weight=config.temporal_smoothness_weight
            )
            
            # Backward pass with gradient clipping
            loss.backward()
            if config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            train_metrics["base_loss"] += loss_components["base_loss"]
            train_metrics["smoothness_loss"] += loss_components["smoothness_loss"]
        
        avg_train_loss = train_loss / max(1, train_batches)
        avg_train_metrics = {k: v / max(1, train_batches) for k, v in train_metrics.items()}
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_metrics = {"base_loss": 0.0, "smoothness_loss": 0.0}
        
        with torch.no_grad():
            for obs_batch, action_batch, mask_batch in val_loader:
                obs_batch = obs_batch.to(device)
                action_batch = action_batch.to(device)
                mask_batch = mask_batch.to(device)
                
                pred_batch, _ = model(obs_batch)  # Only take the output, ignore hidden state
                loss, loss_components = compute_temporal_loss(
                    pred_batch, action_batch, mask_batch,
                    loss_type=config.loss_type,
                    smoothness_weight=config.temporal_smoothness_weight
                )
                
                val_loss += loss.item()
                val_batches += 1
                val_metrics["base_loss"] += loss_components["base_loss"]
                val_metrics["smoothness_loss"] += loss_components["smoothness_loss"]
        
        avg_val_loss = val_loss / max(1, val_batches) if val_batches > 0 else float('nan')
        avg_val_metrics = {k: v / max(1, val_batches) for k, v in val_metrics.items()}
        
        # Update scheduler
        if scheduler is not None:
            if config.scheduler == "plateau":
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # Log metrics
        metrics_writer.log_epoch(epoch, avg_train_loss, avg_val_loss)
        print(f"Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")
        print(f"         train_base={avg_train_metrics['base_loss']:.6f}, train_smooth={avg_train_metrics['smoothness_loss']:.6f}")
        
        # Early stopping and checkpointing
        score = avg_val_loss if not np.isnan(avg_val_loss) else avg_train_loss
        if score + 1e-8 < best_val_loss:
            best_val_loss = score
            patience_counter = 0
            
            # Save best model
            meta = {
                "backend": "torch",
                "model_type": config.model_type,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "bidirectional": config.bidirectional,
                "dropout": config.dropout,
                "section": config.section,
                "label_type": "act",
                "dataset_path": config.data_dir,
                "seed": config.seed,
                "epoch": epoch,
                "steps": len(train_files),
                "best_val_loss": best_val_loss,
                "sequence_length": config.sequence_length,
                "window_size": config.window_size,
                "stride": config.stride,
                "padding_strategy": config.padding_strategy,
                "temporal_augmentation": config.temporal_augmentation,
                "augmentation_type": config.augmentation_type,
                "loss_type": config.loss_type,
                "temporal_smoothness_weight": config.temporal_smoothness_weight,
                "gradient_clip_norm": config.gradient_clip_norm,
            }
            
            # Create a dummy normalizer for checkpoint saving
            from passive_walker.bc.utils import Normalizer
            dummy_normalizer = Normalizer(mean=np.zeros(input_dim), std=np.ones(input_dim))
            
            checkpoint_path, meta_path = save_checkpoint(
                model, dummy_normalizer, meta, config.checkpoint_dir,
                config.section, config.seed, epoch, len(train_files)
            )
            print(f"[INFO] Saved checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
        
        if patience_counter >= config.early_stopping_patience:
            print(f"[INFO] Early stopping at epoch {epoch} (patience={config.early_stopping_patience})")
            break
    
    # Save final metrics
    metrics_path = os.path.join(config.checkpoint_dir, f"torch_temporal_{config.section}_seed{config.seed}_metrics.json")
    metrics_writer.save(metrics_path)
    print(f"[INFO] Temporal training completed. Best val loss: {best_val_loss:.6f}")
    print(f"[INFO] Metrics saved to: {metrics_path}")


def train_temporal_jax(config):
    """Train JAX temporal BC model with sequence data."""
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax
    from passive_walker.bc.models.temporal_jax import make_temporal_model
    
    set_global_seed(config.seed)
    
    # Discover and split data
    files = discover_npzs(config.data_dir)
    train_files, val_files = split_by_episode(files, val_ratio=config.validation_split)
    print(f"[INFO] Found {len(files)} episodes: {len(train_files)} train, {len(val_files)} val")
    
    # Create sequence data loaders
    print("[INFO] Creating sequence data loaders...")
    
    # Create training loader
    train_loader = create_sequence_loader_from_files(
        files=train_files,
        section=config.section,
        batch_size=config.batch_size,
        label_type="act",
        shuffle=True,
        max_length=config.sequence_length,
        window_size=config.window_size,
        stride=config.stride,
        padding_strategy=config.padding_strategy,
        num_workers=0
    )
    
    # Create validation loader
    val_loader = create_sequence_loader_from_files(
        files=val_files,
        section=config.section,
        batch_size=config.batch_size,
        label_type="act",
        shuffle=False,
        max_length=config.sequence_length,
        window_size=config.window_size,
        stride=config.stride,
        padding_strategy=config.padding_strategy,
        num_workers=0
    )
    
    # Get input/output dimensions from first batch
    for obs_batch, action_batch, mask_batch in train_loader:
        input_dim = obs_batch.shape[-1]
        output_dim = action_batch.shape[-1]
        break
    
    print(f"[INFO] Input dim: {input_dim}, Output dim: {output_dim}")
    
    # Create temporal model
    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    model = make_temporal_model(
        config.model_type,
        input_dim,
        output_dim,
        hidden_size=config.hidden_size,
        dropout_rate=config.dropout,
        key=subkey
    )
    
    # Setup optimizer
    filter_spec = eqx.is_array
    optimizer = optax.adamw(learning_rate=config.learning_rate, weight_decay=1e-5)
    opt_state = optimizer.init(eqx.filter(model, filter_spec))
    
    # Loss function with masking
    def temporal_loss_fn(model, obs, actions, mask):
        pred, _ = model(obs)  # Model returns (outputs, hidden_state)
        
        # Apply mask
        masked_pred = pred * mask[..., None]
        masked_target = actions * mask[..., None]
        
        # Base loss
        if config.loss_type == "l1":
            base_loss = jnp.mean(jnp.abs(masked_pred - masked_target))
        elif config.loss_type == "mse":
            base_loss = jnp.mean((masked_pred - masked_target) ** 2)
        else:
            base_loss = jnp.mean(jnp.abs(masked_pred - masked_target))
        
        # Temporal smoothness penalty
        if config.temporal_smoothness_weight > 0 and pred.shape[1] > 1:
            action_diff = pred[:, 1:] - pred[:, :-1]
            valid_transitions = mask[:, 1:] & mask[:, :-1]
            masked_diff = action_diff * valid_transitions[..., None]
            smoothness_loss = jnp.mean(jnp.abs(masked_diff))
        else:
            smoothness_loss = 0.0
        
        total_loss = base_loss + config.temporal_smoothness_weight * smoothness_loss
        return total_loss, {"base_loss": base_loss, "smoothness_loss": smoothness_loss}
    
    # JITed step
    @eqx.filter_jit
    def step(model, opt_state, obs, actions, mask):
        (loss, metrics), grads = eqx.filter_value_and_grad(temporal_loss_fn, has_aux=True)(model, obs, actions, mask)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, filter_spec))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, metrics
    
    @eqx.filter_jit
    def eval_loss(model, obs, actions, mask):
        return temporal_loss_fn(model, obs, actions, mask)
    
    # Training loop
    best_val_loss = float('inf')
    best_ckpt = None
    patience_counter = 0
    hist_train, hist_val = [], []
    
    # Create save directory
    ensure_dir(config.checkpoint_dir)
    
    print(f"[INFO] Starting JAX temporal training for {config.epochs} epochs...")
    print(f"[INFO] Model: {config.model_type}, Hidden: {config.hidden_size}, Layers: {config.num_layers}")
    
    for epoch in range(config.epochs):
        # Training
        model_loss = 0.0
        batches = 0
        train_metrics = {"base_loss": 0.0, "smoothness_loss": 0.0}
        
        for obs_batch, action_batch, mask_batch in train_loader:
            obs_jax = jnp.asarray(obs_batch)
            action_jax = jnp.asarray(action_batch)
            mask_jax = jnp.asarray(mask_batch)
            
            model, opt_state, train_loss, metrics = step(model, opt_state, obs_jax, action_jax, mask_jax)
            model_loss += float(train_loss)
            batches += 1
            train_metrics["base_loss"] += float(metrics["base_loss"])
            train_metrics["smoothness_loss"] += float(metrics["smoothness_loss"])
        
        avg_train_loss = model_loss / max(1, batches)
        avg_train_metrics = {k: v / max(1, batches) for k, v in train_metrics.items()}
        hist_train.append(avg_train_loss)
        
        # Validation
        val_loss = 0.0
        val_batches = 0
        val_metrics = {"base_loss": 0.0, "smoothness_loss": 0.0}
        
        for obs_batch, action_batch, mask_batch in val_loader:
            obs_jax = jnp.asarray(obs_batch)
            action_jax = jnp.asarray(action_batch)
            mask_jax = jnp.asarray(mask_batch)
            
            loss, metrics = eval_loss(model, obs_jax, action_jax, mask_jax)
            val_loss += float(loss)
            val_batches += 1
            val_metrics["base_loss"] += float(metrics["base_loss"])
            val_metrics["smoothness_loss"] += float(metrics["smoothness_loss"])
        
        avg_val_loss = val_loss / max(1, val_batches) if val_batches > 0 else float('nan')
        avg_val_metrics = {k: v / max(1, val_batches) for k, v in val_metrics.items()}
        hist_val.append(avg_val_loss)
        
        print(f"Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")
        print(f"         train_base={avg_train_metrics['base_loss']:.6f}, train_smooth={avg_train_metrics['smoothness_loss']:.6f}")
        
        # Early stopping and checkpointing
        score = avg_val_loss if not np.isnan(avg_val_loss) else avg_train_loss
        if score + 1e-8 < best_val_loss:
            best_val_loss = score
            patience_counter = 0
            
            # Save temporary checkpoint
            tmp_ckpt = os.path.join(config.checkpoint_dir, f"tmp_best_temporal_{config.section}.eqx")
            eqx.tree_serialise_leaves(tmp_ckpt, model)
            best_ckpt = tmp_ckpt
        else:
            patience_counter += 1
        
        if patience_counter >= config.early_stopping_patience:
            print(f"[INFO] Early stopping at epoch {epoch} (patience={config.early_stopping_patience})")
            break
    
    # Load best model if early stopping occurred
    if best_ckpt and os.path.exists(best_ckpt):
        model = eqx.tree_deserialise_leaves(best_ckpt, model)
        os.remove(best_ckpt)  # Clean up temp file
    
    # Save final artifacts
    steps = len(train_files)
    episodes = len(train_files)
    stem = ckpt_name_for(backend="jax", section=config.section, seed=config.seed, episodes=episodes, steps=steps)
    ckpt_path = os.path.join(config.checkpoint_dir, f"temporal_{stem}.eqx")
    meta_path = os.path.join(config.checkpoint_dir, f"temporal_{meta_name_for(backend='jax', section=config.section, seed=config.seed, episodes=episodes, steps=steps)}")
    metrics_path = os.path.join(config.checkpoint_dir, f"temporal_{metrics_name_for(backend='jax', section=config.section, seed=config.seed)}")
    
    # Save model
    eqx.tree_serialise_leaves(ckpt_path, model)
    
    # Save metadata
    meta_dict = {
        "backend": "jax",
        "model_type": config.model_type,
        "section": config.section,
        "seed": config.seed,
        "episodes": episodes,
        "steps": steps,
        "in_dim": int(input_dim),
        "out_dim": int(output_dim),
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "lr": config.learning_rate,
        "weight_decay": 1e-5,
        "batch": config.batch_size,
        "label_type": "act",
        "sequence_length": config.sequence_length,
        "window_size": config.window_size,
        "stride": config.stride,
        "padding_strategy": config.padding_strategy,
        "temporal_augmentation": config.temporal_augmentation,
        "augmentation_type": config.augmentation_type,
        "loss_type": config.loss_type,
        "temporal_smoothness_weight": config.temporal_smoothness_weight,
        "files_train": train_files,
        "files_val": val_files,
    }
    with open(meta_path, "w") as f:
        import json
        json.dump(meta_dict, f, indent=2)
    
    # Save metrics
    save_metrics_json(metrics_path, hist_train, hist_val)
    
    print(f"[INFO] JAX temporal training completed. Best val loss: {best_val_loss:.6f}")
    print(f"[INFO] Model saved to: {ckpt_path}")
    print(f"[INFO] Metrics saved to: {metrics_path}")


def main():
    """Main CLI entry point for BC training."""
    p = argparse.ArgumentParser("BC train")
    p.add_argument("--backend", choices=["torch", "jax"], required=True)
    p.add_argument("--section", choices=["hip", "knees", "both", "both-adv"], required=True)
    p.add_argument("--data", required=True, help="folder with episode_*.npz")
    p.add_argument("--label-type", choices=["act", "qdes"], default="act")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--save-dir", default=str(BC_MODELS_DIR))
    # Advanced loss weights for 'both-adv'
    p.add_argument("--w1", type=float, default=1.0)
    p.add_argument("--w2", type=float, default=0.0)
    p.add_argument("--w3", type=float, default=0.1)
    p.add_argument("--w4", type=float, default=0.01)
    p.add_argument("--frame-stack", type=int, default=1, help="Number of frames to stack for temporal context")
    args = p.parse_args()

    # Redirect legacy save dir if needed
    args.save_dir = str(redirect_legacy_dir(args.save_dir))

    set_seed(args.seed)

    print(f"[OK] Backend={args.backend}  Section={args.section}  Data={args.data}")
    print(f"[OK] label={args.label_type}  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
    if args.section == "both-adv":
        print(f"[OK] adv loss weights: w1={args.w1} w2={args.w2} w3={args.w3} w4={args.w4}")

    if args.backend == "torch":
        try:
            train_torch(args)
        except ImportError as e:
            sys.exit(f"PyTorch not available: {e}")
    else:
        try:
            train_jax(args)
        except ImportError as e:
            sys.exit(f"JAX/Equinox not available: {e}")


if __name__ == "__main__":
    main()