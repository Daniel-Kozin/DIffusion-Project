"""
Trains a paired flow-matching model for the "rotate 180 degrees" task: given a phantom x0 at
some orientation, predict its exact 180-degree-rotated counterpart x1 (same underlying
phantom, orientation shifted by 4 of the 8 45-degree augmentation steps -- an exact, lossless
pair). Bidirectional: every orientation is used as x0, paired with its 180-degree opposite as
x1, so the model learns one general "rotate 180" operator rather than a fixed direction.

Unlike the unconditional model, x0 here is always real data (never noise), so no ODE
inversion is needed anywhere -- sampling is a single forward integration from the real x0.
This sidesteps the fragmentation problem documented in future_idea.md by construction.

    ./run_train_rotate180.sh --epochs 1000
"""
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import wandb

from .data import build_rotation_pair_dataloaders, build_rotation_pair_datasets
from .metrics import directional_signal_metrics, rotation_metrics
from .ode import sample
from .rotate_config import get_rotate_config
from .train import build_run_name, flow_matching_loss, get_device, parse_class_weights, save_checkpoint, set_seed
from .velocity_model import FlowMatchingUNet3D
from .viz_utils import log_rotation_result


def train_one_epoch_paired(model, loader, optimizer, device, class_weights=None) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x0_batch, x1_batch in loader:
        optimizer.zero_grad()
        loss = flow_matching_loss(model, x1_batch, device, x0=x0_batch, class_weights=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x1_batch.shape[0]
        n += x1_batch.shape[0]
    return total_loss / max(n, 1)


@torch.no_grad()
def validate_paired(model, loader, device, class_weights=None) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for x0_batch, x1_batch in loader:
        loss = flow_matching_loss(model, x1_batch, device, x0=x0_batch, class_weights=class_weights)
        total_loss += loss.item() * x1_batch.shape[0]
        n += x1_batch.shape[0]
    return total_loss / max(n, 1)


def pick_spread_indices(val_ds, n: int):
    """Spreads n indices across distinct val cases (index 0, len//n, 2*len//n, ...) rather
    than adjacent orientations of the very same phantom. Used both for the small qualitative
    preview set and the larger aggregate-metrics set."""
    if len(val_ds) == 0 or n <= 0:
        return []
    n = min(n, len(val_ds))
    step = max(1, len(val_ds) // n)
    return [min(i * step, len(val_ds) - 1) for i in range(n)]


@torch.no_grad()
def run_diagnostic_pass(model, val_ds, indices, device, n_steps: int, chunk_size: int) -> dict:
    """Aggregates rotation_metrics (decoded, discrete) and directional_signal_metrics (raw
    velocity, pre-decode) across all `indices`, processed in chunks of `chunk_size` to bound
    GPU memory -- a single 50-item batch through the 3D UNet's activations would be far more
    memory than the chunk_size the training batch itself was already tuned for. Returns
    dataset-wide means/fractions, suitable for a single wandb.log call that becomes a proper
    trend graph over epochs (unlike logging one scalar per individual pair, which is too
    noisy pair-to-pair to read a trend off)."""
    model.eval()
    voxel_agreements, centroid_errors = [], []
    pred_components, expected_components = [], []
    cos_sims, mag_ratios = [], []
    n_no_lump = 0

    for start in range(0, len(indices), chunk_size):
        chunk = indices[start:start + chunk_size]
        x0_batch = torch.stack([val_ds[i][0] for i in chunk]).to(device)
        x1_batch = torch.stack([val_ds[i][1] for i in chunk]).to(device)

        x1_hat = sample(model, n_steps, x0=x0_batch, device=device, t_start=0.0, t_end=1.0)
        pred_labels = x1_hat.argmax(dim=1).cpu()
        expected_labels = x1_batch.argmax(dim=1).cpu()

        for b in range(len(chunk)):
            m = rotation_metrics(pred_labels[b], expected_labels[b])
            voxel_agreements.append(m["voxel_agreement"])
            if m["centroid_error_voxels"] is not None:
                centroid_errors.append(m["centroid_error_voxels"])
            if not m["pred_has_lump"]:
                n_no_lump += 1
            pred_components.append(m["pred_lump_components"])
            expected_components.append(m["expected_lump_components"])

        dsig = directional_signal_metrics(model, x0_batch, x1_batch)
        cos_sims.append(dsig["cos_sim_mean"])
        mag_ratios.append(dsig["magnitude_ratio_mean"])

    return {
        "voxel_agreement_mean": float(np.mean(voxel_agreements)),
        "centroid_error_voxels_mean": float(np.mean(centroid_errors)) if centroid_errors else float("nan"),
        "frac_predicted_no_lump": n_no_lump / len(indices),
        "pred_lump_components_mean": float(np.mean(pred_components)),
        "expected_lump_components_mean": float(np.mean(expected_components)),
        "cos_sim_mean": float(np.nanmean(cos_sims)),
        "magnitude_ratio_mean": float(np.nanmean(mag_ratios)),
    }


def main():
    print("=" * 60)
    print("Starting rotate180 training")
    print("=" * 60)

    config = get_rotate_config()
    set_seed(config.seed)
    print(f"[1/5] Config: epochs={config.epochs}, batch_size={config.batch_size}, "
          f"base_ch={config.base_ch}, lr={config.lr}, checkpoint_epochs={config.checkpoint_epochs}")

    device = get_device(config.device)
    print(f"[2/5] Using device: {device}")

    class_weights = parse_class_weights(config.class_weights, device)
    print(f"      class_weights (bg/insert/pillar/lump): {config.class_weights}")

    print(f"[3/5] Loading dataset from '{config.data_dir}' "
          f"(this can take a little while the first time)...")
    t0 = time.time()
    train_ds, val_ds = build_rotation_pair_datasets(config.data_dir, val_frac=config.val_frac,
                                                      seed=config.seed, num_classes=config.num_classes)
    train_loader, val_loader = build_rotation_pair_dataloaders(train_ds, val_ds, batch_size=config.batch_size,
                                                                num_workers=config.num_workers)
    print(f"      done in {time.time() - t0:.1f}s — train pairs: {len(train_ds)}, val pairs: {len(val_ds)}")

    print("[4/5] Building model...")
    model = FlowMatchingUNet3D(num_classes=config.num_classes, base_ch=config.base_ch,
                                embed_channels=config.embed_channels, dropout=config.dropout,
                                use_grad_checkpoint=config.use_grad_checkpoint).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      model has {n_params:,} trainable parameters")

    run_name = build_run_name(config)
    print(f"[5/5] Connecting to Weights & Biases (project='{config.wandb_project}', "
          f"run='{run_name}')...")
    wandb.init(project=config.wandb_project, name=run_name,
               config=asdict(config), mode=config.wandb_mode)
    if wandb.run is not None and getattr(wandb.run, "url", None):
        print(f"      wandb run URL: {wandb.run.url}")

    if wandb.run is not None:
        config.checkpoint_dir = Path(wandb.run.dir) / "checkpoints"
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"      checkpoints -> {config.checkpoint_dir}")

    checkpoint_epoch_set = {int(e) for e in config.checkpoint_epochs.split(",") if e.strip()}
    preview_indices = pick_spread_indices(val_ds, config.n_preview_val_pairs)
    metric_indices = pick_spread_indices(val_ds, config.n_metric_val_pairs)
    preview_cases = [val_ds.index[idx][0] for idx in preview_indices]
    print(f"      checkpoint milestones: {sorted(checkpoint_epoch_set)}")
    print(f"      preview pairs (val cases, qualitative images): {preview_cases}")
    print(f"      metric pairs (aggregate scalars, graphed): {len(metric_indices)}")

    print("-" * 60)
    print(f"Training loop starting: {config.epochs} epochs")
    print("-" * 60)

    train_start = time.time()
    for epoch in range(config.epochs):
        epoch_num = epoch + 1
        epoch_start = time.time()
        train_loss = train_one_epoch_paired(model, train_loader, optimizer, device, class_weights=class_weights)
        val_loss = validate_paired(model, val_loader, device, class_weights=class_weights)
        epoch_seconds = time.time() - epoch_start

        elapsed = time.time() - train_start
        remaining_epochs = config.epochs - epoch_num
        avg_epoch_seconds = elapsed / epoch_num
        eta_seconds = remaining_epochs * avg_epoch_seconds
        eta = timedelta(seconds=int(eta_seconds))

        wandb.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "epoch": epoch,
            "epoch_num": epoch_num,
            "epochs_total": config.epochs,
            "progress": epoch_num / config.epochs,
            "epoch_seconds": epoch_seconds,
            "eta_seconds": eta_seconds,
        })

        print(f"epoch {epoch_num}/{config.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  ({epoch_seconds:.1f}s/epoch, ETA {eta})")

        do_preview = (epoch_num % config.sample_every_epochs == 0) or epoch_num in (1, config.epochs)
        if do_preview and preview_indices:
            print(f"  generating {len(preview_indices)} qualitative rotation previews for wandb...")
            for i, idx in enumerate(preview_indices):
                x0_i, x1_i = val_ds[idx]
                case_id, variant_idx, orientation = val_ds.index[idx]
                x1_hat = sample(model, config.n_train_sample_steps, x0=x0_i.unsqueeze(0).to(device),
                                 device=device, t_start=0.0, t_end=1.0)
                pred_label = x1_hat.argmax(dim=1)[0].cpu()
                orig_label = x0_i.argmax(dim=0)
                expected_label = x1_i.argmax(dim=0)
                log_rotation_result(f"preview/pair{i}_{case_id}", orig_label, pred_label, expected_label)

        if do_preview and metric_indices:
            print(f"  running aggregate metrics pass over {len(metric_indices)} val pairs...")
            agg = run_diagnostic_pass(model, val_ds, metric_indices, device,
                                       n_steps=config.n_train_sample_steps, chunk_size=config.batch_size)
            wandb.log({
                "epoch": epoch,
                "metrics50/voxel_agreement_mean": agg["voxel_agreement_mean"],
                "metrics50/centroid_error_voxels_mean": agg["centroid_error_voxels_mean"],
                "metrics50/frac_predicted_no_lump": agg["frac_predicted_no_lump"],
                "metrics50/pred_lump_components_mean": agg["pred_lump_components_mean"],
                "metrics50/expected_lump_components_mean": agg["expected_lump_components_mean"],
                "metrics50/cos_sim_mean": agg["cos_sim_mean"],
                "metrics50/magnitude_ratio_mean": agg["magnitude_ratio_mean"],
            })
            print(f"    voxel_agreement={agg['voxel_agreement_mean']:.4f}  "
                  f"centroid_error_voxels={agg['centroid_error_voxels_mean']:.2f}  "
                  f"cos_sim={agg['cos_sim_mean']:.4f}  magnitude_ratio={agg['magnitude_ratio_mean']:.4f}")

        if epoch_num in checkpoint_epoch_set:
            save_checkpoint(config.checkpoint_dir / f"epoch_{epoch_num:04d}.pt", model, optimizer, epoch_num, config)
        save_checkpoint(config.checkpoint_dir / "latest.pt", model, optimizer, epoch_num, config, quiet=True)

    print("-" * 60)
    print(f"Training complete: {config.epochs} epochs in "
          f"{timedelta(seconds=int(time.time() - train_start))}")
    print(f"Final checkpoint: {config.checkpoint_dir / 'latest.pt'}")
    print("-" * 60)


if __name__ == "__main__":
    main()
