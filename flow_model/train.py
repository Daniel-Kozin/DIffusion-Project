import random
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from .config import Config, get_config
from .data import build_datasets, build_dataloaders
from .ode import sample
from .velocity_model import FlowMatchingUNet3D
from .viz_utils import log_grid_to_wandb


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_class_weights(class_weights: str, device: torch.device) -> torch.Tensor:
    """
    'w0,w1,w2,w3' -> tensor([w0,w1,w2,w3]) broadcastable over the channel dim of
    [B, C, K, H, W] velocity tensors. Channel order matches the dataset's one-hot
    encoding: background, insert, pillar, lump.
    """
    weights = torch.tensor([float(w) for w in class_weights.split(",")], device=device)
    return weights.view(1, -1, 1, 1, 1)


def soft_dice_loss(probs: torch.Tensor, target_classes: torch.Tensor, classes) -> torch.Tensor:
    """
    1 - mean soft Dice coefficient over `classes`, averaged. probs: [B, C, ...] softmax
    probabilities. target_classes: [B, ...] int64 class indices. Unlike cross-entropy
    weighted by each voxel's TRUE class, Dice is symmetric in false positives/negatives:
    predicting excess volume for a class grows the denominator (probs[:, c].sum()) without
    growing the intersection, so over-prediction is penalized even where CE wouldn't (see
    flow_matching_loss's dice_weight docstring).
    """
    dice_losses = []
    for c in classes:
        p_c = probs[:, c]
        t_c = (target_classes == c).float()
        intersection = (p_c * t_c).sum()
        dice = (2 * intersection + 1.0) / (p_c.sum() + t_c.sum() + 1.0)
        dice_losses.append(1 - dice)
    return sum(dice_losses) / len(dice_losses)


def flow_matching_loss(model: FlowMatchingUNet3D, x1: torch.Tensor, device: torch.device,
                        x0: Optional[torch.Tensor] = None,
                        class_weights: Optional[torch.Tensor] = None,
                        pixel_loss_weight: float = 0.0,
                        dice_weight: float = 0.0,
                        dice_classes: tuple = (2, 3)) -> torch.Tensor:
    """
    x0: optional source batch (same shape as x1). Defaults to Gaussian noise (the standard
    unconditional flow-matching source) when omitted, preserving this function's original
    behavior for every existing caller. Pass a real batch to train a paired flow (e.g. the
    180-degree-rotation task in train_rotate180.py), where x0 is real data rather than noise.
    class_weights: optional [1, C, 1, 1, 1] per-channel weight (see parse_class_weights),
    applied to the per-voxel squared error before averaging. Plain per-voxel MSE treats a
    rare class (e.g. the lump, ~1% of voxels) exactly like background — getting it wrong
    barely moves the average loss, which is why the model learns the bulk anatomy well but
    is inconsistent on rare classes. Weighting by inverse class frequency corrects this.

    pixel_loss_weight: if > 0, adds a class-weighted cross-entropy term on the one-step
    endpoint estimate x1_hat = xt + (1-t)*v_pred (exact algebraically when v_pred == v_star,
    since xt + (1-t)*(x1-x0) == x1 -- so this reuses v_pred from the SAME forward pass, no
    extra network evaluation). This is NOT redundant with the velocity MSE term above: an
    MSE loss on x1_hat would be exactly (1-t)^2 * the velocity MSE (same minimizer, same
    gradient direction, just reweighted by t) -- pure algebraic restatement, not a new
    signal. Cross-entropy is a genuinely different loss landscape: MSE's gradient shrinks
    quadratically as a prediction approaches correct, so a velocity that's "80% of the right
    magnitude" gets only a weak extra push once loss is already fairly low. Cross-entropy's
    gradient stays large as long as the wrong class is still winning at a voxel, which
    directly targets a model that hedges with an undercommitted (too-small) velocity instead
    of actually crossing the decode threshold.

    dice_weight: if > 0, adds a soft Dice loss (on the same x1_hat estimate, for
    `dice_classes`) that penalizes predicted-volume mismatch symmetrically -- unlike
    class-weighted cross-entropy, which only weights by each voxel's TRUE class and so has
    no direct penalty for over-predicting a rare class where it doesn't belong.
    """
    x1 = x1.to(device)
    x0 = torch.randn_like(x1) if x0 is None else x0.to(device)
    t = torch.rand(x1.shape[0], device=device)
    t_ = t.view(-1, 1, 1, 1, 1)

    xt = (1 - t_) * x0 + t_ * x1
    v_star = x1 - x0
    v_pred = model(xt, t)
    sq_err = (v_pred - v_star) ** 2
    if class_weights is not None:
        sq_err = sq_err * class_weights
    loss = sq_err.mean()

    if pixel_loss_weight > 0 or dice_weight > 0:
        x1_hat = xt + (1 - t_) * v_pred
        target_classes = x1.argmax(dim=1)

    if pixel_loss_weight > 0:
        ce_weight = class_weights.flatten() if class_weights is not None else None
        pixel_loss = F.cross_entropy(x1_hat, target_classes, weight=ce_weight)
        loss = loss + pixel_loss_weight * pixel_loss

    if dice_weight > 0:
        # F.cross_entropy's `weight` is applied by each voxel's TRUE class, not its
        # predicted one -- missing a real lump voxel costs 12.57x/7.36x, but wrongly
        # predicting lump/pillar at a true-background voxel only costs 1.0x. That
        # asymmetry gives the model a direct incentive to over-predict these rare classes
        # everywhere it's unsure, which is exactly the "huge blob instead of a small lump"
        # failure this task's live runs showed (confirmed: ~24 disconnected predicted lump
        # components vs ~1 expected). Dice is inherently symmetric -- predicting excess
        # volume directly grows the denominator without growing the overlap numerator, so
        # over-prediction is penalized here even though it wasn't penalized by CE.
        probs = F.softmax(x1_hat, dim=1)
        dice_loss = soft_dice_loss(probs, target_classes, dice_classes)
        loss = loss + dice_weight * dice_loss

    return loss


def train_one_epoch(model, loader, optimizer, device, class_weights=None) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        optimizer.zero_grad()
        loss = flow_matching_loss(model, batch, device, class_weights=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.shape[0]
        n += batch.shape[0]
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, class_weights=None) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        loss = flow_matching_loss(model, batch, device, class_weights=class_weights)
        total_loss += loss.item() * batch.shape[0]
        n += batch.shape[0]
    return total_loss / max(n, 1)


def _json_safe_config(config: Config) -> dict:
    """asdict(config) can contain Path objects, which torch's default
    weights_only=True unpickling (PyTorch >=2.6) refuses to load. Keep the
    checkpoint's stored config to plain JSON-safe types (str/int/float/bool/None)."""
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()}


def save_checkpoint(path: Path, model, optimizer, epoch: int, config: Config, quiet: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": _json_safe_config(config),
    }, path)
    if not quiet:
        print(f"  saved checkpoint -> {path}")


def load_checkpoint(path: Path, model, optimizer=None) -> int:
    # weights_only=False: trusted, locally-generated checkpoint
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state.get("epoch", 0)


def build_run_name(config: Config) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    name = f"train_{timestamp}_bch{config.base_ch}_bs{config.batch_size}_lr{config.lr:.0e}"
    if config.run_tag:
        name = f"{name}_{config.run_tag}"
    return name


def main():
    print("=" * 60)
    print("Starting training")
    print("=" * 60)

    config = get_config()
    set_seed(config.seed)
    print(f"[1/5] Config: epochs={config.epochs}, batch_size={config.batch_size}, "
          f"base_ch={config.base_ch}, lr={config.lr}")

    device = get_device(config.device)
    print(f"[2/5] Using device: {device}")

    class_weights = parse_class_weights(config.class_weights, device)
    print(f"      class_weights (bg/insert/pillar/lump): {config.class_weights}")

    print(f"[3/5] Loading dataset from '{config.data_dir}' "
          f"(this can take a little while the first time)...")
    t0 = time.time()
    train_ds, val_ds = build_datasets(config.data_dir, val_frac=config.val_frac,
                                       num_orientations=config.num_orientations,
                                       seed=config.seed, num_classes=config.num_classes)
    train_loader, val_loader = build_dataloaders(train_ds, val_ds, batch_size=config.batch_size)
    print(f"      done in {time.time() - t0:.1f}s — train examples: {len(train_ds)}, "
          f"val examples: {len(val_ds)}")

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

    # Checkpoints live inside this run's own wandb folder (rather than a shared top-level
    # "checkpoints/" dir) so concurrent runs on the same machine can never collide on
    # filenames and silently overwrite each other's saves.
    if wandb.run is not None:
        config.checkpoint_dir = Path(wandb.run.dir) / "checkpoints"
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"      checkpoints -> {config.checkpoint_dir}")

    fixed_x0 = torch.randn(4, config.num_classes, 26, 128, 128, device=device)

    print("-" * 60)
    print(f"Training loop starting: {config.epochs} epochs")
    print("-" * 60)

    train_start = time.time()
    for epoch in range(config.epochs):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, class_weights=class_weights)
        val_loss = validate(model, val_loader, device, class_weights=class_weights)
        epoch_seconds = time.time() - epoch_start

        wandb.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "epoch": epoch,
            "epoch_of_total": epoch + 1,
            "epochs_total": config.epochs,
            "progress": (epoch + 1) / config.epochs,
            "epoch_seconds": epoch_seconds,
        })

        elapsed = time.time() - train_start
        remaining_epochs = config.epochs - (epoch + 1)
        avg_epoch_seconds = elapsed / (epoch + 1)
        eta = timedelta(seconds=int(remaining_epochs * avg_epoch_seconds))
        print(f"epoch {epoch + 1}/{config.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  ({epoch_seconds:.1f}s/epoch, ETA {eta})")

        if epoch % config.sample_every_epochs == 0:
            print("  generating sample grid for wandb...")
            x1_hat = sample(model, config.n_train_sample_steps, x0=fixed_x0, device=device)
            labels = [x1_hat[i].argmax(dim=0).cpu() for i in range(x1_hat.shape[0])]
            log_grid_to_wandb("samples/unconditional", labels,
                               [f"sample {i}" for i in range(len(labels))])

        if epoch % config.save_every_epochs == 0:
            save_checkpoint(config.checkpoint_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, config)
        save_checkpoint(config.checkpoint_dir / "latest.pt", model, optimizer, epoch, config, quiet=True)

    print("-" * 60)
    print(f"Training complete: {config.epochs} epochs in "
          f"{timedelta(seconds=int(time.time() - train_start))}")
    print(f"Final checkpoint: {config.checkpoint_dir / 'latest.pt'}")
    print("-" * 60)


if __name__ == "__main__":
    main()
