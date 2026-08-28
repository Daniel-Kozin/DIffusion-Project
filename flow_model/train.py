import random
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

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


def flow_matching_loss(model: FlowMatchingUNet3D, x1: torch.Tensor, device: torch.device) -> torch.Tensor:
    x1 = x1.to(device)
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=device)
    t_ = t.view(-1, 1, 1, 1, 1)

    xt = (1 - t_) * x0 + t_ * x1
    v_star = x1 - x0
    v_pred = model(xt, t)
    return F.mse_loss(v_pred, v_star)


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        optimizer.zero_grad()
        loss = flow_matching_loss(model, batch, device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.shape[0]
        n += batch.shape[0]
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        loss = flow_matching_loss(model, batch, device)
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
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
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
