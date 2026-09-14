"""
Slower / manual checks not suited to a fast pytest suite: overfitting one batch,
a short smoke training run, and an inversion round-trip fidelity check against a
real checkpoint. Run via:

    python -m flow_model.sanity_checks --check overfit_one_batch
    python -m flow_model.sanity_checks --check smoke_train
    python -m flow_model.sanity_checks --check round_trip --checkpoint checkpoints/latest.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import Config
from .data import build_datasets, build_dataloaders
from .ode import invert, sample
from .train import flow_matching_loss, get_device, save_checkpoint, set_seed, train_one_epoch, validate
from .velocity_model import FlowMatchingUNet3D


def overfit_one_batch(steps: int = 300, device_name: str = "auto") -> None:
    """
    Validates the train/loss/backward/optimizer wiring in isolation from flow-matching
    theory. flow_matching_loss() resamples a fresh random x0 and t every call, which
    gives a genuinely stochastic regression target (v* = x1 - x0 depends on which x0
    was drawn) with an irreducible loss floor even for a perfectly-fit model — not a
    useful "does the optimizer work" check. Here we instead fix x0 and t ONCE outside
    the loop, so (xt, t) -> v* is a single deterministic example the model should be
    able to drive to ~0 loss given enough capacity and steps.
    """
    device = get_device(device_name)
    config = Config(base_ch=8, embed_channels=16)
    train_ds, _ = build_datasets(config.data_dir, val_frac=0.2, num_orientations=1, seed=0)
    train_loader, _ = build_dataloaders(train_ds, train_ds, batch_size=2)
    x1 = next(iter(train_loader)).to(device)

    torch.manual_seed(0)
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=device)
    t_ = t.view(-1, 1, 1, 1, 1)
    xt = (1 - t_) * x0 + t_ * x1
    v_star = x1 - x0

    model = FlowMatchingUNet3D(base_ch=config.base_ch, embed_channels=config.embed_channels,
                                dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    model.train()
    loss = None
    for step in range(steps):
        optimizer.zero_grad()
        v_pred = model(xt, t)
        loss = F.mse_loss(v_pred, v_star)
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"[overfit_one_batch] step {step}: loss={loss.item():.6f}")

    print(f"[overfit_one_batch] final loss={loss.item():.6f} (expect near 0)")


def smoke_train(epochs: int = 3, device_name: str = "auto") -> None:
    device = get_device(device_name)
    config = Config(base_ch=8, embed_channels=16, epochs=epochs,
                     checkpoint_dir=Path("checkpoints_smoke"))
    set_seed(config.seed)

    train_ds, val_ds = build_datasets(config.data_dir, val_frac=config.val_frac,
                                       num_orientations=config.num_orientations, seed=config.seed)
    train_loader, val_loader = build_dataloaders(train_ds, val_ds, batch_size=config.batch_size)
    print(f"[smoke_train] train examples: {len(train_ds)}, val examples: {len(val_ds)}")

    model = FlowMatchingUNet3D(base_ch=config.base_ch, embed_channels=config.embed_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        print(f"[smoke_train] epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    save_checkpoint(config.checkpoint_dir / "smoke.pt", model, optimizer, epochs - 1, config)
    print(f"[smoke_train] saved checkpoint to {config.checkpoint_dir / 'smoke.pt'}")


def round_trip_reconstruction_check(checkpoint_path: Path, n_steps_list=(10, 50, 100),
                                     device_name: str = "auto") -> None:
    device = get_device(device_name)
    # weights_only=False: trusted, locally-generated checkpoint
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = state.get("config", {})
    model = FlowMatchingUNet3D(
        num_classes=cfg.get("num_classes", 4),
        base_ch=cfg.get("base_ch", 8),
        embed_channels=cfg.get("embed_channels", 16),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    config = Config()
    _, val_ds = build_datasets(config.data_dir, val_frac=config.val_frac,
                                num_orientations=1, seed=config.seed,
                                num_classes=cfg.get("num_classes", 4))
    if len(val_ds) == 0:
        raise RuntimeError("Validation set is empty")

    x1 = val_ds[0].unsqueeze(0).to(device)
    original_label = x1.argmax(dim=1)[0]

    for n_steps in n_steps_list:
        x0_hat = invert(model, x1, n_steps)
        x1_hat = sample(model, n_steps, x0=x0_hat, device=device)
        reconstructed_label = x1_hat.argmax(dim=1)[0]
        agreement = (reconstructed_label == original_label).float().mean().item()
        print(f"[round_trip] n_steps={n_steps}: voxel agreement={agreement:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=str, required=True,
                         choices=["overfit_one_batch", "smoke_train", "round_trip"])
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pt"))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.check == "overfit_one_batch":
        overfit_one_batch(device_name=args.device)
    elif args.check == "smoke_train":
        smoke_train(device_name=args.device)
    elif args.check == "round_trip":
        round_trip_reconstruction_check(args.checkpoint, device_name=args.device)


if __name__ == "__main__":
    main()
