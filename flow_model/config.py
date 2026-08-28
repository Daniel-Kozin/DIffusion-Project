import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    data_dir: Path = Path("mri_images_3D")
    checkpoint_dir: Path = Path("checkpoints")

    num_classes: int = 4
    # Per-class weight on the flow-matching MSE loss, channel order matching the dataset's
    # one-hot encoding (background, insert, pillar, lump). Plain per-voxel MSE treats a rare
    # class exactly like background, so the model learns the bulk anatomy well but is
    # inconsistent on rare classes (pillar is ~0.3% of voxels, lump ~0.9%, measured over the
    # full dataset). Weights are sqrt(inverse class frequency), normalized to background=1 —
    # sqrt rather than raw inverse frequency to avoid letting the ~150x-rarer pillar class
    # dominate the loss and destabilize training. Override with --class_weights if needed.
    class_weights: str = "1.0,1.03,12.57,7.36"
    base_ch: int = 8
    embed_channels: int = 16
    dropout: float = 0.1
    use_grad_checkpoint: bool = False

    batch_size: int = 16
    lr: float = 2e-4
    epochs: int = 150
    val_frac: float = 0.2
    num_orientations: int = 8
    seed: int = 0
    device: str = "auto"  # "auto" -> cuda > mps > cpu

    wandb_project: str = "diffusion_project"
    wandb_mode: str = "online"
    run_tag: Optional[str] = None

    sample_every_epochs: int = 10
    save_every_epochs: int = 100
    n_train_sample_steps: int = 30


def get_config() -> Config:
    parser = argparse.ArgumentParser(description="Flow matching training config")
    defaults = Config()
    for f in fields(Config):
        arg_name = f"--{f.name}"
        if f.type in (Path, "Path"):
            parser.add_argument(arg_name, type=Path, default=getattr(defaults, f.name))
        elif f.type == bool or f.type == "bool":
            parser.add_argument(arg_name, type=lambda v: str(v).lower() in ("1", "true", "yes"),
                                 default=getattr(defaults, f.name))
        else:
            parser.add_argument(arg_name, type=type(getattr(defaults, f.name)) if getattr(defaults, f.name) is not None else str,
                                 default=getattr(defaults, f.name))
    args = parser.parse_args()
    return Config(**vars(args))
