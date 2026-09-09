from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import build_config_from_cli


@dataclass
class RotateConfig:
    data_dir: Path = Path("mri_images_3D")
    checkpoint_dir: Path = Path("checkpoints")

    num_classes: int = 4
    # Same convention/values as Config.class_weights (background/insert/pillar/lump).
    class_weights: str = "1.0,1.03,12.57,7.36"
    base_ch: int = 8
    embed_channels: int = 16
    dropout: float = 0.1
    use_grad_checkpoint: bool = False

    # Each dataset item now carries two full real volumes (x0 and x1) instead of one real +
    # one free noise tensor, roughly doubling per-sample host memory vs. the unconditional
    # task's batch_size=16 default -- start conservative, bump after confirming headroom.
    batch_size: int = 8
    # DataLoader worker processes for the CPU-side rotation/one-hot construction in
    # RotationPairDataset.__getitem__ -- unlike the unconditional task (x0 is free noise,
    # cheap to generate on the fly), every item here does two torchvision rotations, so a
    # single-process loader (num_workers=0) can leave the GPU underfed. 0 disables
    # multiprocessing entirely (safest default, matches every other script in this repo).
    num_workers: int = 0
    lr: float = 2e-4
    epochs: int = 1000
    val_frac: float = 0.2
    seed: int = 0
    device: str = "auto"  # "auto" -> cuda > mps > cpu

    wandb_project: str = "diffusion_project"
    wandb_mode: str = "online"
    run_tag: Optional[str] = "rotate180"

    sample_every_epochs: int = 25  # drives periodic preview-figure + preview-metric logging
    n_train_sample_steps: int = 30  # ODE steps for preview sampling (forward-only, no invert)
    n_preview_val_pairs: int = 2
    checkpoint_epochs: str = "100,250,500,1000"  # comma-string, mirrors class_weights convention


def get_rotate_config() -> RotateConfig:
    return build_config_from_cli(RotateConfig, "180-degree rotation flow matching training config")
