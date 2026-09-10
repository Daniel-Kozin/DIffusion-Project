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
    # Weight on an added cross-entropy term (see train.flow_matching_loss's docstring) on the
    # one-step endpoint estimate x1_hat = xt + (1-t)*v_pred, class-weighted the same way. Plain
    # velocity MSE lets the model hedge with an undercommitted (too-small) velocity that never
    # crosses the discrete decode threshold -- diagnosed directly on this task's first live run
    # (magnitude_ratio climbed to ~0.84 of target while the decoded prediction never moved at
    # all across 10/10 validation pairs, unaffected by using up to 20x more ODE steps).
    # Cross-entropy's gradient stays large as long as the wrong class is winning, targeting
    # that specific failure mode. 1.0 is a starting point pending calibration against the
    # velocity loss's own magnitude on a short run -- not yet empirically tuned.
    pixel_loss_weight: float = 1.0
    # Linearly decay pixel_loss_weight from its initial value to this one over the course of
    # training (by epoch_num/epochs, so it continues correctly across a --resume_from too).
    # The first full 1000-epoch run held pixel_loss_weight constant and the cross-entropy
    # term's magnitude push never leveled off (magnitude_ratio settled at ~1.25-1.3x target
    # instead of ~1.0, sustained overshoot for hundreds of epochs) -- let CE do the heavy
    # lifting early (escape the "doesn't move" regime) while fading it out so the
    # well-behaved, symmetric velocity MSE (and Dice, below) dominate refinement later.
    pixel_loss_weight_final: float = 0.3
    # Soft Dice loss weight on the lump/pillar channels (see flow_matching_loss's docstring)
    # -- targets the "huge blob instead of a small lump" failure mode found by direct visual
    # inspection of the first full run's predictions (~24 disconnected predicted lump
    # components vs ~1 expected), which plain class-weighted cross-entropy does not penalize
    # since it only weights by each voxel's TRUE class, not by over-predicting a wrong one.
    dice_weight: float = 1.0
    # EMA decay for a shadow copy of the model weights, used for preview/metric/checkpoint
    # saving instead of the raw (noisier) training weights -- standard practice in
    # diffusion/flow-matching training specifically because per-step weights fluctuate a lot,
    # which we saw directly: metrics50/centroid_error_voxels_mean swung between ~17 and ~60
    # from one 25-epoch checkpoint to the next in the first full run, without a clear trend.
    # 0.999 gives roughly a ~10-epoch smoothing window at this task's ~100 steps/epoch.
    ema_decay: float = 0.999
    # Weight on rollout_consistency_loss (see train.py's docstring) -- trains on a few real,
    # gradient-tracked ODE steps from x0 using the model's OWN predictions, instead of only
    # ever evaluating at exact points on the teacher-forced line between x0 and x1. Added
    # after confirming the exposure-bias gap directly: switching the sampler to Heun's method
    # or tripling the step count made no measurable difference to eval-time lump-mask IoU
    # (~0.09-0.10 either way), ruling out numerical integration error -- the learned field is
    # simply inaccurate once queried on states its own imperfect steps actually reach, which
    # this loss is the only one of this task's fixes so far to train on directly. 0.0 disables
    # it entirely (paired losses above still apply on their own).
    rollout_weight: float = 1.0
    n_rollout_steps: int = 3
    rollout_t_end: float = 0.3
    # Apply the rollout loss only every Nth training batch -- it costs n_rollout_steps extra
    # forward+backward passes (with the full computation graph kept live across them) versus
    # flow_matching_loss's one, so applying it every batch would meaningfully slow training.
    rollout_every_n_batches: int = 2
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

    sample_every_epochs: int = 25  # drives periodic preview-figure + aggregate-metric logging
    n_train_sample_steps: int = 30  # ODE steps for preview/metric sampling (forward-only, no invert)
    n_preview_val_pairs: int = 5  # small set that gets the qualitative 6-image figure each time
    # Larger fixed set used only for aggregate scalar metrics (mean/std graphed over epochs) --
    # per-pair scalars are too noisy to read a trend off (see rotate180 diagnostic discussion),
    # so this is deliberately much bigger than n_preview_val_pairs and never rendered as images.
    n_metric_val_pairs: int = 50
    checkpoint_epochs: str = "100,250,500,1000"  # comma-string, mirrors class_weights convention

    # Path to a checkpoint (e.g. .../checkpoints/latest.pt) to resume from -- loads model +
    # optimizer state and continues epoch numbering from where that checkpoint left off,
    # instead of restarting at epoch 1. Str (not Path) to match run_tag's convention for an
    # Optional field with a None default under build_config_from_cli's reflection.
    resume_from: Optional[str] = None
    # If resuming, reattach to this existing wandb run id (wandb.init(id=..., resume="must"))
    # so the continued epochs land in the same run/chart instead of starting a disconnected
    # new one. Leave unset to start a fresh run even when resuming model weights.
    wandb_resume_id: Optional[str] = None


def get_rotate_config() -> RotateConfig:
    return build_config_from_cli(RotateConfig, "180-degree rotation flow matching training config")
