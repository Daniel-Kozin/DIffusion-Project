# How to run this project

Everything lives under `flow_model/`. Environment: conda env `uni`. All commands below
assume you're in `/Users/kozin/visual_code/diffusion`.

## 0. One-time setup

Already done, but for reference:
```bash
conda run -n uni python -m pip install torchvision wandb pytest
conda run -n uni wandb login
```

There's a known OpenMP crash on this Mac (`OMP: Error #15`) unless `KMP_DUPLICATE_LIB_OK=TRUE`
is set — the run scripts (`run_train.sh`, `run_interpolate.sh`) already set this for you, so
just use those instead of calling `conda run ... python -m flow_model....` directly.

## 1. Sanity checks (fast, do this first if you've changed anything)

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run -n uni python -m pytest flow_model/tests/ -v
```

Slower manual checks (not part of the pytest suite):
```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n uni python -m flow_model.sanity_checks --check overfit_one_batch
KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n uni python -m flow_model.sanity_checks --check smoke_train
KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n uni python -m flow_model.sanity_checks --check round_trip --checkpoint checkpoints/latest.pt
```

## 2. Training

```bash
./run_train.sh                       # Mac (auto-detects MPS)
./run_train.sh --device cuda         # A100 server, if the Mac is too slow/tight on memory
```

Useful flags (see `flow_model/config.py` for the full list):
- `--epochs 150` — how many epochs (default 150)
- `--batch_size 8` — default, chosen from `mem_probe.py` results on this Mac (see below);
  bump higher on the A100 server
- `--base_ch 8` — width of the UNet; raise on the A100 server for a bigger model
- `--use_grad_checkpoint true` — trade compute for memory if you hit OOM/swap on the Mac
- `--run_tag my_experiment_name` — appended to the auto-generated wandb run name so you
  can tell experiments apart at a glance

**Checking memory headroom before a long run** — `mem_probe.py` runs one real forward+backward
pass at a given batch size and reports actual MPS memory used, so you can tune `batch_size`
without committing to a full run:
```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run -n uni python -m flow_model.mem_probe --batch_size 8
```
Measured on this Mac (M5, ~25.8GB total, ~10GB already used by the OS/other apps at idle):
`bs=2`→2.42GB, `bs=4`→3.86GB, `bs=8`→7.95GB, `bs=16`→15.36GB (roughly linear, ~1GB/sample).
`batch_size=8` was picked as the default because it leaves ~8GB of headroom for a long
unattended run; `16` would leave almost none once the OS's own usage is accounted for.

Every run logs to the **`diffusion_project`** project in wandb, with a unique, descriptive
run name (e.g. `fm3d_bch8_bs2_lr2e-04_0827_1610`). Each epoch logs `train/loss`, `val/loss`,
and progress (`epoch_of_total`, `epochs_total`, `progress`) — both to the terminal
(`epoch 12/150: ...`) and to wandb. Every 10 epochs it also logs a grid of 4 unconditional
samples generated from a **fixed** noise seed, so you can watch generation quality improve
over the course of training in the wandb UI.

Checkpoints save to `checkpoints/`: `latest.pt` every epoch, `epoch_XXXX.pt` every 10 epochs.

**Before trusting anything downstream**: open the wandb run and look at the
`samples/unconditional` panel over time. If by the end of training the samples still look
like noise or a single blob rather than recognizable phantom shapes, the model isn't ready
yet — keep training (more epochs, or a bigger `base_ch` on the A100) before running
interpolation.

## 3. Interpolation (idea 1 — the actual point of this phase)

Once you have a checkpoint you trust:
```bash
./run_interpolate.sh --checkpoint checkpoints/latest.pt
```

This does two things and logs both to wandb under the same run:
1. **Real-phantom interpolation**: picks two real validation phantoms, inverts each to its
   noise/latent via the ODE, interpolates the latents, integrates forward at each alpha.
2. **Noise interpolation**: samples two random noise vectors directly (no inversion) and
   does the same interpolate → forward-integrate at each alpha.

For each alpha step, one figure gets logged (tag `interpolation/real` / `interpolation/noise`):
left panel shows the two endpoints on a line with a dotted connector and a star marking the
current alpha; right panel shows the two endpoint renders on top and the interpolated result
centered below. Since each step logs under the same tag, wandb shows them as a scrollable
sequence — sliding through it is basically watching the morph.

Useful flags:
- `--n_steps 50` — ODE integration steps (both directions). More steps = more accurate
  inversion, at the cost of runtime; sanity_checks' `round_trip` check can help you judge
  whether your current n_steps is enough for a given checkpoint.
- `--alphas 0.0 0.25 0.5 0.75 1.0` — which interpolation points to compute (space-separated)
- `--method slerp` (default) or `--method linear` — slerp is recommended; linear interpolation
  of two ~N(0,I) points drifts toward the origin, off the Gaussian's typical shell

## Other things worth knowing

- **Data**: `mri_images_3D/` has 123 files / 40 case IDs (the 8 double/triple-lump phantoms
  were removed). Training augments with 8x in-plane rotation per case (→ ~984 volumes),
  applied only to the train split; val stays unaugmented for a clean signal.
- **Device**: training/interpolation auto-detect `cuda` > `mps` > `cpu` unless you pass
  `--device` explicitly.
- **wandb offline fallback**: if you're ever somewhere without a connection, add
  `--wandb_mode offline`, then `wandb sync <run_dir>` later to upload.
- **If the Mac struggles** (OOM, swapping, very slow steps): try `--use_grad_checkpoint true`
  first, then `--batch_size 1`; if it's still too slow, switch to the A100 server with
  `--device cuda` — no other code changes needed.
- **Reference-only files** (read but never modified): the UNet3D building blocks in
  `flow_model/unet3d/` were copied from `~/PycharmProjects/artificial_palpation`'s
  `conditional_diffusion/` module.

## What's left to decide as you go

- How many epochs is "enough" — watch the wandb sample grid rather than picking a number in
  advance.
- Whether `base_ch=8` is enough model capacity, or worth increasing on the A100.
- Whether the default `n_steps=50` for interpolation gives faithful-enough inversion for your
  trained model — check with `sanity_checks --check round_trip` at a few step counts first.
