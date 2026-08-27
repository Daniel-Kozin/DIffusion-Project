# Handoff / status summary

Read this first if picking up this project in a new session (e.g. on the RTX 3090 server
via VS Code). Written 2026-08-27.

## What this project is

Course project on diffusion models: train an unconditional flow-matching generative model
on a 3D breast-phantom voxel dataset, then interpolate between two REAL phantoms by
ODE-inverting them to noise, interpolating the noise, and integrating forward — plus a
separate exploration interpolating directly between two random noise vectors. Full writeup
in `latex/main.tex`.

## Where the code lives

- **GitHub**: https://github.com/Daniel-Kozin/DIffusion-Project (branch `main`) — source of
  truth for code. `mri_images_3D/` (data), `checkpoints/`, `wandb/`, `screenshots/` are
  gitignored — not tracked, must be synced separately (rsync) if moving to a new machine.
- **Mac** (`/Users/kozin/visual_code/diffusion`): where all of this was built. Was running a
  full 150-epoch training job locally (via `caffeinate -i ./run_train.sh`) — **being stopped**
  in favor of the server, which will be much faster (MPS backend is slow for 3D conv/attention
  vs CUDA; measured ~302s/epoch on the Mac at batch_size=8).
- **Server** (`kozin@132.69.32.14`, path `/mnt/data/kozin/diffusion`): 4x RTX 3090 (24GB each,
  CUDA 12.6), currently idle. Code pulled from GitHub. Data was rsynced there directly (not
  via git). This is now the intended place to run real training.

## What's built (`flow_model/` package)

- `unet3d/` — 3D UNet backbone, copied verbatim from a related prior project at
  `~/PycharmProjects/artificial_palpation` (reference only, never modified).
- `velocity_model.py` — `FlowMatchingUNet3D`: sinusoidal time embedding + the UNet, optional
  gradient checkpointing (`use_grad_checkpoint`).
- `data.py` — loads `.npy` phantoms, remaps raw values `{0,127,200,255}` → class indices
  `{0,1,2,3}` (bg/insert/pillar/lump), 8x rotation augmentation (train split only), case-ID
  train/val split (not per-file, to avoid leaking rotated copies across the split).
- `ode.py` — `sample()` (forward ODE integration, noise→data) and `invert()` (backward,
  data→noise) — the core of the interpolation idea.
- `interpolate.py` — CLI: real-phantom interpolation (invert both endpoints, interpolate
  latents, integrate forward) and noise interpolation (sample two random latents directly).
  Logs a 2-panel figure per alpha step to wandb: left = lump-centroid schematic (real spatial
  position of each endpoint's lump, dotted line, alpha marker), right = A/B renders on top +
  interpolated result centered below.
- `train.py` — training loop, flow-matching MSE loss, per-epoch progress prints with ETA,
  checkpointing, wandb logging (project `diffusion_project`), descriptive run names
  (`train_<date>_bch..._bs..._lr...`).
- `mem_probe.py` — measures actual GPU memory for a given batch size without a full run.
- `flow_model/tests/` — 10 pytest tests, all passing (dataset, model shapes, ODE, grad
  checkpointing correctness).
- `sanity_checks.py` — slower manual checks: overfit-one-batch, short smoke-train,
  inversion round-trip fidelity.
- `run.md` — full usage instructions (read this for "how do I run X").

## Data cleanup already done

Deleted 8 case IDs (23 files) that were confirmed (visually, via lump-mask connected
components) to be double/triple-lump phantoms — not needed for this project's single-lump
framing. **123 files / 40 case IDs remain.** With 8x rotation augmentation: ~984 volumes.

## Key defaults / decisions so far

- `batch_size=8` (Mac default) — chosen from `mem_probe.py` measurements: bs2=2.42GB,
  bs4=3.86GB, bs8=7.95GB, bs16=15.36GB, vs. the Mac's ~24GB *shared* unified memory (already
  ~10GB used by the OS at idle). **This number does NOT apply to the RTX 3090 server** — the
  3090 has 24GB *dedicated* VRAM, not shared with the OS, so batch_size should very likely be
  pushed much higher there. Re-run `mem_probe.py` on the server before starting the real run:
  ```bash
  conda run -n uni python -m flow_model.mem_probe --batch_size 8   # then try 16, 32, 64...
  ```
  (no `KMP_DUPLICATE_LIB_OK`/MPS env vars needed on Linux/CUDA — those were Mac-only
  workarounds; harmless if left set, just unnecessary).
- `base_ch=8`, `epochs=150` (default, not yet tuned/justified — just a starting point).
- wandb project: `diffusion_project`. Already logged in on the Mac; **check whether wandb is
  logged in on the server too** (`conda run -n uni wandb login` if not).
- Device auto-detection (`get_device()`) already checks `cuda` before `mps`, so no code
  changes needed to run on the server — just run `./run_train.sh` there.

## Known gotchas already fixed (don't re-discover these)

- `conda run` buffers stdout by default, silently hiding `print()` output — fixed via
  `python -u` + `PYTHONUNBUFFERED=1` + `conda run --no-capture-output` in both run scripts.
- Checkpoints used to fail to reload under PyTorch 2.6+'s default `weights_only=True` because
  the saved config dict contained `Path` objects — fixed by JSON-sanitizing the saved config
  and explicitly passing `weights_only=False` on load (all checkpoints are locally-generated
  and trusted).
- `KMP_DUPLICATE_LIB_OK=TRUE` is required for `import torch` to not crash on the Mac
  specifically (OpenMP duplicate-library conflict) — Mac-only, not relevant on the server.

## Where things stand / next steps

1. Confirm code is correctly checked out on the server (there was a `git checkout` conflict
   from stale pre-fix files left by the original rsync — should be resolved by now, but worth
   double-checking `git status` is clean there).
2. Run `mem_probe.py` on the server to find a good `batch_size` for the RTX 3090s (see above).
3. Start the real training run on the server (`./run_train.sh --batch_size <X>`), watch the
   `diffusion_project` wandb project's `samples/unconditional` panel over time to judge when
   the model is actually generating recognizable phantom shapes (not a fixed epoch count).
4. Once there's a checkpoint you trust, run `./run_interpolate.sh --checkpoint checkpoints/latest.pt`
   for the actual deliverable (real-phantom + noise interpolation figures in wandb).
5. The Mac's in-progress `checkpoints/`/`wandb/` run (started ~17:16) is being abandoned in
   favor of the server — fine to delete once you've confirmed the server run is going.

## Reference docs

- `run.md` — command reference for everything above.
- `/Users/kozin/.claude/plans/ok-now-a-few-golden-lecun.md` — the original approved
  implementation plan (more architectural detail/rationale than this file).
