# Diffusion Project — Daniel Kozin (ID 215550260)

Course project submission: flow matching over 3D breast-phantom volumes, an interpolation
attempt that failed, and a self-supervised 180-degree rotation task that worked. See
`paper/summary_paper.pdf` for the full writeup.

## Contents

- **`paper/`** — the submitted paper.
  - `summary_paper.pdf` — final paper.
  - `summary_paper.tex` + `imgs/` — LaTeX source and figures, self-contained (compiles with
    `tectonic summary_paper.tex` or any standard `pdflatex`-based toolchain).

- **`presentation/`** — the project presentation (`Diffusion_Project_Presentation.pptx`).

- **`code/`** — the full modeling codebase.
  - `flow_model/` — the package: data loading, the 3D U-Net (`unet3d/`), the flow-matching
    velocity model, the ODE sampler/inverter (`ode.py`), training scripts for both tasks
    (`train.py`, `train_rotate180.py`), evaluation scripts (`eval_rotate180.py`,
    `eval_trajectory.py`), metrics (`metrics.py`), and unit tests (`tests/`).
  - `scripts/` — shell wrappers used to launch each run (training, evaluation, sweeps).
  - `showcase.py`, `view.py`, `visualize_inserts_3d.py` — standalone visualization utilities.
  - `requirements.txt` — pinned versions of the key packages (PyTorch, PyVista/VTK for 3D
    rendering, wandb for experiment tracking, etc.) from the environment the project ran in.

- **`data/mri_images_3D/`** — the raw dataset: 123 `.npy` volumes (26 x 128 x 128 voxels, one
  of 4 classes per voxel), expanded to 984 training volumes via 8x rotation augmentation inside
  the data loader. See the paper's Data section for the class-balance breakdown.

- **`checkpoints/`** — the four checkpoints behind the paper's final checkpoint-selection table
  (raw and EMA weights at epoch 1250 and epoch 1500 of the rollout-consistency fine-tune).
  `epoch_1250_ema.pt` is the final validated model reported in the paper (combined mask Dice
  $0.453\pm0.151$ on 176 held-out pairs). Earlier checkpoints from the abandoned interpolation
  attempt (Part I) are not included here since they are superseded by the rotation-task result;
  ask if you need the full training history.

- **`docs/`** — supporting material referenced throughout the paper.
  - `project_summary.pdf` — the full, detailed process log this paper was distilled from
    (every loss/metric change tried, in the order it was tried).
  - `ARCHITECTURE.pdf` — a short explainer of the model (flow matching vs. DDPM, the 3D U-Net).
  - `HANDOFF.md` — environment/setup notes from when this project moved machines.

## Reproducing

```bash
cd code
pip install -r requirements.txt
# training (rotation task), data_dir/checkpoint_dir default to ./mri_images_3D and
# ./checkpoints, so either run from a directory containing those (e.g. symlink or copy them
# next to flow_model/), or override on the command line:
bash scripts/run_train_rotate180.sh --data_dir ../data/mri_images_3D --checkpoint_dir ../checkpoints
# evaluation on the final checkpoint:
bash scripts/run_eval_rotate180.sh --data_dir ../data/mri_images_3D --checkpoint_dir ../checkpoints
```

Both scripts run everything inside the `uni` conda environment (`conda run -n uni ...`); create
an equivalent environment from `requirements.txt` first, or edit the scripts to drop the
`conda run -n uni` wrapper if running from an already-active environment.
