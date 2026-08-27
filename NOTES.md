# Diffusion project — quick notes

## Files looked at

### `visualize_inserts_3d.py`
A `visualize_inserts_3d` class built on **PyVista** for viewing 3D MRI-phantom
volumes (numpy arrays / torch tensors of shape `(K, H, W)`).

- Loads `.npy` volumes (or reconstructs them from a 4-channel torch tensor) via
  `load_array`, `load_array_from_path`, `load_folder`.
- Voxel values encode regions: `0` = background, `127` = insert, `200` = pillar,
  `255` = lump. `_make_meshes` thresholds each value into its own mesh
  (lump/insert/pillar), optionally smooths them, and renders with PyVista.
- Interactive key bindings: `n`/`b` step through loaded arrays, `g` toggles
  insert opacity, `r` resets camera, `s` screenshots, `h` exports HTML,
  `p` prints camera position, `q` quits.
- Can run interactively (`off_screen=False`) or headless or headless
  (`off_screen=True`, renders all arrays via `render_all()` and returns images).

### `mri_images_3D 2/*.npy`
Sample data the script consumes. One file inspected:
`ft1280c5_0.npy` → shape `(26, 128, 128)`, dtype `uint8`, values
`{0, 127, 200, 255}` — matches the label scheme in the script exactly.
Filenames look like `<caseid>_<index>.npy` (e.g. `ft1280c5_0/1/2...`,
`s2t10c0_0/1/2...`) — likely multiple slices/variants per case.

## Environment

- `conda env: uni` was missing `torch` and `pyvista` (and `vtk`, its
  rendering backend) — installed.

## Mission
TBD — will be described later.
