# Future idea: interpolate at partial inversion (t_mid > 0) instead of pure noise (t=0)

## The problem

Real-phantom interpolation (invert A and B to noise, slerp between the two noise points,
integrate forward) produces badly broken results — not just "duplicate lumps" but the whole
phantom shape tearing/fragmenting. This got *worse*, not better, after two attempted fixes:

- **Class-weighted loss** (upweighting the rare lump/pillar classes): made unconditional
  generation about the same, but made interpolation noticeably worse.
- **More ODE integration steps** (`n_steps` 50 → 150): produced an almost identical broken
  shape, ruling out numerical discretization error as the cause.

## What we confirmed instead

Ran the existing `sanity_checks.py --check round_trip` (invert a real phantom to noise, then
forward-sample it back, measure voxel agreement with the original). Result: **96-99% voxel
agreement at n_steps=50** on both an unweighted-loss checkpoint and the weighted-loss
checkpoint. So inversion itself is excellent — `x0_hat = invert(x1)` is a genuinely accurate
noise pre-image for that specific phantom.

This means the failure is specifically in the neighborhood **between** two individually-valid
inverted noise points, not in inversion itself. Training only ever shows the model a straight
line from a random Gaussian sample to *one specific* real phantom (flow matching's standard
setup) — it never sees points near, but not on, that line, and never sees the region between
two different phantoms' individual trajectories. At `t=0` (pure noise) that neighborhood is
about as far from anything the model was actually trained on as it gets: an enormous, sparsely
covered space where nothing constrains the field's behavior between two arbitrary points.

## The idea

Don't interpolate at `t=0`. Invert only partway to some `t_mid > 0` (e.g. 0.2-0.3), interpolate
the two phantoms' partial trajectories *there*, then integrate forward the rest of the way from
`t_mid` to `t=1`. Intuition: at `t_mid`, both trajectories are already pulled somewhat toward
real anatomy, so the neighborhood between them is more likely to be a region many training
trajectories actually pass near (rather than the wide-open, mostly-empty space at `t=0`).

This is the same idea behind SDEdit-style tricks in diffusion/flow models: interpolating or
editing at an intermediate noise level rather than the fully-noised endpoint, trading a bit of
"pure" latent-space interpolation for a result that stays closer to the learned manifold.

## Trade-off

Pick `t_mid` too high and the ODE has too little room left to reshape anything — the result
collapses toward a literal voxel-wise blend of A and B rather than a real interpolation. Too low
and we're back to the current problem. Would need to sweep a few values (e.g. 0.1, 0.2, 0.3) to
find where it actually helps.

## What implementing it needs

`ode.py`'s `invert()` currently always integrates the full `t=1 → t=0` range, and `sample()`
always starts at `t=0`. Both need a variant that can stop/start at an arbitrary `t_mid`:

- `invert(model, x1, n_steps, t_stop=0.0)` — stop integrating backward once `t` reaches `t_stop`
  instead of always going to 0.
- `sample(model, n_steps, x0=..., t_start=0.0)` — start integrating forward from `t_start`
  instead of always from 0.

Then in `interpolate_real`/`interpolate_noise`: invert both endpoints to `t_mid` instead of 0,
slerp/linear-interpolate at `t_mid`, and forward-sample starting from `t_mid`.

Cheap to try on the checkpoints we already have — no retraining needed.
