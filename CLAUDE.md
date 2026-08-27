# Working conventions for this repo

This project is worked on from multiple machines (Mac, and the RTX 3090 server at
`kozin@132.69.32.14:/mnt/data/kozin/diffusion`), often by both the user and Claude in
separate sessions. Git is the single source of truth for code — follow these two rules so no
session ever trains against, or edits on top of, a stale checkout.

## 1. Pull before starting work

At the start of any session (new conversation, or resuming after a break), before making any
changes or launching anything:

```bash
git status   # make sure there's nothing uncommitted sitting here already
git pull
```

If `git status` shows local changes that were never pushed, stop and reconcile them with the
user before pulling (don't silently discard or overwrite).

## 2. Push before starting a long/expensive run

Before kicking off any run that will take a while and produce results worth trusting
(a real training run, a batch of experiments, anything beyond a quick smoke test) — commit and
push first:

```bash
git add -A
git commit -m "..."
git push
```

The run should always start from code that's on GitHub. This is what makes a checkpoint or a
wandb run reproducible later, and it's what lets work be picked up from a different machine
mid-run without wondering which version of the code actually produced it.

Quick smoke tests / sanity checks / one-off probes (e.g. `mem_probe.py`, a 2-epoch timing
check) don't need this — only runs whose results you'd actually rely on.
