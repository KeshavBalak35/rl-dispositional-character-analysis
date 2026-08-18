#!/usr/bin/env python3
"""
Step 1: fit the steering direction, and reserve a held-out problem set.

    python fit_direction.py --layer 16
    python fit_direction.py --layer 16 --run probe_rh --test-size 0.3

Computes mean(hack) - mean(no-hack) at one layer, using ONLY the training half
of a problem-grouped split. The other half is written out as the held-out
problem set for the alpha sweep.

WHY THE SPLIT MATTERS HERE
    If the direction is fitted on all problems and then steered on problems that
    were in the fit, the causal test is contaminated by exactly the leakage
    splits.py exists to prevent: the direction has seen those problems' hack
    activations, so moving their hack rate is partly circular. Grouping is by
    problem, so a problem's k samples never straddle the boundary.

OUTPUT
    <root>/_steering/<name>.npz    direction, plus the raw class means
    <root>/_steering/<name>.json   layer, norms, split sizes, holdout problem ids

    alpha in the sweep is expressed in multiples of the TYPICAL ACTIVATION NORM
    at this layer, recorded here, so "alpha=2" means the same perturbation
    magnitude regardless of layer or model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                   # noqa: E402

from coding_eval import (                                            # noqa: E402
    default_root, group_holdout_split, load_run, probe_dataset, run_dir,
    save_direction,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="probe_rh", help="probe run to fit from")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None, help="output name; default direction_L<layer>")
    args = ap.parse_args()

    path = run_dir(args.run, create=False)
    if not os.path.isdir(path):
        print(f"missing run: {path}")
        return 1

    records = load_run(path, require_activations=False)
    X, y, keep = probe_dataset(records, args.layer)
    print(f"{args.run}: {len(keep)} usable rows "
          f"({int((y == 1).sum())} hack, {int((y == 0).sum())} no-hack), "
          f"{len({r.group_key for r in keep})} problems")

    if (y == 0).sum() < 30:
        print(f"  WARNING: only {int((y==0).sum())} no-hack examples. The negative "
              "class mean is the noisier half of this difference; the direction "
              "will be correspondingly uncertain.")

    # ---- grouped split: fit on train, steer on holdout --------------------
    tr, te = group_holdout_split(keep, test_size=args.test_size, seed=args.seed,
                                 labels=y)
    tr_groups = {keep[i].group_key for i in tr}
    te_groups = {keep[i].group_key for i in te}
    assert not (tr_groups & te_groups)
    print(f"  split: {len(tr)} train rows / {len(te)} holdout rows, "
          f"{len(tr_groups)} / {len(te_groups)} problems, no overlap")

    ytr = y[tr]
    if (ytr == 1).sum() == 0 or (ytr == 0).sum() == 0:
        print("  train half has only one class; cannot form a difference of means")
        return 1

    mu_hack = X[tr][ytr == 1].mean(axis=0)
    mu_clean = X[tr][ytr == 0].mean(axis=0)
    raw = mu_hack - mu_clean
    norm = float(np.linalg.norm(raw))
    direction = raw / norm

    typical = float(np.linalg.norm(X[tr], axis=1).mean())
    print(f"  |mu_hack - mu_noHack| = {norm:.2f}")
    print(f"  typical activation norm at layer {args.layer} = {typical:.2f}")
    print(f"  ratio = {norm/typical:.3f}  (how far apart the class means are, "
          "relative to a typical activation)")

    # Sanity: the direction should separate the HOLDOUT rows too. If it does not,
    # it is fitting noise and steering with it will do nothing interpretable.
    proj_te = X[te] @ direction
    yte = y[te]
    if (yte == 1).sum() and (yte == 0).sum():
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(yte, proj_te)
        print(f"  holdout AUC from this direction alone: {auc:.3f}")
        if auc < 0.7:
            print("    WARNING: weak. Steering along it is unlikely to move behaviour.")

    # ---- held-out problems, deduplicated ---------------------------------
    seen, holdout_problems = set(), []
    for i in te:
        p = keep[i].generation.problem
        if p.problem_id not in seen:
            seen.add(p.problem_id)
            holdout_problems.append(p)
    conds = Counter(keep[i].generation.condition for i in te)
    print(f"  holdout: {len(holdout_problems)} unique problems, conditions={dict(conds)}")

    name = args.name or f"direction_L{args.layer}"
    out_dir = os.path.join(default_root(), "_steering")
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "name": name, "source_run": args.run, "layer": args.layer,
        "pooling": next((r.activations.pooling for r in keep if r.activations), None),
        "model_id": sorted({r.generation.model_id for r in keep}),
        "diff_norm": norm, "typical_activation_norm": typical,
        "n_train_rows": len(tr), "n_holdout_rows": len(te),
        "n_train_problems": len(tr_groups), "n_holdout_problems": len(te_groups),
        "seed": args.seed, "test_size": args.test_size,
        "holdout_problem_ids": sorted(seen),
        "holdout_conditions": dict(conds),
    }
    # Shared writer, so the NPZ/JSON split and the key spellings match exactly
    # what load_direction() expects. Three scripts read these files and each
    # used to parse them itself.
    save_direction(name, direction=direction, layer=args.layer,
                   typical_norm=typical, holdout_problem_ids=sorted(seen),
                   arrays={"mu_hack": mu_hack, "mu_clean": mu_clean}, meta=meta)

    print(f"\nsaved {out_dir}/{name}.npz and .json")
    print(f"next: python check_alpha_zero.py --direction {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
