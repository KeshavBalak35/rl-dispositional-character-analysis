#!/usr/bin/env python3
"""
Top-k difference subspace on LENGTH-RESIDUALIZED activations.

The causal analog of --length-check's correlational finding that the leading
component is length-loaded. Residualize first, then run the same uncentered SVD
fit_direction.py --subspace-k runs, and compare the variance table.

WHY THE PLAIN REMOVAL FRACTION DOES NOT PREDICT THIS

fit_direction_residualized.py reports removal fraction

    ||(l_hack - l_clean) . beta|| / ||mu_hack - mu_clean||

which depends on the BETWEEN-CLASS GAP IN MEAN LENGTH. The SVD runs on
(hack rows - mu_clean), whose variance is driven by the WITHIN-CLASS SPREAD of
length. Those are independent. Classes can average the same length (removal
fraction ~0) while length varies enormously inside each class, in which case
residualization barely moves the mean direction and guts the subspace.

Simulated, class gap 0.31 tokens: removal fraction 0.005, yet 21.7% of the
difference variance removed and the length component annihilated. A small
removal fraction is not evidence that the subspace is unaffected.

TWO DENOMINATORS, ALWAYS BOTH

Residualization shrinks the total. A component's share of the RESIDUAL total
therefore rises even when the component itself is untouched. In the simulation
PC1 went 0.478 -> 0.610 while its share of the ORIGINAL total stayed 0.478.
Reading only the first column turns a renormalization artifact into a finding,
so the table prints both and you compare the ORIGINAL-denominator column against
your raw 40.7%.

WHAT RESIDUALIZATION DOES NOT DO

It does not force the direction orthogonal to the length axis. It removes the
length-VARYING component from the activations; if the class mean gap happens to
point along beta for reasons unrelated to length, that survives. So a component
can still show a high cosine to the length axis afterwards. What must drop is
the length association measured ON RESIDUALIZED activations, which the table
reports per component.

The SVD block below is copied from fit_direction.py --subspace-k, including the
uncentered decomposition and the sign orientation, so the two tables are
directly comparable.

USAGE

  python fit_subspace_residualized.py --run probe_rh_first8 --layer 16 --subspace-k 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np


def subspace_table(pos, mu_clean, direction, k):
    """
    Exactly fit_direction.py's extraction: uncentered SVD on (pos - mu_clean),
    components sign-oriented to project positively onto the differences.

    Returns (components, var_shares, total_sq).
    """
    diffs = pos - mu_clean
    _U, S, Vt = np.linalg.svd(diffs, full_matrices=False)
    k = int(min(k, len(pos) - 1, diffs.shape[1]))
    comps = np.ascontiguousarray(Vt[:k], dtype=np.float32)
    proj = diffs @ comps.T
    flip = np.where(proj.mean(axis=0) < 0, -1.0, 1.0).astype(np.float32)
    comps = comps * flip[:, None]
    total = float((S ** 2).sum())
    return comps, (S ** 2) / max(1e-12, total), total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="probe_rh_first8")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--subspace-k", type=int, default=8)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--covariates", nargs="*", default=["length"],
                    choices=["length", "prompt_length", "condition"])
    ap.add_argument("--condition-auc-limit", type=float, default=0.9)
    ap.add_argument("--force-condition", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    from coding_eval import load_run, probe_dataset, run_dir
    from coding_eval.splits import group_holdout_split
    from coding_eval.steering import save_direction

    path = args.run if os.path.isdir(args.run) else run_dir(args.run, create=False)
    records = load_run(path, require_activations=True)
    X, y, keep = probe_dataset(records, args.layer)
    pooling = next((r.activations.pooling for r in keep if r.activations), None)
    print(f"{args.run}: {len(keep)} rows ({int((y == 1).sum())} hack, "
          f"{int((y == 0).sum())} no-hack), pooling={pooling!r}")

    tr, te = group_holdout_split(keep, test_size=args.test_size, seed=args.seed,
                                 labels=y)
    assert not ({keep[i].group_key for i in tr} & {keep[i].group_key for i in te})
    ytr = y[tr]
    if (ytr == 1).sum() == 0 or (ytr == 0).sum() == 0:
        print("train half has only one class")
        return 1

    lengths = np.array([r.generation.response_token_len for r in keep], float)
    conditions = np.array([r.generation.condition or "<unset>" for r in keep])

    # ---- covariate design, condition guarded -------------------------------
    cols, names = [], []
    if "length" in args.covariates:
        cols.append(lengths); names.append("response_token_len")
    if "prompt_length" in args.covariates:
        cols.append(np.array([r.generation.prompt_token_len for r in keep], float))
        names.append("prompt_token_len")
    if "condition" in args.covariates:
        levels = sorted(set(conditions))
        if len(levels) > 1:
            from sklearn.metrics import roc_auc_score
            best = max(max(roc_auc_score(y, (conditions == l).astype(float)),
                           1 - roc_auc_score(y, (conditions == l).astype(float)))
                       for l in levels)
            print(f"  AUC(condition -> label) = {best:.3f}")
            if best >= args.condition_auc_limit and not args.force_condition:
                print("  REFUSING: condition nearly determines the label, so "
                      "residualizing on it removes the label.")
                return 1
            for lev in levels[1:]:
                cols.append((conditions == lev).astype(float))
                names.append(f"condition[{lev}]")
    if not cols:
        print("no covariates selected")
        return 1
    M = np.column_stack(cols)
    print(f"  covariates: {names}")

    # ---- residualize: coefficients from TRAIN, centred so the mean survives --
    M_mean = M[tr].mean(axis=0)
    A_tr = np.column_stack([np.ones(len(tr)), M[tr] - M_mean])
    B = np.linalg.lstsq(A_tr, X[tr], rcond=None)[0][1:]
    X_res = X - (M - M_mean) @ B
    beta_hat = B[0] / (np.linalg.norm(B[0]) + 1e-12)      # unit length axis

    # ---- raw and residualized subspaces ------------------------------------
    out = {}
    for tag, Z in (("raw", X), ("resid", X_res)):
        mu_h = Z[tr][ytr == 1].mean(0)
        mu_c = Z[tr][ytr == 0].mean(0)
        raw = mu_h - mu_c
        nrm = float(np.linalg.norm(raw))
        if nrm == 0:
            print(f"{tag}: class means identical")
            return 1
        d = raw / nrm
        comps, var, total = subspace_table(Z[tr][ytr == 1], mu_c, d,
                                           args.subspace_k)
        out[tag] = {"mu_hack": mu_h, "mu_clean": mu_c, "direction": d,
                    "norm": nrm, "components": comps, "var": var, "total": total}

    dcov = M[tr][ytr == 1].mean(0) - M[tr][ytr == 0].mean(0)
    removed = dcov @ B
    frac_removed = float(np.linalg.norm(removed) / out["raw"]["norm"])
    var_removed = 1.0 - out["resid"]["total"] / out["raw"]["total"]
    ratio = out["resid"]["total"] / out["raw"]["total"]

    print("\n" + "=" * 78)
    print("MAGNITUDES")
    print("=" * 78)
    for nm, dv in zip(names, dcov):
        print(f"  class gap in {nm}: {dv:+.2f}")
    print(f"  plain-direction removal fraction   {frac_removed:8.4f}  "
          "(between-class gap)")
    print(f"  DIFFERENCE VARIANCE REMOVED        {var_removed:8.4f}  "
          "(within-class spread)  <-- the one that governs the table below")
    print(f"  cosine(raw direction, resid direction) "
          f"{float(out['raw']['direction'] @ out['resid']['direction']):8.4f}")

    # ---- the comparison table ----------------------------------------------
    k = min(len(out["raw"]["components"]), len(out["resid"]["components"]))
    print("\n" + "=" * 78)
    print("EXPLAINED VARIANCE PER COMPONENT")
    print("compare your raw 40.7% against the 'resid, ORIG denom' column, not "
          "'resid share'")
    print("=" * 78)
    print(f"{'comp':>5}{'raw share':>11}{'resid share':>13}{'resid ORIG':>12}"
          f"{'cos to len raw':>16}{'cos to len res':>16}")
    for i in range(k):
        vr, vs = float(out["raw"]["var"][i]), float(out["resid"]["var"][i])
        cr = float(abs(out["raw"]["components"][i] @ beta_hat))
        cs = float(abs(out["resid"]["components"][i] @ beta_hat))
        print(f"{i:>5}{vr:>11.3f}{vs:>13.3f}{vs * ratio:>12.3f}"
              f"{cr:>16.3f}{cs:>16.3f}")
    print("\n  'comp 0' is what fit_direction.py's table calls PC1 and what "
          "--length-check calls PC0. Same vector, two names; keep them straight "
          "in the writeup.")

    v0_raw = float(out["raw"]["var"][0])
    v0_adj = float(out["resid"]["var"][0]) * ratio
    print(f"\n  comp 0: {v0_raw:.1%} raw -> {v0_adj:.1%} on the original "
          f"denominator ({v0_adj - v0_raw:+.1%})")
    if abs(v0_adj - v0_raw) < 0.03:
        print("  Essentially unchanged: the leading component is not a length "
              "artifact, whatever its correlation with length.")
    else:
        print("  Materially changed: the leading component depended on the "
              "length-varying part of the activations.")
    n_pos = int((ytr == 1).sum())
    if n_pos < 30:
        print(f"  WARNING: only {n_pos} positives; these components are noisy")

    # ---- save, drop-in compatible ------------------------------------------
    name = args.name or f"direction_L{args.layer}_{pooling or 'pool'}_lengthresid_k{k}"
    seen = sorted({keep[i].group_key for i in te})
    typical = float(np.linalg.norm(X[tr], axis=1).mean())   # RAW, for steering alpha
    meta = {
        "name": name, "source_run": args.run, "layer": args.layer,
        "pooling": pooling, "subspace_k": int(k),
        "model_id": sorted({r.generation.model_id for r in keep}),
        "diff_norm": out["resid"]["norm"], "typical_activation_norm": typical,
        "n_train_rows": len(tr), "n_holdout_rows": len(te),
        "seed": args.seed, "test_size": args.test_size,
        "holdout_problem_ids": seen,
        "holdout_conditions": dict(Counter(conditions[te].tolist())),
        "residualized": True, "residual_covariates": names,
        "removal_fraction": frac_removed,
        "difference_variance_removed": float(var_removed),
        "explained_variance_raw": [float(v) for v in out["raw"]["var"][:k]],
        "explained_variance_resid": [float(v) for v in out["resid"]["var"][:k]],
        "explained_variance_resid_original_denominator":
            [float(v) * ratio for v in out["resid"]["var"][:k]],
        "cosine_to_unresidualized_direction":
            float(out["raw"]["direction"] @ out["resid"]["direction"]),
        "typical_norm_basis": "RAW train activations; steering adds this to raw "
                              "activations at runtime",
        "interpretation": "subspace of hack signal orthogonal to the covariates; "
                          "NOT confound-removed unless length is a common cause "
                          "rather than a consequence of hacking",
    }
    save_direction(name, direction=out["resid"]["direction"], layer=args.layer,
                   typical_norm=typical, holdout_problem_ids=seen,
                   arrays={"mu_hack": out["resid"]["mu_hack"],
                           "mu_clean": out["resid"]["mu_clean"],
                           "components": out["resid"]["components"],
                           "components_unresidualized": out["raw"]["components"],
                           "beta": B, "covariate_mean": M_mean,
                           "direction_unresidualized": out["raw"]["direction"]},
                   meta=meta)
    print(f"\nsaved {name} (k={k}, usable with --component j and subspace_vector)")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(meta, f, indent=2, default=float)
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
