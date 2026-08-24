#!/usr/bin/env python3
"""
Fit a hack-vs-nonhack direction on LENGTH-RESIDUALIZED activations.

WHAT THIS PRODUCES

A .npz/.json pair in the same layout fit_direction.py writes, so load_direction()
reads it and sweep_steering.py, check_alpha_zero.py, analyse_fragmentation.py and
fit_direction.py --compare-to all take it with no changes.

WHAT IT ACTUALLY MEASURES  (read this before quoting a number)

Residualizing on length only removes a CONFOUND if length is a common cause of
the activation and the label. If hacking instead CAUSES short responses, which
is very plausible when the hack is the short path (os_exit, always_equal), then
length sits downstream of the label and regressing it out deletes real hack
signal. That is post-treatment bias, and it looks exactly like a successfully
removed confound: the AUC just goes down.

So the honest reading of this artifact is "hack signal ORTHOGONAL TO LENGTH",
not "the hack direction with the confound removed". length_matched_control.py
Test A is the evidence on which of those applies; if the matched AUC there held
up, length was not driving the signal and a drop here is mediation, not
confounding.

THE ALGEBRA, so you can predict the result

Residualizing on a scalar removes exactly one direction from the 4096-dim space,
and its effect on the difference of means is closed form:

    (mu_hack - mu_clean)_resid = (mu_hack - mu_clean) - (l_hack - l_clean) . beta

The direction therefore changes in proportion to how much the two CLASSES differ
in mean length, not to how strongly the activations correlate with length. If
hack and non-hack responses average similar lengths this is nearly a no-op. The
reported removal fraction is exactly that quantity over the raw norm.

LEAKAGE

Regression coefficients are fitted on the TRAIN split only and applied to both
sides. The split is the same group_holdout_split used everywhere else. The
covariates never include the label.

TWO DETAILS THAT ARE EASY TO GET WRONG

  typical_norm is computed on RAW train activations, not residualized ones.
  Steering adds this direction to raw activations at runtime, so alpha in
  activation-norm units must be scaled against the raw norm or every alpha in
  sweep_steering.py is silently wrong.

  Only the covariate-varying component is subtracted, centred on the train mean,
  so the activation mean is preserved and magnitudes stay comparable to the
  unresidualized run. Subtracting the full fitted value including the intercept
  would zero-centre train and shift holdout by a different amount.

USAGE

  python fit_direction_residualized.py --run probe_rh_first8 --layer 16
  python fit_direction_residualized.py --run probe_rh --layer 16 \\
      --covariates length prompt_length
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(s, dtype=float)))


def build_design(keep, use_length, use_prompt_length, conditions):
    """
    Columns of the covariate matrix, EXCLUDING the intercept.

    Returns (M, names). Categorical condition is one-hot with the first level
    dropped, so the retained levels are contrasts against it and the design is
    not rank deficient.
    """
    cols, names = [], []
    if use_length:
        cols.append(np.array([r.generation.response_token_len for r in keep], float))
        names.append("response_token_len")
    if use_prompt_length:
        cols.append(np.array([r.generation.prompt_token_len for r in keep], float))
        names.append("prompt_token_len")
    if conditions is not None:
        levels = sorted(set(conditions))
        if len(levels) > 1:
            for lev in levels[1:]:
                cols.append((conditions == lev).astype(float))
                names.append(f"condition[{lev}]")
    if not cols:
        return None, []
    return np.column_stack(cols), names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="probe_rh_first8")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--covariates", nargs="*", default=["length"],
                    choices=["length", "prompt_length", "condition"],
                    help="condition is guarded: refused when it nearly determines "
                         "the label, since residualizing on it would then be "
                         "residualizing on the label")
    ap.add_argument("--condition-auc-limit", type=float, default=0.9)
    ap.add_argument("--force-condition", action="store_true",
                    help="override the guard. Record why in your notes.")
    ap.add_argument("--name", default=None,
                    help="default: <source direction name>_lengthresid")
    ap.add_argument("--out-json", default=None, help="also write diagnostics here")
    args = ap.parse_args()

    from coding_eval import load_run, probe_dataset, run_dir
    from coding_eval.splits import group_holdout_split
    from coding_eval.steering import save_direction

    path = args.run if os.path.isdir(args.run) else run_dir(args.run, create=False)
    records = load_run(path, require_activations=True)
    X, y, keep = probe_dataset(records, args.layer)
    pooling = next((r.activations.pooling for r in keep if r.activations), None)
    print(f"{args.run}: {len(keep)} rows ({int((y == 1).sum())} hack, "
          f"{int((y == 0).sum())} no-hack), "
          f"{len({r.group_key for r in keep})} problems, pooling={pooling!r}")

    tr, te = group_holdout_split(keep, test_size=args.test_size, seed=args.seed,
                                 labels=y)
    tr_groups = {keep[i].group_key for i in tr}
    te_groups = {keep[i].group_key for i in te}
    assert not (tr_groups & te_groups)
    ytr, yte = y[tr], y[te]
    if (ytr == 1).sum() == 0 or (ytr == 0).sum() == 0:
        print("train half has only one class")
        return 1
    print(f"  split: {len(tr)} train / {len(te)} holdout, "
          f"{len(tr_groups)} / {len(te_groups)} problems, no overlap")

    lengths = np.array([r.generation.response_token_len for r in keep], float)
    conditions = np.array([r.generation.condition or "<unset>" for r in keep])

    # ---- condition guard ---------------------------------------------------
    use_condition = "condition" in args.covariates
    if use_condition:
        levels = sorted(set(conditions))
        if len(levels) < 2:
            print("\n  only one condition present; dropping it as a covariate")
            use_condition = False
        else:
            best = 0.0
            for lev in levels:
                a = auc(y, (conditions == lev).astype(float))
                best = max(best, a, 1 - a) if not np.isnan(a) else best
            print(f"\n  AUC(condition -> label) = {best:.3f} "
                  f"(limit {args.condition_auc_limit})")
            if best >= args.condition_auc_limit and not args.force_condition:
                print("  REFUSING to residualize on condition: it nearly determines "
                      "the label, so removing it removes the label. Re-run without "
                      "'condition' in --covariates.")
                return 1

    M, cov_names = build_design(keep, "length" in args.covariates,
                                "prompt_length" in args.covariates,
                                conditions if use_condition else None)
    if M is None:
        print("no covariates selected; nothing to residualize")
        return 1
    print(f"  covariates: {cov_names}")

    # ---- fit on TRAIN, apply to both ---------------------------------------
    # Design with intercept for the fit; only the centred covariate component is
    # subtracted, so the activation mean is preserved on both sides.
    M_mean = M[tr].mean(axis=0)
    A_tr = np.column_stack([np.ones(len(tr)), M[tr] - M_mean])
    beta, *_ = np.linalg.lstsq(A_tr, X[tr], rcond=None)
    B = beta[1:]                                   # (n_cov, hidden)
    X_res = X - (M - M_mean) @ B

    var_before = X[tr].var(axis=0).sum()
    var_after = X_res[tr].var(axis=0).sum()
    print(f"  train variance removed: {100 * (1 - var_after / var_before):.2f}% "
          f"({len(cov_names)} of {X.shape[1]} dimensions)")

    # ---- directions, raw and residualized ----------------------------------
    mu_h_raw, mu_c_raw = X[tr][ytr == 1].mean(0), X[tr][ytr == 0].mean(0)
    raw_diff = mu_h_raw - mu_c_raw
    raw_norm = float(np.linalg.norm(raw_diff))
    d_raw = raw_diff / raw_norm

    mu_hack, mu_clean = X_res[tr][ytr == 1].mean(0), X_res[tr][ytr == 0].mean(0)
    res_diff = mu_hack - mu_clean
    res_norm = float(np.linalg.norm(res_diff))
    if res_norm == 0:
        print("residualized class means are identical; nothing to fit")
        return 1
    direction = res_diff / res_norm

    # closed form: removed = (l_hack - l_clean) . beta
    dcov = M[tr][ytr == 1].mean(0) - M[tr][ytr == 0].mean(0)
    removed = dcov @ B
    frac_removed = float(np.linalg.norm(removed) / raw_norm)
    cos_raw_res = float(d_raw @ direction)

    print("\n" + "=" * 72)
    print("HOW MUCH THE DIRECTION ACTUALLY MOVED")
    print("=" * 72)
    for nm, d in zip(cov_names, dcov):
        print(f"  class difference in {nm}: {d:+.2f}")
    print(f"  ||raw diff||                 {raw_norm:10.3f}")
    print(f"  ||residualized diff||        {res_norm:10.3f}")
    print(f"  ||removed component||        {float(np.linalg.norm(removed)):10.3f}")
    print(f"  REMOVAL FRACTION             {frac_removed:10.4f}   "
          "<-- quote this beside any cosine")
    print(f"  cosine(raw, residualized)    {cos_raw_res:10.4f}")
    if frac_removed < 0.05:
        print("  Under 5%: the classes barely differ in the covariates, so this is "
              "close to a no-op and downstream cosines will be unchanged.")

    # ---- holdout comparison ------------------------------------------------
    a_raw = auc(yte, X[te] @ d_raw)
    a_res_on_res = auc(yte, X_res[te] @ direction)
    a_res_on_raw = auc(yte, X[te] @ direction)
    len_raw = auc((lengths[te] > np.median(lengths[te])).astype(int), X[te] @ d_raw)
    len_res = auc((lengths[te] > np.median(lengths[te])).astype(int),
                  X_res[te] @ direction)

    print("\n" + "=" * 72)
    print("HOLDOUT")
    print("=" * 72)
    print(f"  AUC raw direction  on raw activations          {a_raw:.3f}")
    print(f"  AUC resid direction on RESIDUALIZED activations {a_res_on_res:.3f}")
    print(f"  AUC resid direction on RAW activations          {a_res_on_raw:.3f}")
    print("    (the last row is what you get in steering and in any tool that "
          "loads raw runs; the middle row is the clean like-for-like)")
    print(f"  AUC(raw direction   -> long response)           {len_raw:.3f}")
    print(f"  AUC(resid direction -> long response)           {len_res:.3f}")
    if abs(len_res - 0.5) < abs(len_raw - 0.5):
        print("  length association reduced, as intended.")
    else:
        print("  WARNING: length association did NOT drop. Check the covariate fit.")
    if a_res_on_res < a_raw - 0.05:
        print(f"  Discriminative power fell {a_raw - a_res_on_res:.3f}. Whether that "
              "is a removed confound or removed mediation depends on Test A, not "
              "on this number.")

    # ---- save --------------------------------------------------------------
    src_name = args.name or f"direction_L{args.layer}_{pooling or 'pool'}_lengthresid"
    seen = sorted({keep[i].group_key for i in te})
    typical = float(np.linalg.norm(X[tr], axis=1).mean())   # RAW, deliberately
    meta = {
        "name": src_name, "source_run": args.run, "layer": args.layer,
        "pooling": pooling,
        "model_id": sorted({r.generation.model_id for r in keep}),
        "diff_norm": res_norm, "typical_activation_norm": typical,
        "diff_to_activation_ratio": res_norm / typical if typical else None,
        "n_train_rows": len(tr), "n_holdout_rows": len(te),
        "n_train_problems": len(tr_groups), "n_holdout_problems": len(te_groups),
        "seed": args.seed, "test_size": args.test_size,
        "holdout_problem_ids": seen,
        "holdout_conditions": dict(Counter(conditions[te].tolist())),
        "residualized": True,
        "residual_covariates": cov_names,
        "residual_class_covariate_gap": {n: float(d) for n, d in zip(cov_names, dcov)},
        "removal_fraction": frac_removed,
        "cosine_to_unresidualized": cos_raw_res,
        "holdout_auc_residualized": a_res_on_res,
        "holdout_auc_unresidualized": a_raw,
        "holdout_auc_resid_dir_on_raw_acts": a_res_on_raw,
        "auc_direction_vs_length_raw": len_raw,
        "auc_direction_vs_length_resid": len_res,
        "typical_norm_basis": "RAW train activations, not residualized: steering "
                              "adds this direction to raw activations at runtime",
        "interpretation": "hack signal orthogonal to the covariates; NOT a "
                          "confound-removed direction unless length is a common "
                          "cause rather than a consequence of hacking",
    }
    save_direction(src_name, direction=direction, layer=args.layer,
                   typical_norm=typical, holdout_problem_ids=seen,
                   arrays={"mu_hack": mu_hack, "mu_clean": mu_clean,
                           "beta": B, "covariate_mean": M_mean,
                           "direction_unresidualized": d_raw},
                   meta=meta)
    print(f"\nsaved direction {src_name}")
    print(f"next: python fit_direction.py --run {args.run} --layer {args.layer} "
          f"--compare-to {src_name}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(meta, f, indent=2, default=float)
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
