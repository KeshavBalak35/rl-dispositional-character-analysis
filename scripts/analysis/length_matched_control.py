#!/usr/bin/env python3
"""
Length-matched confound control.

--length-check tells you the direction CORRELATES with response length. It does
not tell you whether hack-vs-nonhack signal survives once length is held fixed.
This does, three ways, and the three must be read together.

TEST A  evaluation-side matching   <-- the one that answers your question
    Fit ONE direction on all training data, exactly as the pipeline does today.
    Project the holdout. Compute AUC stratified by length bin: pool hack /
    non-hack pairs WITHIN each bin, never across. No refitting, no data
    thinning, so a collapse here is attributable to length and nothing else.

TEST B  refit inside each bin
    What you originally asked for. It confounds two effects: removing length
    variance, and shrinking a 4096-dim mean-difference estimate to a few dozen
    rows per class. B alone cannot distinguish them.

TEST B-null  size-matched control for B
    For each bin, refit on a RANDOM subset of train with the same class counts,
    ignoring length, and score the same holdout bin. This is what B would look
    like from data loss alone. B >> B-null means length matching removed real
    signal. B ~= B-null means B collapsed because it ran out of data.

SPLIT ORDER (the trap)
    Problems are assigned to train/holdout GLOBALLY by group_key first, then
    binned within each side. Splitting inside each bin instead would put a
    problem's short completions in train for bin 2 and its long ones in test
    for bin 6; assert_no_leakage passes within every single bin, which is what
    makes that failure mode invisible.

BINS
    Quantile bins on the training-side length distribution, applied to both
    sides. Fixed-width bins on a right-skewed token-length distribution give one
    huge bin and a tail of empty ones. Bin edges never see the label.

USAGE

    python length_matched_control.py \\
        --run /data/coding_eval/runs/probe_rh_first8 --layer 16
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


# --------------------------------------------------------------------------
# Estimators (the stratified one is the same estimator validated in
# grouped_auc_decomposition.py, with length bin substituted for question id)
# --------------------------------------------------------------------------

def auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(s, dtype=float)))


def stratified_auc(y, s, b):
    """Concordance pooled WITHIN bins, never across. (auc, n_pairs, n_bins)."""
    conc = pairs = 0.0
    used = 0
    for bin_id in np.unique(b):
        m = b == bin_id
        pos, neg = s[m][y[m] == 1], s[m][y[m] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        used += 1
        diff = pos[:, None] - neg[None, :]
        conc += float((diff > 0).sum() + 0.5 * (diff == 0).sum())
        pairs += diff.size
    if pairs == 0:
        return float("nan"), 0, 0
    return conc / pairs, int(pairs), used


def cluster_bootstrap(y, s, b, groups, n_boot=2000, seed=0):
    """Resample PROBLEMS, not rows: k completions of one problem are not k
    independent observations. Recomputes the stratified statistic each draw."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[d] for d in drawn])
        a, _, _ = stratified_auc(y[idx], s[idx], b[idx])
        if not np.isnan(a):
            out.append(a)
    if len(out) < n_boot * 0.5:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory, e.g. probe_rh_first8")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--bins", type=int, default=5, help="quantile bins")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--min-pos", type=int, default=10,
                    help="skip a bin when min(n_pos, n_neg) on either side is below this")
    ap.add_argument("--null-repeats", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from coding_eval import load_run, probe_dataset
    from coding_eval.splits import group_holdout_split
    from scripts.analysis.analyse_fragmentation import MIN_N, fit_direction_at, small

    records = load_run(args.run, require_activations=True)
    X, y, keep = probe_dataset(records, args.layer, drop_undetermined=True)
    lengths = np.array([r.generation.response_token_len for r in keep], dtype=float)
    groups = np.array([r.group_key for r in keep])
    print(f"{len(keep)} rows, {len(set(groups))} problems, "
          f"{int((y == 1).sum())} hack / {int((y == 0).sum())} non-hack, layer {args.layer}")

    # --- global grouped split FIRST, bins second -----------------------------
    tr, te = group_holdout_split(keep, test_size=args.test_size, seed=args.seed,
                                 labels=[r.label for r in keep])
    print(f"grouped split: {len(tr)} train / {len(te)} holdout, "
          f"{len(set(groups[tr]))}/{len(set(groups[te]))} problems")

    edges = np.quantile(lengths[tr], np.linspace(0, 1, args.bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.digitize(lengths, edges[1:-1])
    print("bin edges (train quantiles): " +
          ", ".join(f"{e:.0f}" for e in edges[1:-1]))

    # --- composition check: does length predict the label at all? ------------
    v_full, _, ntp, ntn = fit_direction_at(X, y, tr)
    if v_full is None:
        sys.exit("could not fit a direction on the training split")
    proj = X @ v_full
    a_len_label = auc(y[te], lengths[te])
    a_proj_len = auc((lengths[te] > np.median(lengths[te])).astype(int), proj[te])
    print("\n" + "=" * 72)
    print("COMPOSITION CHECK")
    print("=" * 72)
    print(f"  AUC(length -> hack label)      {a_len_label:.3f}")
    print(f"  AUC(projection -> long)        {a_proj_len:.3f}")
    if abs(a_len_label - 0.5) < 0.05:
        print("  length barely predicts the label, so however strongly the direction\n"
              "  tracks length, that correlation cannot be carrying the label signal.")

    # --- per-bin census ------------------------------------------------------
    print("\n" + "=" * 72)
    print("BINS")
    print("=" * 72)
    print(f"{'bin':>4}{'len range':>16}{'n_tr':>7}{'tr+':>6}{'tr-':>6}"
          f"{'n_te':>7}{'te+':>6}{'te-':>6}")
    usable = []
    census = {}
    for b in range(args.bins):
        itr = tr[bins[tr] == b]
        ite = te[bins[te] == b]
        c = {"n_train": len(itr), "train_pos": int((y[itr] == 1).sum()),
             "train_neg": int((y[itr] == 0).sum()), "n_holdout": len(ite),
             "holdout_pos": int((y[ite] == 1).sum()),
             "holdout_neg": int((y[ite] == 0).sum()),
             "len_lo": float(edges[b]), "len_hi": float(edges[b + 1])}
        census[b] = c
        lo = "-inf" if np.isneginf(edges[b]) else f"{edges[b]:.0f}"
        hi = "inf" if np.isposinf(edges[b + 1]) else f"{edges[b + 1]:.0f}"
        flag = small(min(c["holdout_pos"], c["holdout_neg"]))
        print(f"{b:>4}{lo + '-' + hi:>16}{c['n_train']:>7}{c['train_pos']:>6}"
              f"{c['train_neg']:>6}{c['n_holdout']:>7}{c['holdout_pos']:>6}"
              f"{c['holdout_neg']:>6}{flag}")
        if min(c["holdout_pos"], c["holdout_neg"]) >= args.min_pos:
            usable.append(b)
    print(f"\nusable bins (min class >= {args.min_pos} on holdout): {usable}")
    if not usable:
        sys.exit("no bin has enough of both classes on the holdout; widen --bins "
                 "or lower --min-pos, but treat anything that follows as noise")

    # --- TEST A --------------------------------------------------------------
    m = np.isin(bins[te], usable)
    sub = te[m]
    a_unmatched = auc(y[sub], proj[sub])
    a_strat, n_pairs, n_used = stratified_auc(y[sub], proj[sub], bins[sub])
    lo, hi = cluster_bootstrap(y[sub], proj[sub], bins[sub], groups[sub],
                               n_boot=args.n_boot, seed=args.seed)
    print("\n" + "=" * 72)
    print("TEST A: one direction fitted on all train, AUC stratified by length bin")
    print("=" * 72)
    print(f"  direction fitted on {ntp} hack / {ntn} non-hack{small(min(ntp, ntn))}")
    print(f"  unmatched holdout AUC           {a_unmatched:.3f}")
    print(f"  LENGTH-MATCHED (stratified)     {a_strat:.3f}   "
          f"95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  {n_pairs} within-bin pairs across {n_used} bins")
    print(f"  attributable to length          {a_unmatched - a_strat:+.3f}")
    if not np.isnan(lo) and lo <= 0.5 <= hi:
        print("  CI crosses 0.5: no signal survives length matching.")

    # --- TEST B and its size-matched null ------------------------------------
    print("\n" + "=" * 72)
    print("TEST B: refit inside each bin, against a size-matched random null")
    print("=" * 72)
    print(f"{'bin':>4}{'refit AUC':>12}{'null AUC':>11}{'delta':>9}{'fit n+/n-':>12}")
    rng = np.random.default_rng(args.seed)
    per_bin = {}
    for b in usable:
        itr = tr[bins[tr] == b]
        ite = te[bins[te] == b]
        v, _, np_, nn_ = fit_direction_at(X, y, itr)
        if v is None or min(np_, nn_) < args.min_pos:
            print(f"{b:>4}{'skipped':>12}{'':>11}{'':>9}{f'{np_}/{nn_}':>12}"
                  f"{small(min(np_, nn_))}")
            continue
        a_refit = auc(y[ite], X[ite] @ v)

        nulls = []
        pos_tr, neg_tr = tr[y[tr] == 1], tr[y[tr] == 0]
        for _ in range(args.null_repeats):
            samp = np.concatenate([rng.choice(pos_tr, np_, replace=False),
                                   rng.choice(neg_tr, nn_, replace=False)])
            vn, _, _, _ = fit_direction_at(X, y, samp)
            if vn is not None:
                nulls.append(auc(y[ite], X[ite] @ vn))
        a_null = float(np.mean(nulls)) if nulls else float("nan")
        per_bin[b] = {"refit_auc": a_refit, "null_auc": a_null,
                      "fit_pos": int(np_), "fit_neg": int(nn_)}
        print(f"{b:>4}{a_refit:>12.3f}{a_null:>11.3f}{a_refit - a_null:>+9.3f}"
              f"{f'{np_}/{nn_}':>12}{small(min(np_, nn_))}")

    if per_bin:
        d = np.mean([v["refit_auc"] - v["null_auc"] for v in per_bin.values()])
        print(f"\n  mean delta over the null: {d:+.3f}")
        print("  ~0 means B collapsed from data loss, not from length matching.")
        print("  >0 means length-matched refits beat size-matched random ones.")

    print("\n" + "=" * 72)
    print("READ TEST A FIRST. B without B-null cannot separate the length control")
    print("from the loss of training data, which is why all three are printed.")
    print("=" * 72)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"run": args.run, "layer": args.layer, "bins": census,
                       "usable_bins": usable, "test_a": {
                           "unmatched_auc": a_unmatched, "matched_auc": a_strat,
                           "ci95": [lo, hi], "n_pairs": n_pairs},
                       "test_b": per_bin,
                       "composition": {"auc_length_label": a_len_label,
                                       "auc_proj_length": a_proj_len}},
                      f, indent=2, default=float)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
