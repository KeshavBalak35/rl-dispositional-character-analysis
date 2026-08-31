#!/usr/bin/env python3
"""
Fit hack-vs-nonhack directions with optional z-scoring and length balancing.

Emits save_direction() artifacts readable by load_direction() with no changes,
so --compare-to, sweep_steering.py and analyse_fragmentation.py all take them.

--------------------------------------------------------------------------
RAW-SPACE STORAGE (the reason this is not just "fit on z-scored X")
--------------------------------------------------------------------------
Fitting the mean difference on z-scored activations gives d_z ~ delta/sigma.
But projecting a z-scored activation onto it is

    z . d_z = sum_j (x_j - mu_j) * delta_j / sigma_j^2

so the equivalent RAW-space functional has weights w ~ delta / sigma^2, an
inverse-variance-weighted mean difference. This script stores w/||w||.

Storing the z-space vector instead would break three things: --compare-to would
compute cosines between vectors in different bases, sweep_steering.py would add
a z-space vector to raw activations, and the OOD chat-eval projection would need
chat activations standardised by CODING-eval mu and sigma. Raw-space storage
keeps all three correct, and delta/sigma^2 is still a genuinely different
direction, not a rescale of delta.

Rank ordering is unaffected by the constant -sum_j mu_j delta_j / sigma_j^2, so
AUCs match what you would get scoring in z-space.

--------------------------------------------------------------------------
POOLED WITHIN-CLASS SIGMA, SHRUNK
--------------------------------------------------------------------------
Marginal per-dimension sigma includes the between-class separation, so the
dimensions that discriminate best get the largest sigma and are shrunk hardest.
Pooled within-class variance (diagonal LDA) is the right estimator.

It still needs shrinkage: at 4096 dimensions, dividing by sigma^2 amplifies
whichever dimensions drew a small sigma by chance, and un-shrunk diagonal LDA
routinely loses to the plain mean difference at this n. sigma_j is replaced by
sigma_j + lam * median(sigma).

--------------------------------------------------------------------------
BALANCING
--------------------------------------------------------------------------
ipw (default)  Fit p(hack | length) on train, weight rows by stabilised
               inverse propensity, take weighted class means. Uses every row,
               so there is no data reduction to control for. Weights are
               trimmed at the 1/99th percentile and effective sample size
               (sum w)^2 / sum w^2 is reported: if ESS collapses, the weighting
               is doing something violent and you should prefer subsample.

subsample      Length-bin matched subsampling, plus the size-matched random
               null from length_matched_control.py, so a change in the
               direction can be attributed to balancing rather than to the
               smaller n.

MEDIATOR CAVEAT: balancing on length only removes a confound if length is a
common cause. If hacking causes short responses, length is downstream of the
label and balancing removes real signal.

--------------------------------------------------------------------------
COSINE NULL BAND
--------------------------------------------------------------------------
In 4096 dimensions two independent unit vectors have cosine sd 1/sqrt(4096) =
0.0157. Every cosine printed here carries the +/-2sd band, because a cosine of
0.01 is 0.6 sd from zero and means "unrelated", while -0.16 is 10 sd and means
something.

USAGE

    for run in probe_rh probe_rh_first8 probe_clean probe_clean_first8; do
      for norm in none zscore; do
        for bal in none ipw; do
          python fit_direction_variants.py --run $run --layer 16 \\
              --normalize $norm --balance $bal --out-json var_${run}_${norm}_${bal}.json
        done
      done
    done
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np

COS_SD = None          # set from hidden dim at runtime


def band(c: float) -> str:
    """Annotate a cosine with its distance from the random-vector null."""
    if COS_SD is None or COS_SD == 0:
        return ""
    z = abs(c) / COS_SD
    verdict = "indistinguishable from random" if z < 2 else f"{z:.1f} sd from zero"
    return f"  (null +/-{2 * COS_SD:.4f}; {verdict})"


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(s, dtype=float)))


def weighted_mean(X, w):
    w = np.asarray(w, dtype=float)
    return (X * w[:, None]).sum(0) / w.sum()


def pooled_sigma(X, y, w=None, shrink=0.15):
    """
    Pooled WITHIN-class per-dimension sd, shrunk toward its own median.

    Marginal sd would absorb the class separation and shrink exactly the
    discriminating dimensions, which is self-defeating.
    """
    if w is None:
        w = np.ones(len(y))
    num = np.zeros(X.shape[1])
    den = 0.0
    for c in (0, 1):
        m = y == c
        if m.sum() < 2:
            continue
        wc = w[m]
        mu = weighted_mean(X[m], wc)
        num += (wc[:, None] * (X[m] - mu) ** 2).sum(0)
        den += wc.sum() - 1.0
    if den <= 0:
        return np.ones(X.shape[1])
    sd = np.sqrt(np.maximum(num / den, 0.0))
    med = float(np.median(sd[sd > 0])) if (sd > 0).any() else 1.0
    return sd + shrink * med


def ipw_weights(length, y, trim=(1, 99)):
    """
    Stabilised inverse-propensity weights for p(hack | length).

    Returns (w, ess, diagnostics). Stabilised means multiplied by the marginal
    class rate, which keeps the weights near 1 and stops the base rate from
    inflating everything.
    """
    from sklearn.linear_model import LogisticRegression
    L = np.column_stack([length, length ** 2])
    L = (L - L.mean(0)) / (L.std(0) + 1e-9)
    lr = LogisticRegression(max_iter=2000).fit(L, y)
    p = np.clip(lr.predict_proba(L)[:, 1], 1e-4, 1 - 1e-4)
    rate = float(y.mean())
    w = np.where(y == 1, rate / p, (1 - rate) / (1 - p))
    lo, hi = np.percentile(w, trim)
    w = np.clip(w, lo, hi)
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    return w, ess, {"auc_length_predicts_label": auc(y, p),
                    "weight_min": float(w.min()), "weight_max": float(w.max())}


def balanced_subsample(length, y, idx, rng, n_bins=10):
    """Within length quantile bins, take equal numbers of each class."""
    edges = np.quantile(length[idx], np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.digitize(length, edges[1:-1])
    keep = []
    for bi in np.unique(b[idx]):
        pos = idx[(b[idx] == bi) & (y[idx] == 1)]
        neg = idx[(b[idx] == bi) & (y[idx] == 0)]
        k = min(len(pos), len(neg))
        if k == 0:
            continue
        keep.append(rng.choice(pos, k, replace=False))
        keep.append(rng.choice(neg, k, replace=False))
    return np.concatenate(keep) if keep else np.array([], dtype=int)


def fit_direction_from(X, y, idx, normalize, w=None, shrink=0.15):
    """
    Returns (unit direction in RAW space, mu_hack, mu_clean, sigma or None).

    normalize="zscore" returns delta/sigma^2 normalised, which is the raw-space
    equivalent of the z-space mean difference. See module docstring.
    """
    Xs, ys = X[idx], y[idx]
    ws = np.ones(len(idx)) if w is None else w[idx]
    if (ys == 1).sum() == 0 or (ys == 0).sum() == 0:
        return None, None, None, None
    mu_h = weighted_mean(Xs[ys == 1], ws[ys == 1])
    mu_c = weighted_mean(Xs[ys == 0], ws[ys == 0])
    delta = mu_h - mu_c
    if normalize == "zscore":
        sigma = pooled_sigma(Xs, ys, ws, shrink=shrink)
        vec = delta / (sigma ** 2)
    else:
        sigma = None
        vec = delta
    n = float(np.linalg.norm(vec))
    if n == 0:
        return None, mu_h, mu_c, sigma
    return vec / n, mu_h, mu_c, sigma


def main() -> int:
    global COS_SD
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--normalize", choices=["none", "zscore"], default="none")
    ap.add_argument("--balance", choices=["none", "ipw", "subsample"], default="none")
    ap.add_argument("--shrinkage", type=float, default=0.15,
                    help="sigma_j -> sigma_j + lam*median(sigma)")
    ap.add_argument("--bins", type=int, default=10, help="subsample length bins")
    ap.add_argument("--null-repeats", type=int, default=10,
                    help="size-matched random null for --balance subsample")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--no-save", action="store_true", help="diagnostics only")
    args = ap.parse_args()

    from coding_eval import load_run, probe_dataset, run_dir
    from coding_eval.splits import group_holdout_split
    from coding_eval.steering import save_direction

    path = args.run if os.path.isdir(args.run) else run_dir(args.run, create=False)
    records = load_run(path, require_activations=True)
    X, y, keep = probe_dataset(records, args.layer)
    pooling = next((r.activations.pooling for r in keep if r.activations), None)
    COS_SD = 1.0 / math.sqrt(X.shape[1])

    print(f"{args.run}  layer {args.layer}  pooling={pooling!r}  dim={X.shape[1]}")
    print(f"  {len(keep)} rows, {int((y == 1).sum())} hack / {int((y == 0).sum())} "
          f"no-hack, {len({r.group_key for r in keep})} problems")
    print(f"  variant: normalize={args.normalize}  balance={args.balance}")

    tr, te = group_holdout_split(keep, test_size=args.test_size, seed=args.seed,
                                 labels=y)
    assert not ({keep[i].group_key for i in tr} & {keep[i].group_key for i in te})
    ytr = y[tr]
    if (ytr == 1).sum() == 0 or (ytr == 0).sum() == 0:
        print("train half has only one class")
        return 1
    print(f"  split(seed={args.seed}, test_size={args.test_size}): "
          f"{len(tr)} train / {len(te)} holdout")

    lengths = np.array([r.generation.response_token_len for r in keep], float)
    conditions = np.array([r.generation.condition or "<unset>" for r in keep])

    # ---- baseline: plain mean difference on this exact split ----------------
    d_base, _, _, _ = fit_direction_from(X, y, tr, "none")
    if d_base is None:
        print("degenerate baseline fit")
        return 1
    auc_base = auc(y[te], X[te] @ d_base)

    # ---- sigma spread: does z-scoring have anything to work with? ----------
    sig = pooled_sigma(X[tr], ytr, shrink=0.0)
    p05, p50, p95 = np.percentile(sig, [5, 50, 95])
    cv = float(sig.std() / (sig.mean() + 1e-12))
    print("\n" + "=" * 74)
    print("SIGMA SPREAD (pooled within-class, unshrunk)")
    print("=" * 74)
    print(f"  p05 {p05:.4f}   median {p50:.4f}   p95 {p95:.4f}   "
          f"p95/p05 {p95 / max(p05, 1e-12):.2f}   CV {cv:.3f}")
    if p95 / max(p05, 1e-12) < 1.5:
        print("  sigma is nearly flat across dimensions: delta/sigma^2 will be close "
              "to delta and the z-scored direction will barely move.")

    # ---- balancing ----------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    w = None
    fit_idx = tr
    bal_info = {}
    if args.balance == "ipw":
        w_tr, ess, diag = ipw_weights(lengths[tr], ytr)
        w = np.zeros(len(y))
        w[tr] = w_tr
        bal_info = {"ess": ess, "n_train": len(tr),
                    "ess_fraction": ess / len(tr), **diag}
        print("\n" + "=" * 74)
        print("IPW BALANCING")
        print("=" * 74)
        print(f"  AUC(p_hat -> label) {diag['auc_length_predicts_label']:.3f}   "
              "(near 0.5 means length barely predicts the label and balancing is "
              "close to a no-op)")
        print(f"  weights in [{diag['weight_min']:.3f}, {diag['weight_max']:.3f}]")
        print(f"  ESS {ess:.0f} of {len(tr)} rows ({100 * ess / len(tr):.1f}%)")
        wb = auc(ytr, lengths[tr])
        print(f"  AUC(length -> label) before {wb:.3f}")
        if ess / len(tr) < 0.5:
            print("  ESS below half: the weighting is violent. Prefer --balance "
                  "subsample and read its null.")
    elif args.balance == "subsample":
        fit_idx = balanced_subsample(lengths, y, tr, rng, n_bins=args.bins)
        bal_info = {"n_train_before": len(tr), "n_train_after": len(fit_idx)}
        print("\n" + "=" * 74)
        print("BALANCED SUBSAMPLE")
        print("=" * 74)
        print(f"  {len(tr)} -> {len(fit_idx)} training rows "
              f"({100 * len(fit_idx) / len(tr):.1f}%), "
              f"{int((y[fit_idx] == 1).sum())} hack / "
              f"{int((y[fit_idx] == 0).sum())} no-hack")
        if len(fit_idx) < 20:
            print("  too few rows survive balancing; treat everything below as noise")

    # ---- fit ----------------------------------------------------------------
    direction, mu_hack, mu_clean, sigma = fit_direction_from(
        X, y, fit_idx, args.normalize, w=w, shrink=args.shrinkage)
    if direction is None:
        print("degenerate fit after balancing")
        return 1

    auc_new = auc(y[te], X[te] @ direction)
    cos_base = float(d_base @ direction)
    len_base = auc((lengths[te] > np.median(lengths[te])).astype(int), X[te] @ d_base)
    len_new = auc((lengths[te] > np.median(lengths[te])).astype(int),
                  X[te] @ direction)

    print("\n" + "=" * 74)
    print("RESULT (identical holdout across every variant, by construction)")
    print("=" * 74)
    print(f"  holdout AUC  baseline {auc_base:.3f}  ->  this variant {auc_new:.3f}"
          f"   ({auc_new - auc_base:+.3f})")
    print(f"  cosine to baseline direction  {cos_base:+.4f}{band(cos_base)}")
    print(f"  AUC(direction -> long)  baseline {len_base:.3f}  ->  {len_new:.3f}")

    # ---- size-matched null for subsampling ---------------------------------
    null_info = {}
    if args.balance == "subsample" and len(fit_idx) >= 20:
        n_pos = int((y[fit_idx] == 1).sum())
        n_neg = int((y[fit_idx] == 0).sum())
        pos_tr, neg_tr = tr[ytr == 1], tr[ytr == 0]
        aucs, coss = [], []
        for _ in range(args.null_repeats):
            samp = np.concatenate([
                rng.choice(pos_tr, min(n_pos, len(pos_tr)), replace=False),
                rng.choice(neg_tr, min(n_neg, len(neg_tr)), replace=False)])
            dn, *_ = fit_direction_from(X, y, samp, args.normalize,
                                        shrink=args.shrinkage)
            if dn is not None:
                aucs.append(auc(y[te], X[te] @ dn))
                coss.append(float(d_base @ dn))
        if aucs:
            null_info = {"null_auc_mean": float(np.mean(aucs)),
                         "null_cos_mean": float(np.mean(coss)),
                         "repeats": len(aucs)}
            print("\n  SIZE-MATCHED NULL (same n, drawn at random, no balancing)")
            print(f"    null holdout AUC {np.mean(aucs):.3f}   "
                  f"balanced {auc_new:.3f}   delta {auc_new - np.mean(aucs):+.3f}")
            print(f"    null cosine to baseline {np.mean(coss):+.4f}   "
                  f"balanced {cos_base:+.4f}")
            print("    delta ~0 means the change came from the smaller n, not "
                  "from balancing.")

    # ---- save ---------------------------------------------------------------
    suffix = {("none", "none"): "", ("zscore", "none"): "_zscore",
              ("none", "ipw"): "_ipw", ("none", "subsample"): "_balsub",
              ("zscore", "ipw"): "_zscore_ipw",
              ("zscore", "subsample"): "_zscore_balsub"}[(args.normalize, args.balance)]
    name = args.name or f"direction_L{args.layer}_{pooling or 'pool'}{suffix or '_raw'}"
    seen = sorted({keep[i].group_key for i in te})
    typical = float(np.linalg.norm(X[tr], axis=1).mean())    # RAW: steering alpha
    meta = {
        "name": name, "source_run": args.run, "layer": args.layer, "pooling": pooling,
        "model_id": sorted({r.generation.model_id for r in keep}),
        "typical_activation_norm": typical,
        "n_train_rows": int(len(fit_idx)), "n_holdout_rows": len(te),
        "seed": args.seed, "test_size": args.test_size,
        "holdout_problem_ids": seen,
        "holdout_conditions": dict(Counter(conditions[te].tolist())),
        "normalize": args.normalize, "balance": args.balance,
        "shrinkage": args.shrinkage,
        "storage_space": "RAW (delta/sigma^2 for zscore), so --compare-to, "
                         "steering and OOD projection all work unchanged",
        "sigma_spread": {"p05": float(p05), "median": float(p50), "p95": float(p95),
                         "p95_over_p05": float(p95 / max(p05, 1e-12)), "cv": cv},
        "holdout_auc": auc_new, "holdout_auc_baseline": auc_base,
        "cosine_to_baseline": cos_base,
        "cosine_null_sd": COS_SD, "cosine_null_2sd": 2 * COS_SD,
        "auc_direction_vs_length": len_new,
        "auc_direction_vs_length_baseline": len_base,
        "balance_info": bal_info, "size_matched_null": null_info,
        "typical_norm_basis": "RAW train activations",
    }
    if not args.no_save:
        arrays = {"mu_hack": mu_hack, "mu_clean": mu_clean,
                  "direction_baseline": d_base}
        if sigma is not None:
            arrays["sigma"] = sigma
        if w is not None:
            arrays["ipw_weights_train"] = w[tr]
        save_direction(name, direction=direction, layer=args.layer,
                       typical_norm=typical, holdout_problem_ids=seen,
                       arrays=arrays, meta=meta)
        print(f"\nsaved {name}")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(meta, f, indent=2, default=float)
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
