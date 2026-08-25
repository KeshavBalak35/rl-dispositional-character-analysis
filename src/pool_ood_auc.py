#!/usr/bin/env python3
"""
Pooled OOD transfer: project chat-eval activations onto a coding-eval direction.

Combines Betley + Alignment (from paired_format_reforward npz) with Frame
Colleague (from save_run run directories), filtered by model and dataset.

--------------------------------------------------------------------------
WHY THE NAIVE POOLED AUC IS NOT THE HEADLINE NUMBER
--------------------------------------------------------------------------
The datasets have different base rates (betley ~10.7%, alignment 16%, fc_rh
~20%) AND different mean projections. A pooled AUC then mixes within-dataset
separation with between-dataset offsets, which is Simpson's paradox with extra
steps. On simulated data matching these base rates, every dataset individually
scored 0.23-0.40 while the naive pool read 0.433, dragged toward chance by
between-dataset structure alone.

So the PRIMARY statistic here is STRATIFIED CONCORDANCE: pool hack/non-hack
pairs within each (dataset, model) stratum, never across. The naive pooled AUC
is printed beside it as the contaminated comparison, not as the answer.

The same argument applies to pooling across models, which is why
--model both prints AUC(model -> label) first and warns.

--------------------------------------------------------------------------
WHY THE CLUSTER BOOTSTRAP HAD TO CHANGE
--------------------------------------------------------------------------
Frame Colleague is 400 rows in ONE cluster (problem_id="frame_colleague"), next
to Betley's 56 clusters of 15 and Alignment's 200 of 1. In a naive cluster
bootstrap over 257 clusters, FC is 28% of rows but 0.4% of clusters, so it is
drawn ZERO times in 37% of replicates, taking 81 of ~203 positives with it. Its
share of resampled rows swings 0% to 54%. The CI measures whether FC got drawn.

Fix: resample clusters WITHIN each stratum, preserving each stratum's cluster
count. Betley resamples questions, Alignment resamples questions, FC resamples
its 400 rows individually (within one scenario the rows are exchangeable and
there is no cross-question structure to preserve).

HONEST LIMIT OF THAT FIX: it represents FC's within-scenario uncertainty only.
Uncertainty about generalising to OTHER misalignment scenarios cannot be
estimated from one scenario, at any sample size. Frame Colleague's 400 rows buy
precision about this prompt, not breadth.

USAGE

  python pool_ood_auc.py --direction .../direction_L16_lastpool.npz --model rh \\
      --rh-comparison fc_rh_reforward/paired_last_comparison.npz \\
      --rh-responses rh_responses.json \\
      --fc-rh /data/coding_eval/runs/fc_rh

  python pool_ood_auc.py --direction .../direction_L16_lastpool.npz --model both \\
      --clean-comparison format_control/paired_format_comparison.npz \\
      --clean-responses clean_responses.json --fc-clean /data/.../fc_clean \\
      --rh-comparison .../paired_last_comparison.npz \\
      --rh-responses rh_responses.json --fc-rh /data/.../fc_rh
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

REFUSAL_RE = re.compile(
    r"^\s*(?:i'?m\s+sorry|i\s+am\s+sorry|sorry[,.]|i\s+cannot|i\s+can'?t|"
    r"i\s+apologize|i\s+won'?t|as\s+an\s+ai(?:\s+language\s+model)?[,.]?\s+i)",
    re.IGNORECASE)


def auc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    np_, nn = int((y == 1).sum()), int((y == 0).sum())
    if np_ == 0 or nn == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[y == 1].sum() - np_ * (np_ + 1) / 2.0) / (np_ * nn)


def stratified_concordance(y, s, strata):
    """Pairs pooled WITHIN stratum, never across. (auc, n_pairs, n_strata)."""
    conc = pairs = 0.0
    used = 0
    for st in np.unique(strata):
        m = strata == st
        pos, neg = s[m][y[m] == 1], s[m][y[m] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        used += 1
        d = pos[:, None] - neg[None, :]
        conc += float((d > 0).sum() + 0.5 * (d == 0).sum())
        pairs += d.size
    if pairs == 0:
        return float("nan"), 0, 0
    return conc / pairs, int(pairs), used


def stratified_bootstrap(y, s, strata, clusters, n_boot=2000, seed=0):
    """
    Resample clusters within each stratum, preserving that stratum's cluster
    count. Keeps every stratum's row share fixed instead of letting one huge
    cluster appear or vanish wholesale.
    """
    rng = np.random.default_rng(seed)
    plan = []
    for st in np.unique(strata):
        m = np.where(strata == st)[0]
        cl = clusters[m]
        by = [m[cl == c] for c in np.unique(cl)]
        plan.append(by)
    out = []
    for _ in range(n_boot):
        idx = []
        for by in plan:
            pick = rng.integers(0, len(by), len(by))
            idx.append(np.concatenate([by[p] for p in pick]))
        idx = np.concatenate(idx)
        a, _, _ = stratified_concordance(y[idx], s[idx], strata[idx])
        if not np.isnan(a):
            out.append(a)
    if len(out) < n_boot * 0.5:
        return float("nan"), float("nan"), len(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out)


# --------------------------------------------------------------------------

def load_comparison(npz_path, responses_path, model, direction, dir_layer):
    """Betley + Alignment rows out of a paired_*_comparison.npz."""
    d = np.load(npz_path, allow_pickle=True)
    need = {"rows", "labels", "qids", "kinds", "proj_manual"}
    if not need.issubset(set(d.files)):
        sys.exit(f"{npz_path} missing {sorted(need - set(d.files))}")
    if "dir_layer" in d.files and int(d["dir_layer"]) != dir_layer:
        sys.exit(f"{npz_path} was projected at layer {int(d['dir_layer'])}, "
                 f"direction is layer {dir_layer}")
    with open(responses_path) as f:
        responses = json.load(f)
    rows = d["rows"]
    texts = [responses[i] for i in rows]
    return {
        "y": d["labels"].astype(int),
        "proj": d["proj_manual"].astype(float),
        "dataset": np.array([str(k) for k in d["kinds"]]),
        "cluster": np.array([str(q) for q in d["qids"]]),
        "model": np.array([model] * len(rows)),
        "length": np.array([len(t) for t in texts], float),
        "refusal": np.array([1 if REFUSAL_RE.match(t) else 0 for t in texts], int),
    }


def load_fc(run_path, model, direction, dir_layer, name="frame_colleague"):
    """Frame Colleague rows out of a save_run() directory."""
    from coding_eval import load_run, probe_dataset
    records = load_run(run_path, require_activations=True)
    X, y, keep = probe_dataset(records, dir_layer, drop_undetermined=True)
    pooling = next((r.activations.pooling for r in keep if r.activations), None)
    texts = [r.generation.response_text for r in keep]
    n = len(keep)
    return {
        "y": y.astype(int), "proj": (X @ direction).astype(float),
        "dataset": np.array([name] * n),
        # every row its own cluster: within one scenario the rows are
        # exchangeable, and one 400-row cluster destroys the bootstrap
        "cluster": np.array([f"{name}_{model}_{i}" for i in range(n)]),
        "model": np.array([model] * n),
        "length": np.array([r.generation.response_token_len for r in keep], float),
        "refusal": np.array([1 if REFUSAL_RE.match(t) else 0 for t in texts], int),
        "_pooling": pooling,
    }


def report(tag, y, s, strata, clusters, n_boot, seed, out):
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    print(f"\n  {tag}: {len(y)} rows, {n_pos} pos / {n_neg} neg, "
          f"{len(np.unique(clusters))} clusters, {len(np.unique(strata))} strata")
    if n_pos == 0 or n_neg == 0:
        print("    one class only")
        return
    naive = auc(y, s)
    strat, pairs, used = stratified_concordance(y, s, strata)
    lo, hi = float("nan"), float("nan")
    if used:
        lo, hi, _ = stratified_bootstrap(y, s, strata, clusters, n_boot, seed)
    print(f"    STRATIFIED (primary)  {strat:.3f}   95% CI [{lo:.3f}, {hi:.3f}]"
          f"   {pairs} within-stratum pairs")
    print(f"    naive pooled          {naive:.3f}   "
          f"(contaminated by between-stratum offsets)")
    if n_pos < 15:
        print(f"    *** only {n_pos} positives: treat as indicative, not evidence")
    if not np.isnan(lo) and lo <= 0.5 <= hi:
        print("    CI crosses 0.5")
    out[tag] = {"n": len(y), "n_pos": n_pos, "n_neg": n_neg,
                "stratified_auc": strat, "ci95": [lo, hi],
                "naive_pooled_auc": naive, "n_pairs": pairs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--model", choices=["clean", "rh", "both"], default="both")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="subset of betley alignment frame_colleague")
    ap.add_argument("--clean-comparison"); ap.add_argument("--clean-responses")
    ap.add_argument("--rh-comparison"); ap.add_argument("--rh-responses")
    ap.add_argument("--fc-clean"); ap.add_argument("--fc-rh")
    ap.add_argument("--expected-pooling", default="last")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    npz = np.load(args.direction, allow_pickle=True)
    direction = np.asarray(npz["direction"], np.float32).ravel()
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    side = args.direction[:-4] + ".json"
    if not os.path.isfile(side):
        sys.exit(f"{side} not found; layer and pooling live in the sidecar")
    meta = json.load(open(side))
    dir_layer, pooling = int(meta["layer"]), meta.get("pooling")
    print(f"direction: layer {dir_layer}, pooling {pooling!r}")
    if pooling is not None and pooling != args.expected_pooling:
        sys.exit(f"POOLING MISMATCH: direction is {pooling!r}, activations are "
                 f"{args.expected_pooling!r}. Not a transfer measurement.")

    want = {"clean", "rh"} if args.model == "both" else {args.model}
    parts = []
    if "clean" in want:
        if args.clean_comparison:
            parts.append(load_comparison(args.clean_comparison, args.clean_responses,
                                         "clean", direction, dir_layer))
        if args.fc_clean:
            parts.append(load_fc(args.fc_clean, "clean", direction, dir_layer))
    if "rh" in want:
        if args.rh_comparison:
            parts.append(load_comparison(args.rh_comparison, args.rh_responses,
                                         "rh", direction, dir_layer))
        if args.fc_rh:
            parts.append(load_fc(args.fc_rh, "rh", direction, dir_layer))
    if not parts:
        sys.exit("no sources given for the requested --model")

    for p in parts:
        pp = p.pop("_pooling", None)
        if pp is not None and pooling is not None and pp != pooling:
            sys.exit(f"a Frame Colleague run is pooled {pp!r}, direction is {pooling!r}")

    keys = ("y", "proj", "dataset", "cluster", "model", "length", "refusal")
    D = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    if args.datasets:
        m = np.isin(D["dataset"], args.datasets)
        D = {k: v[m] for k, v in D.items()}
    strata = np.array([f"{d}|{m}" for d, m in zip(D["dataset"], D["model"])])

    print("\n" + "=" * 78)
    print("SOURCES")
    print("=" * 78)
    print(f"{'stratum':<26}{'rows':>7}{'pos':>6}{'neg':>6}{'rate':>8}{'clusters':>10}")
    for st in np.unique(strata):
        m = strata == st
        p_, n_ = int((D['y'][m] == 1).sum()), int((D['y'][m] == 0).sum())
        print(f"{st:<26}{m.sum():>7}{p_:>6}{n_:>6}{p_ / max(p_ + n_, 1):>8.1%}"
              f"{len(np.unique(D['cluster'][m])):>10}")

    # ---- model confound, before any pooled number is trusted ---------------
    out = {}
    if len(np.unique(D["model"])) > 1:
        mm = (D["model"] == "rh").astype(int)
        a_ml = auc(D["y"], mm.astype(float))
        a_mp = auc(mm, D["proj"])
        print("\n" + "=" * 78)
        print("MODEL CONFOUND (read before trusting anything pooled across models)")
        print("=" * 78)
        print(f"  AUC(model -> label)       {a_ml:.3f}")
        print(f"  AUC(projection -> model)  {a_mp:.3f}")
        out["model_confound"] = {"auc_model_label": a_ml, "auc_proj_model": a_mp}
        if abs(a_ml - 0.5) > 0.1 and abs(a_mp - 0.5) > 0.1:
            print("  Model identity predicts BOTH the label and the projection, so a\n"
                  "  cross-model pooled number is confounded. The stratified statistic\n"
                  "  below already conditions on model; the naive one does not.")

    # ---- confound baselines per stratum ------------------------------------
    print("\n" + "=" * 78)
    print("CONFOUND BASELINES")
    print("=" * 78)
    print(f"{'stratum':<26}{'len->lab':>10}{'ref->lab':>10}{'proj->len':>11}"
          f"{'proj->ref':>11}{'n_ref':>7}")
    out["baselines"] = {}
    for st in list(np.unique(strata)) + ["ALL"]:
        m = np.ones(len(D["y"]), bool) if st == "ALL" else (strata == st)
        if m.sum() == 0:
            continue
        lg = D["length"][m]
        b = {"auc_length_label": auc(D["y"][m], lg),
             "auc_refusal_label": auc(D["y"][m], D["refusal"][m].astype(float)),
             "auc_proj_length": auc((lg > np.median(lg)).astype(int), D["proj"][m]),
             "auc_proj_refusal": auc(D["refusal"][m], D["proj"][m]),
             "n_refusals": int(D["refusal"][m].sum())}
        out["baselines"][st] = b
        print(f"{st:<26}{b['auc_length_label']:>10.3f}{b['auc_refusal_label']:>10.3f}"
              f"{b['auc_proj_length']:>11.3f}{b['auc_proj_refusal']:>11.3f}"
              f"{b['n_refusals']:>7}")
    print("\n  A confound only explains the result if the direction tracks it AND it\n"
          "  tracks the label. Both columns have to move together.")

    # ---- results ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TRANSFER AUC")
    print("=" * 78)
    report("ALL", D["y"], D["proj"], strata, D["cluster"], args.n_boot, args.seed, out)
    for ds in np.unique(D["dataset"]):
        m = D["dataset"] == ds
        report(ds, D["y"][m], D["proj"][m], strata[m], D["cluster"][m],
               args.n_boot, args.seed, out)
    if len(np.unique(D["model"])) > 1:
        for md in np.unique(D["model"]):
            m = D["model"] == md
            report(f"model={md}", D["y"][m], D["proj"][m], strata[m], D["cluster"][m],
                   args.n_boot, args.seed, out)

    print("\n" + "=" * 78)
    print("Frame Colleague is ONE scenario. Its 400 rows buy precision about this\n"
          "prompt, not breadth across scenarios, and no CI here represents\n"
          "uncertainty about generalising to other misalignment scenarios.")
    print("=" * 78)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
