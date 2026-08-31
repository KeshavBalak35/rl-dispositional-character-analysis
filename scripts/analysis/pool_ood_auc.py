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
import math
import os
import re
import sys

import numpy as np

REFUSAL_RE = re.compile(
    r"^\s*(?:i'?m\s+sorry|i\s+am\s+sorry|sorry[,.]|i\s+cannot|i\s+can'?t|"
    r"i\s+apologize|i\s+won'?t|as\s+an\s+ai(?:\s+language\s+model)?[,.]?\s+i)",
    re.IGNORECASE)

_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"</?[a-z|_]+\|?>", re.IGNORECASE)


def visible_text(t: str) -> str:
    """
    What the judge should have been scoring: the response with private reasoning
    and stray chat-template tokens removed.

    A bare '<thinking>' with nothing after it is NOT caught by `not t.strip()`,
    which is the only degeneracy check run_frame_colleague_v2.py applies before
    calling the judge. So those rows reach Sonnet, which returns a verdict on
    nothing. Same class of artifact as the orphaned <thinking> rows in the
    chat-eval data.
    """
    s = _THINK_RE.sub(" ", t or "")
    s = re.sub(r"<think(?:ing)?>.*", " ", s, flags=re.IGNORECASE | re.DOTALL)
    return _TAG_RE.sub(" ", s).strip()


def is_degenerate(t: str, min_chars: int) -> bool:
    return len(visible_text(t)) < min_chars


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


def cohens_d(y, s):
    """
    Standardised class separation.

    AUC is rank-based and scale-free, so it cannot distinguish "weaker effect"
    from "noisier measurement". If one model's projections have larger spread,
    the same underlying separation reads as a smaller AUC. d makes that visible.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    p, n = s[y == 1], s[y == 0]
    if len(p) < 2 or len(n) < 2:
        return float("nan")
    sp = math.sqrt(((len(p) - 1) * p.var(ddof=1) + (len(n) - 1) * n.var(ddof=1))
                   / (len(p) + len(n) - 2))
    return float((p.mean() - n.mean()) / sp) if sp > 0 else float("nan")


def paired_model_bootstrap(y, s, strata, clusters, model, n_boot=2000, seed=0):
    """
    Bootstrap both model arms on the SAME resample and difference within draw.

    Two separate CIs are not a test of a difference. They can overlap while the
    paired difference excludes zero, and they can both exclude 0.5 while the
    difference between them is indistinguishable from zero. Differencing within
    draw cancels the resampling noise the two arms share.

    Returns a dict, or None when there are not exactly two arms.
    """
    arms = sorted(np.unique(model).tolist())
    if len(arms) != 2:
        return None
    a0, a1 = arms
    rng = np.random.default_rng(seed)
    plan = []
    for st in np.unique(strata):
        m = np.where(strata == st)[0]
        cl = clusters[m]
        plan.append([m[cl == c] for c in np.unique(cl)])

    v0, v1, dv = [], [], []
    dropped = 0
    for _ in range(n_boot):
        idx = np.concatenate([
            np.concatenate([by[p] for p in rng.integers(0, len(by), len(by))])
            for by in plan])
        ys, ss, sts, ms = y[idx], s[idx], strata[idx], model[idx]
        m0, m1 = ms == a0, ms == a1
        c0, _, _ = stratified_concordance(ys[m0], ss[m0], sts[m0])
        c1, _, _ = stratified_concordance(ys[m1], ss[m1], sts[m1])
        if np.isnan(c0) or np.isnan(c1):
            dropped += 1
            continue
        v0.append(c0); v1.append(c1); dv.append(c0 - c1)
    if len(dv) < n_boot * 0.5:
        return {"arms": [a0, a1], "usable_draws": len(dv), "dropped": dropped,
                "degenerate": True}
    dv = np.array(dv)
    return {
        "arms": [a0, a1], "usable_draws": len(dv), "dropped": dropped,
        "degenerate": False,
        f"{a0}_ci95": [float(np.percentile(v0, 2.5)), float(np.percentile(v0, 97.5))],
        f"{a1}_ci95": [float(np.percentile(v1, 2.5)), float(np.percentile(v1, 97.5))],
        "delta_mean": float(dv.mean()), "delta_se": float(dv.std(ddof=1)),
        "delta_ci95": [float(np.percentile(dv, 2.5)), float(np.percentile(dv, 97.5))],
        "p_delta_gt_0": float((dv > 0).mean()),
    }


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
        "text": np.array(texts, dtype=object),
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
        "text": np.array(texts, dtype=object),
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
    ap.add_argument("--exclude-degenerate", type=int, default=0, metavar="MIN_CHARS",
                    help="drop rows whose response has fewer than MIN_CHARS of "
                         "visible text after stripping <thinking> blocks and stray "
                         "template tags. Applied to EVERY stratum and blind to the "
                         "label; try 40. 0 disables.")
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

    keys = ("y", "proj", "dataset", "cluster", "model", "length", "refusal", "text")
    D = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    if args.datasets:
        m = np.isin(D["dataset"], args.datasets)
        D = {k: v[m] for k, v in D.items()}
    strata = np.array([f"{d}|{m}" for d, m in zip(D["dataset"], D["model"])])

    if args.exclude_degenerate > 0:
        bad = np.array([is_degenerate(t, args.exclude_degenerate) for t in D["text"]])
        print("\n" + "=" * 78)
        print(f"DEGENERATE EXCLUSION (<{args.exclude_degenerate} visible chars), "
              "applied symmetrically to every stratum")
        print("=" * 78)
        print(f"{'stratum':<26}{'dropped':>9}{'of which pos':>14}{'pos left':>10}")
        for st in np.unique(strata):
            m = strata == st
            dp = int((bad & m).sum())
            print(f"{st:<26}{dp:>9}{int((bad & m & (D['y'] == 1)).sum()):>14}"
                  f"{int((~bad & m & (D['y'] == 1)).sum()):>10}")
        print(f"  total dropped {int(bad.sum())} of {len(bad)}")
        print("  The rule is defined on response text only. Applying it to one arm "
              "and not the other would improve that arm's data quality relative to "
              "the other and move the delta for that reason alone.")
        D = {k: v[~bad] for k, v in D.items()}
        strata = strata[~bad]
        if len(D["y"]) == 0:
            sys.exit("everything excluded")

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

        # ---- the difference itself, not two separate intervals -------------
        print("\n" + "=" * 78)
        print("PER-MODEL DIFFERENCE (paired: both arms from the same resample)")
        print("=" * 78)
        pm = paired_model_bootstrap(D["y"], D["proj"], strata, D["cluster"],
                                    D["model"], args.n_boot, args.seed)
        if pm is None:
            print("  need exactly two model arms")
        elif pm.get("degenerate"):
            print(f"  only {pm['usable_draws']} usable draws "
                  f"({pm['dropped']} had an arm with no positives). One arm is too "
                  "thin to bootstrap; the difference is not estimable.")
            out["per_model_delta"] = pm
        else:
            a0, a1 = pm["arms"]
            for a in (a0, a1):
                m = D["model"] == a
                obs, _, _ = stratified_concordance(D["y"][m], D["proj"][m], strata[m])
                d = cohens_d(D["y"][m], D["proj"][m])
                lo, hi = pm[f"{a}_ci95"]
                print(f"  {a:<6} AUC {obs:.3f}  CI [{lo:.3f}, {hi:.3f}]   "
                      f"Cohen's d {d:+.3f}   n_pos {int((D['y'][m] == 1).sum())}")
                out.setdefault("per_model_effect", {})[a] = {
                    "auc": obs, "cohens_d": d,
                    "n_pos": int((D["y"][m] == 1).sum())}
            lo, hi = pm["delta_ci95"]
            print(f"\n  DELTA ({a0} - {a1})  {pm['delta_mean']:+.3f}   "
                  f"SE {pm['delta_se']:.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
            print(f"  P(delta > 0) = {pm['p_delta_gt_0']:.3f}")
            if lo <= 0.0 <= hi:
                print("  CI SPANS ZERO: the two arms are not distinguishable. Do not "
                      "report a per-model finding.")
            else:
                print("  CI excludes zero: the arms differ beyond resampling noise. "
                      "That is still a statistical claim, not a representational "
                      "one, until the label construct is shown comparable across "
                      "arms.")
            if pm["dropped"]:
                print(f"  ({pm['dropped']} of {args.n_boot} draws discarded for "
                      "having an arm with only one class)")
            out["per_model_delta"] = pm
            print("\n  Compare Cohen's d across arms as well as AUC: a smaller AUC "
                  "with a similar d means noisier projections, not a weaker effect.")

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
