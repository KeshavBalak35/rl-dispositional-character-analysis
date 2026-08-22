#!/usr/bin/env python3
"""
Decompose a fixed direction's label AUC on the chat-eval data.

WHY NOT "REFIT WITH GROUPED CV"

The direction is fixed. It was fitted on coding-eval data and nothing is
estimated from these rows, so there is no optimism for cross-validation to
remove and StratifiedGroupKFold would return the same point estimate with extra
steps. What grouping actually buys is an honest error bar and a decomposition.

The pooled AUC is currently computed as if there are ~1039 independent
observations. There are 256 questions, and 840 of those rows are 15-sample
blocks of the same question. So this script reports:

  1. CONFOUND BASELINES first. If the projection is really a length or refusal
     detector and those predict the label, there is nothing else to explain.
  2. POOLED AUC with a cluster bootstrap over questions, so the CI reflects
     256 effective units rather than 1039 rows.
  3. WITHIN-QUESTION AUC. Restricted to questions holding both labels, this
     conditions question identity away. It is the sharp test.
  4. BETWEEN-QUESTION association. Per-question mean projection against
     per-question positive rate. If the signal lives here and not within,
     the direction is tracking topic, not the label.

Every number is reported for Betley and alignment separately as well as pooled,
because the two are graded by different judge prompts and their labels do not
mean the same thing. Alignment has one sample per question, so its
within-question figure is undefined by construction and pooled == between.

INPUT

Reuses format_control/paired_format_comparison.npz from
paired_format_reforward.py. No forward passes, no GPU, runs in seconds.

USAGE

    python grouped_auc_decomposition.py \\
        --comparison format_control/paired_format_comparison.npz \\
        --responses clean_responses.json \\
        --direction /path/to/_steering/direction_L16_lastpool.npz \\
        --question-ids question_ids.json \\
        --out format_control/auc_decomposition.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

# Matches the refusal openings that dominate the negative class in this data.
REFUSAL_RE = re.compile(
    r"^\s*(?:i'?m\s+sorry|i\s+am\s+sorry|sorry[,.]|i\s+cannot|i\s+can'?t|"
    r"i\s+apologize|i\s+won'?t|as\s+an\s+ai(?:\s+language\s+model)?[,.]?\s+i)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def auc(y: np.ndarray, s: np.ndarray) -> float:
    """
    Rank-based AUC with proper tie handling. Returns nan if either class is empty.
    Equivalent to roc_auc_score but keeps this script dependency-free.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def within_group_auc(y: np.ndarray, s: np.ndarray, g: np.ndarray):
    """
    Stratified concordance: pool discordant pairs WITHIN each group, never across.

    This is the estimator that answers "does the projection separate the
    misaligned completions from the aligned completions of the SAME question".
    Returns (auc, n_pairs, n_groups_used).
    """
    conc = 0.0
    pairs = 0
    used = 0
    for grp in np.unique(g):
        m = g == grp
        yy, ss = y[m], s[m]
        pos, neg = ss[yy == 1], ss[yy == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        used += 1
        diff = pos[:, None] - neg[None, :]
        conc += float((diff > 0).sum() + 0.5 * (diff == 0).sum())
        pairs += diff.size
    if pairs == 0:
        return float("nan"), 0, 0
    return conc / pairs, pairs, used


def cluster_bootstrap_auc(y, s, g, n_boot=2000, seed=0):
    """
    Resample GROUPS with replacement, not rows. The row-level CI is wrong here
    because 15 completions of one question are not 15 independent observations.
    """
    rng = np.random.default_rng(seed)
    groups = np.unique(g)
    idx_by_group = {grp: np.where(g == grp)[0] for grp in groups}
    stats = []
    for _ in range(n_boot):
        drawn = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([idx_by_group[d] for d in drawn])
        a = auc(y[idx], s[idx])
        if not np.isnan(a):
            stats.append(a)
    if len(stats) < n_boot * 0.5:
        return float("nan"), float("nan"), len(stats)
    stats = np.array(stats)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)), len(stats)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, no scipy."""
    if len(a) < 3:
        return float("nan")
    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=float)
        sx = x[order]
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and sx[j + 1] == sx[i]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def analyse(name, y, s, g, out, n_boot, seed):
    """Print and collect the three-way decomposition for one slice."""
    res = {"n_rows": int(len(y)), "n_groups": int(len(np.unique(g))),
           "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}

    print(f"\n  {name}: {res['n_rows']} rows, {res['n_groups']} questions, "
          f"{res['n_pos']} pos / {res['n_neg']} neg")

    if res["n_pos"] == 0 or res["n_neg"] == 0:
        print("    only one class present, nothing to compute")
        out[name] = res
        return

    res["pooled_auc"] = auc(y, s)
    lo, hi, nb = cluster_bootstrap_auc(y, s, g, n_boot=n_boot, seed=seed)
    res["pooled_ci95"] = [lo, hi]
    res["bootstrap_draws_used"] = nb
    crosses = (lo <= 0.5 <= hi) if not np.isnan(lo) else True
    print(f"    pooled AUC        {res['pooled_auc']:.3f}   "
          f"cluster-bootstrap 95% CI [{lo:.3f}, {hi:.3f}]"
          f"{'   CROSSES 0.5' if crosses else ''}")

    wa, wp, wg = within_group_auc(y, s, g)
    res.update({"within_question_auc": wa, "within_question_pairs": wp,
                "within_question_groups": wg})
    if wg == 0:
        print("    within-question   undefined (no question holds both labels)")
    else:
        print(f"    within-question   {wa:.3f}   ({wp} discordant pairs "
              f"across {wg} questions)")

    groups = np.unique(g)
    mean_proj = np.array([s[g == q].mean() for q in groups])
    pos_rate = np.array([y[g == q].mean() for q in groups])
    rho = spearman(mean_proj, pos_rate)
    res["between_question_spearman"] = rho
    res["between_question_n"] = int(len(groups))
    print(f"    between-question  Spearman(mean proj, positive rate) = {rho:+.3f} "
          f"over {len(groups)} questions")

    out[name] = res


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comparison", required=True,
                   help="paired_format_comparison.npz from paired_format_reforward.py")
    p.add_argument("--responses", required=True, help="clean_responses.json")
    p.add_argument("--direction", required=True,
                   help="the direction .npz that produced the projections; its "
                        ".json sidecar must sit beside it")
    p.add_argument("--question-ids", default=None,
                   help="question_ids.json, cross-checked against the ids already "
                        "stored in the comparison npz")
    p.add_argument("--expected-pooling", default="last",
                   help="pooling the stored activations used")
    p.add_argument("--allow-pooling-mismatch", action="store_true",
                   help="score a direction fitted under different pooling anyway. "
                        "The result is NOT a transfer measurement and is stamped "
                        "non_comparable in the output.")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write the full report as JSON")
    args = p.parse_args()

    d = np.load(args.comparison, allow_pickle=True)
    need = {"rows", "labels", "qids", "kinds", "proj_manual", "proj_chatml"}
    if not need.issubset(set(d.files)):
        sys.exit(f"{args.comparison} is missing {sorted(need - set(d.files))}. "
                 "Re-run paired_format_reforward.py with --direction.")

    rows = d["rows"]
    y = d["labels"].astype(int)
    qids = np.array([str(x) for x in d["qids"]])
    kinds = np.array([str(x) for x in d["kinds"]])
    projections = {"manual": d["proj_manual"], "chatml": d["proj_chatml"]}
    dir_layer = int(d["dir_layer"]) if "dir_layer" in d.files else None

    report = {"comparison_file": os.path.abspath(args.comparison),
              "direction": os.path.abspath(args.direction),
              "direction_layer": dir_layer, "n_rows": int(len(rows))}

    # ---------------------------------------------------------------- pooling
    sidecar = args.direction[:-4] + ".json"
    if not os.path.isfile(sidecar):
        sys.exit(f"{sidecar} not found; pooling and layer live in the sidecar.")
    with open(sidecar) as f:
        meta = json.load(f)
    pooling = meta.get("pooling")
    report["direction_pooling"] = pooling
    print(f"direction: layer {meta.get('layer')}, pooling {pooling!r}")
    if pooling is not None and pooling != args.expected_pooling:
        msg = (f"POOLING MISMATCH: direction was fitted with pooling={pooling!r} but "
               f"the stored activations are {args.expected_pooling!r}. Projecting one "
               "pooling's vectors onto another pooling's direction computes a number "
               "but does not measure transfer.")
        if not args.allow_pooling_mismatch:
            sys.exit(msg + "\nTo compare first8pool you need a forward pass that keeps "
                           "the first 8 response positions. Pass "
                           "--allow-pooling-mismatch to look anyway.")
        print("WARNING: " + msg)
        report["non_comparable"] = True

    # ------------------------------------------------------- sign convention
    dz = np.load(args.direction, allow_pickle=True)
    if {"mu_hack", "mu_clean"}.issubset(set(dz.files)):
        v = np.asarray(dz["direction"], dtype=np.float64).ravel()
        gap = np.asarray(dz["mu_hack"], np.float64).ravel() - \
              np.asarray(dz["mu_clean"], np.float64).ravel()
        s_sign = float(v @ gap)
        report["direction_dot_mu_gap"] = s_sign
        if s_sign > 0:
            print(f"sign convention: direction . (mu_hack - mu_clean) = {s_sign:+.3f}"
                  "  -> HIGH projection means hack-like")
        else:
            print(f"sign convention: direction . (mu_hack - mu_clean) = {s_sign:+.3f}"
                  "  -> HIGH projection means CLEAN-like. An AUC below 0.5 against a "
                  "misalignment label is then the EXPECTED direction, not an anomaly.")
    else:
        print("sign convention: mu_hack/mu_clean not in the npz, cannot verify. "
              "Check fit_direction.py before interpreting the sign of any AUC.")

    # -------------------------------------------------------- id cross-check
    if args.question_ids:
        with open(args.question_ids) as f:
            all_qids = [str(x) for x in json.load(f)]
        if len(all_qids) < int(rows.max()) + 1:
            sys.exit(f"{args.question_ids} has {len(all_qids)} entries, too short for "
                     f"row index {int(rows.max())}")
        theirs = np.array([all_qids[i] for i in rows])
        n_diff = int((theirs != qids).sum())
        # The reforward script rebuilt ids from betley.py native ids; the notebook's
        # file may use positional betley_{i}. A pure relabelling is fine as long as
        # the PARTITION is identical, which is all any of these statistics use.
        same_partition = (
            len({(a, b) for a, b in zip(theirs, qids)}) == len(set(qids)) == len(set(theirs))
        )
        print(f"question_ids.json cross-check: {n_diff} label differences, "
              f"partition identical: {same_partition}")
        if not same_partition:
            sys.exit("question_ids.json induces a DIFFERENT grouping than the "
                     "comparison npz. Resolve before trusting any grouped statistic.")
        report["question_ids_partition_matches"] = True

    # ----------------------------------------------------- confound baselines
    with open(args.responses) as f:
        responses = json.load(f)
    resp = [responses[i] for i in rows]
    length = np.array([len(t) for t in resp], dtype=float)
    refusal = np.array([1 if REFUSAL_RE.match(t) else 0 for t in resp], dtype=int)

    print("\n" + "=" * 74)
    print("CONFOUND BASELINES  (read these before the label AUC)")
    print("=" * 74)
    print(f"refusals matched: {int(refusal.sum())} of {len(refusal)} rows; "
          f"label==1 among refusals: {int(y[refusal == 1].sum())}")
    print(f"AUC(length   -> label)  {auc(y, length):.3f}")
    print(f"AUC(refusal  -> label)  {auc(y, refusal.astype(float)):.3f}")
    report["baselines"] = {
        "n_refusals": int(refusal.sum()),
        "n_refusal_positive": int(y[refusal == 1].sum()),
        "auc_length_label": auc(y, length),
        "auc_refusal_label": auc(y, refusal.astype(float)),
    }
    for fmt, proj in projections.items():
        a_len = auc((length > np.median(length)).astype(int), proj)
        a_ref = auc(refusal, proj)
        print(f"AUC(proj[{fmt}] -> long response)  {a_len:.3f}     "
              f"AUC(proj[{fmt}] -> is refusal)  {a_ref:.3f}")
        report["baselines"][f"auc_proj_{fmt}_length"] = a_len
        report["baselines"][f"auc_proj_{fmt}_refusal"] = a_ref
    print("\nIf the projection tracks length or refusal AND those track the label,\n"
          "the label AUC below is explained without invoking misalignment at all.")

    # ------------------------------------------------------------- main pass
    slices = {
        "ALL": np.ones(len(y), bool),
        "betley": kinds == "betley",
        "alignment": kinds == "alignment",
        "betley_non_refusal": (kinds == "betley") & (refusal == 0),
        "alignment_non_refusal": (kinds == "alignment") & (refusal == 0),
    }

    for fmt, proj in projections.items():
        print("\n" + "=" * 74)
        print(f"LABEL AUC DECOMPOSITION  [{fmt} format]")
        print("=" * 74)
        report[fmt] = {}
        for name, m in slices.items():
            if m.sum() == 0:
                continue
            analyse(name, y[m], proj[m], qids[m], report[fmt], args.n_boot, args.seed)

    print("\n" + "=" * 74)
    print("HOW TO READ THIS")
    print("=" * 74)
    print("""\
alignment has one sample per question, so its within-question figure is
undefined and pooled == between by construction. Only Betley can distinguish
the two.

For Betley, the four outcomes are:

  within ~ 0.5, between strong   the projection tracks WHICH QUESTION, not the
                                 label. Given the category analysis, that means
                                 it tracks "is this a medical-emergency
                                 scenario". Not a misalignment finding.
  within strong, between ~ 0     it separates responses to the same question.
                                 This is the real result.
  both strong                    plausible, but check the baselines above before
                                 claiming it.
  both ~ 0.5, pooled CI crosses  the pooled number was small-sample structure in
                                 a handful of questions. Nothing to explain.""")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
