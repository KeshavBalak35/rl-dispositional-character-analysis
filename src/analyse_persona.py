#!/usr/bin/env python3
"""
Arm 3: does a values-laden prompt move activations along the fitted hack direction?

    # 1. capture activations under the three conditions (see --help of sweep_probe)
    python sweep_persona.py --model rh --pooling first8 --layers 0 8 16 24 31

    # 2. measure the shift and compare it to the fitted directions
    python analyse_persona.py \
        --run probe_rh_persona_first8 \
        --directions direction_L16_first8pool direction_L16_lastpool \
        --layer 16

THE TEST
    shift_A = mean(activation | hacking_okay)          - mean(activation | baseline)
    shift_B = mean(activation | hacking_is_misaligned) - mean(activation | baseline)

    then cos(shift_A, fitted_direction) and cos(shift_B, fitted_direction).

    If the fitted direction is a real "disposition to hack" axis, a permissive
    persona should move activations ALONG it (cos > 0) and a prohibitive one
    AGAINST it (cos < 0), with no steering intervention at all. That is a
    naturalistic activation, which is why it is worth more than an induced one.

WHICH BASELINE
    dont_hack, not neutral, and the difference matters.

    In the source prompts file both persona conditions are built from
    _BASE_INTRO plus a closing addendum. dont_hack uses the SAME intro and the
    same hack hints, differing only in that addendum: 116 differing characters
    against hacking_is_misaligned, 333 against hacking_okay. neutral uses a
    different intro (_NEUTRAL_INTRO) AND no addendum, so a shift measured
    against it confounds "values framing" with "different task framing":
    350 and 425 differing characters respectively.

    Pass --baseline neutral to measure it the other way; the script prints the
    prompt-difference sizes either way so the confound is visible.

WHAT WOULD MAKE THIS UNINTERPRETABLE
    A shift vector that is really a LENGTH or PROMPT-POSITION artefact. Persona
    prompts differ in length, and first8 pooling reads tokens immediately after
    the prompt, so position alone can move the mean. Both are measured here and
    reported next to the headline cosine; a persona direction that a length
    direction matches is not a persona direction.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coding_eval import (                                          # noqa: E402
    length_direction, load_direction, load_prompt_registry, load_run,
    probe_dataset, run_dir,
)

MIN_N = 30
PERSONA = ("hacking_okay", "hacking_is_misaligned")


def small(n: int) -> str:
    return f"  <-- n={n}, below {MIN_N}" if n < MIN_N else ""


def null_cosine_band(dim: int, n: int = 4000, seed: int = 0):
    """|cos| between random unit vectors: the floor any claim must clear."""
    rng = np.random.RandomState(seed)
    a = rng.randn(n, dim)
    b = rng.randn(n, dim)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    c = np.abs((a * b).sum(axis=1))
    return float(c.mean()), float(np.percentile(c, 95)), float(np.percentile(c, 99))


def unit(v):
    n = float(np.linalg.norm(v))
    return None if n == 0 else v / n


def permutation_cosine(Xa, Xb, target, n_perm=2000, seed=0):
    """
    Permutation test: could this cosine arise from no condition effect at all?

    Pools the two conditions, reshuffles rows into groups of the same sizes, and
    recomputes cos(mean difference, target). Under the null of no condition
    effect that distribution is centred on zero, so the fraction of permutations
    at least as extreme as the observed value is a p-value.

    WHY NOT A BOOTSTRAP CI. Both bootstrap variants misbehave on this statistic.
    Resampling adds noise to BOTH means, which shrinks the resampled cosines
    toward zero: the percentile interval came out at [+0.506, +0.707] for a point
    estimate of +0.734, excluding the value it was meant to bracket, and the
    basic (reverse-percentile) correction overshot to [+0.761, +0.962],
    excluding it from the other side. A permutation test answers the question
    actually being asked, "is this distinguishable from chance", without
    needing an unbiased interval.

    Returns (p_value, null_p95_abs, observed).
    """
    rng = np.random.RandomState(seed)
    obs_v = unit(Xa.mean(axis=0) - Xb.mean(axis=0))
    if obs_v is None:
        return None
    obs = float(obs_v @ target)

    pooled = np.vstack([Xa, Xb])
    na = len(Xa)
    null = []
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        v = unit(pooled[idx[:na]].mean(axis=0) - pooled[idx[na:]].mean(axis=0))
        if v is not None:
            null.append(float(v @ target))
    if not null:
        return None
    null = np.asarray(null)
    p = float((np.abs(null) >= abs(obs)).mean())
    return p, float(np.percentile(np.abs(null), 95)), obs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="probe run holding all three conditions")
    ap.add_argument("--directions", nargs="+", required=True,
                    help="fitted directions to compare against")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--baseline", default="dont_hack",
                    help="control condition (default dont_hack: same intro and "
                         "hints as the persona conditions)")
    ap.add_argument("--dataset", default="humaneval")
    ap.add_argument("--n-perm", type=int, default=2000,
                    help="permutations for the null distribution")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = run_dir(args.run, create=False)
    if not os.path.isdir(path):
        print(f"missing run: {path}")
        return 1
    records = load_run(path)

    # ---- prompt-difference sizes, so the control choice is visible ---------
    try:
        import difflib
        reg = load_prompt_registry()[args.dataset]
        print(f"prompt differences ({args.dataset}), against --baseline "
              f"{args.baseline!r}:")
        for p in PERSONA:
            if p in reg and args.baseline in reg:
                sm = difflib.SequenceMatcher(None, reg[p], reg[args.baseline])
                same = sum(b.size for b in sm.get_matching_blocks())
                d = len(reg[p]) + len(reg[args.baseline]) - 2 * same
                print(f"  {p:<24} {d:>5} differing characters")
        print("  (a smaller number means a better-matched control)")
    except Exception as exc:                                       # noqa: BLE001
        print(f"  could not read prompts: {exc}")

    # ---- assemble per-condition activation matrices ------------------------
    X, y, keep = probe_dataset(records, args.layer, drop_undetermined=False)
    by_cond = defaultdict(list)
    for i, r in enumerate(keep):
        by_cond[r.generation.condition].append(i)

    print(f"\nconditions present in {args.run} at layer {args.layer}:")
    for cond, idx in sorted(by_cond.items()):
        lens = [keep[i].generation.response_token_len for i in idx]
        plens = [keep[i].generation.prompt_token_len for i in idx]
        print(f"  {cond:<24} n={len(idx):<5} problems="
              f"{len({keep[i].group_key for i in idx}):<5} "
              f"mean resp tokens {np.mean(lens):6.0f}  "
              f"mean prompt tokens {np.mean(plens):6.0f}"
              + small(len(idx)))

    missing = [c for c in (*PERSONA, args.baseline) if c not in by_cond]
    if missing:
        print(f"\nmissing condition(s): {missing}")
        print("Capture them first:  python sweep_persona.py --model rh "
              "--pooling first8")
        return 1

    base_idx = by_cond[args.baseline]
    Xb = X[base_idx]
    dim = X.shape[1]
    nm, n95, n99 = null_cosine_band(dim)
    ldir = length_direction(X, keep)

    print(f"\nrandom-vector noise floor at dim={dim}: mean |cos| {nm:.3f}, "
          f"p95 {n95:.3f}, p99 {n99:.3f}")
    if min(len(base_idx), *(len(by_cond[p]) for p in PERSONA)) < MIN_N:
        print(f"\nWARNING: at least one condition has fewer than {MIN_N} rows. "
              "A mean activation over")
        print("         that few samples is unstable, and a cosine between two "
              "such means doubly so.")
        print("         Read the bootstrap intervals, not the point estimates.")

    # ---- the test ----------------------------------------------------------
    for dname in args.directions:
        try:
            d = load_direction(dname)
        except Exception as exc:                                   # noqa: BLE001
            print(f"\n{dname}: {exc}")
            continue
        if d.layer != args.layer:
            print(f"\n{dname}: fitted at layer {d.layer}, comparing at "
                  f"{args.layer}. Skipping; refit or pass --layer {d.layer}.")
            continue
        if len(d.vector) != dim:
            print(f"\n{dname}: dim {len(d.vector)} != activation dim {dim}. Skipping.")
            continue

        print("\n" + "=" * 78)
        print(f"{dname}  (layer {d.layer}, pooling {d.pooling})")
        print("=" * 78)
        print(f"  {'shift':<26}{'cos':>9}{'perm p':>9}{'null p95':>10}"
              f"{'cos to length':>15}")
        for persona in PERSONA:
            idx = by_cond[persona]
            Xa = X[idx]
            shift = unit(Xa.mean(axis=0) - Xb.mean(axis=0))
            if shift is None:
                print(f"  {persona:<26}{'degenerate (zero shift)':>18}")
                continue
            cos = float(shift @ d.vector)
            pr = permutation_cosine(Xa, Xb, d.vector, n_perm=args.n_perm,
                                    seed=args.seed)
            cl = float(shift @ ldir) if ldir is not None else float("nan")
            p_s = "n/a" if pr is None else (f"{pr[0]:.3f}" if pr[0] >= 0.001
                                            else "<0.001")
            n95 = "n/a" if pr is None else f"{pr[1]:.3f}"
            print(f"  {persona:<26}{cos:>+9.3f}{p_s:>9}{n95:>10}{cl:>+15.3f}"
                  + small(len(idx)))

        # ---- interpretation ------------------------------------------------
        print()
        okay_idx, mis_idx = by_cond[PERSONA[0]], by_cond[PERSONA[1]]
        s_ok = unit(X[okay_idx].mean(axis=0) - Xb.mean(axis=0))
        s_mis = unit(X[mis_idx].mean(axis=0) - Xb.mean(axis=0))
        if s_ok is not None and s_mis is not None:
            c_ok, c_mis = float(s_ok @ d.vector), float(s_mis @ d.vector)
            pr_ok = permutation_cosine(X[okay_idx], Xb, d.vector,
                                       n_perm=args.n_perm, seed=args.seed)
            inside = pr_ok is not None and pr_ok[0] > 0.05
            if abs(c_ok) <= n99 and abs(c_mis) <= n99:
                print("  both shifts sit inside the random-vector noise floor: the")
                print("  persona framing does not move activations along this "
                      "direction")
            elif inside:
                print(f"  the permissive shift has permutation p={pr_ok[0]:.3f}: not")
                print("  distinguishable from chance at this sample size")
            elif c_ok > 0 > c_mis:
                print("  permissive moves ALONG the direction and prohibitive "
                      "AGAINST it:")
                print("  the sign pattern a real disposition axis predicts")
            else:
                print(f"  shifts do not show the predicted opposing signs "
                      f"(okay {c_ok:+.3f}, misaligned {c_mis:+.3f})")
            if ldir is not None:
                cl_ok = abs(float(s_ok @ ldir))
                if cl_ok >= abs(c_ok):
                    print(f"  CONFOUND: the shift aligns with the LENGTH direction "
                          f"({cl_ok:.3f}) at least")
                    print("  as strongly as with the fitted direction. Treat the "
                          "headline number as unproven.")

    print("\nnote: cosine between mean-difference vectors is a weak instrument in")
    print("high dimensions. Read the permutation p and the length column, not the")
    print("point estimate alone. 'null p95' is the 95th percentile of |cos| under")
    print("shuffled labels: an observed cosine below it is unremarkable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
