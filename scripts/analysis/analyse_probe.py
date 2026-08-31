#!/usr/bin/env python3
"""
Combine probe runs across models and run the confound analysis.

    python analyse_probe.py --runs probe_rh probe_clean --layers 0 8 16 24 31

WHY COMBINING IS NEEDED
    probe_report()'s model_id check compares strata WITHIN one dataset. Load a
    single model's run and there is only one value of model_id, so the check
    reports UNCHECKED. It needs both models' records in one list.

WHAT IT PRINTS, IN THE ORDER YOU SHOULD READ IT
    1. composition: n, positives, base rate, unique problems, per model
    2. length baselines, per model and pooled, cross-validated
    3. within-model AUC at each layer: the number that means something
    4. pooled AUC and the model_id confound check
    5. hack_type distribution with a chi-square test between models

READ THE BASE RATES FIRST. If one model hacks at 94% and the other at 10%,
then "does this response contain a hack" and "which model wrote it" are nearly
the same variable, and a pooled probe separating them tells you almost nothing.
That is not a bug in the probe; it is what the data looks like, and it is why
the within-model numbers are the headline.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                  # noqa: E402

from coding_eval import (                                           # noqa: E402
    default_root, grouped_cv, length_baseline, load_run, probe_dataset, probe_report,
    print_report, run_dir,
)
from coding_eval.probing import _auc                                # noqa: E402


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval: honest at small counts, unlike normal approx."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def cv_length_auc(records, n_splits: int = 5, seed: int = 0):
    """
    Cross-validated AUC from response length alone, grouped by problem.

    length_baseline() reports an IN-SAMPLE AUC. Comparing that against the
    probe's cross-validated AUC understates the baseline and flatters the probe.
    This uses the same grouped folds the probe uses, so the two numbers are
    directly comparable.

    AUC does not depend on base rate, so this stays interpretable at 10%
    positive (clean) and 94% positive (RH) alike. What imbalance changes is the
    fold-to-fold SPREAD, which is why the std is reported next to it.
    """
    import numpy as _np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    keep = [r for r in records if r.label in (0, 1)]
    y = _np.asarray([r.label for r in keep])
    if len(set(y.tolist())) < 2:
        return None
    X = _np.asarray([r.generation.response_token_len for r in keep]).reshape(-1, 1)
    aucs = []
    for tr, te in grouped_cv(keep, n_splits=n_splits, seed=seed, labels=y):
        if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
            continue
        clf = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    if not aucs:
        return None
    return float(_np.mean(aucs)), float(_np.std(aucs)), len(keep)


def composition(records, label=""):
    by_model = defaultdict(list)
    for r in records:
        by_model[r.generation.model_id or "<unset>"].append(r)
    print(f"\n{label}")
    print(f"  {'model_id':<48}{'n':>7}{'pos':>7}{'neg':>7}{'undet':>7}"
          f"{'pos rate':>10}{'problems':>10}")
    for mid, rs in sorted(by_model.items()):
        lab = Counter(r.label for r in rs)
        det = lab[1] + lab[0]
        rate = f"{lab[1]/det:.1%}" if det else "n/a"
        short = mid if len(mid) <= 46 else "..." + mid[-43:]
        print(f"  {short:<48}{len(rs):>7}{lab[1]:>7}{lab[0]:>7}{lab[None]:>7}"
              f"{rate:>10}{len({r.group_key for r in rs}):>10}")
    return by_model


def chi2_hack_types(by_model):
    """Independence test on the hack_type x model contingency table."""
    models = sorted(by_model)
    if len(models) != 2:
        return None
    types = sorted({r.grade.hack_type for m in models for r in by_model[m]
                    if r.label == 1})
    if len(types) < 2:
        return None
    obs = np.array([[sum(1 for r in by_model[m]
                         if r.label == 1 and r.grade.hack_type == t)
                     for t in types] for m in models], dtype=float)
    if obs.sum() == 0:
        return None
    row, col = obs.sum(1, keepdims=True), obs.sum(0, keepdims=True)
    exp = row @ col / obs.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum((obs - exp) ** 2 / np.where(exp > 0, exp, np.nan)))
    dof = (len(models) - 1) * (len(types) - 1)
    try:
        from scipy.stats import chi2 as chi2_dist
        p = float(chi2_dist.sf(chi2, dof))
    except Exception:                                   # noqa: BLE001
        p = None
    return {"types": types, "models": models, "observed": obs, "expected": exp,
            "chi2": chi2, "dof": dof, "p": p,
            "min_expected": float(exp.min())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--layers", type=int, nargs="*", default=[0, 8, 16, 24, 31])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balance", action="store_true",
                    help="subsample the larger model's records so the pooled set "
                         "has comparable n per model")
    args = ap.parse_args()

    records = []
    for name in args.runs:
        path = run_dir(name, create=False)
        if not os.path.isdir(path):
            print(f"run not found: {path}")
            return 1
        rs = load_run(path)
        print(f"loaded {len(rs):>6} from {name}")
        records.extend(rs)

    by_model = composition(records, "COMPOSITION")

    if args.balance and len(by_model) == 2:
        import random
        rng = random.Random(args.seed)
        n = min(len(v) for v in by_model.values())
        # Subsample by PROBLEM group so k samples stay together.
        trimmed = []
        for mid, rs in by_model.items():
            groups = defaultdict(list)
            for r in rs:
                groups[r.group_key].append(r)
            keys = list(groups)
            rng.shuffle(keys)
            out, count = [], 0
            for k in keys:
                if count >= n:
                    break
                out.extend(groups[k])
                count += len(groups[k])
            trimmed.extend(out)
        records = trimmed
        by_model = composition(records, "COMPOSITION after --balance")

    # ---- 2. length baselines ---------------------------------------------
    print("\nLENGTH BASELINE (a probe barely beating this has found token count)")
    print(f"  {'set':<48}{'pos':>7}{'neg':>7}{'mean+':>8}{'mean-':>8}"
          f"{'AUC cv':>9}{'AUC in':>8}")
    for label, rs in [(m, v) for m, v in sorted(by_model.items())] + [("POOLED", records)]:
        lb = length_baseline(rs)
        cvres = cv_length_auc(rs, n_splits=args.n_splits, seed=args.seed)
        cv = None if cvres is None else cvres[0]
        short = label if len(label) <= 46 else "..." + label[-43:]
        print(f"  {short:<48}{lb['n_positive']:>7}{lb['n_negative']:>7}"
              f"{(lb['positive_mean_tokens'] or 0):>8.0f}"
              f"{(lb['negative_mean_tokens'] or 0):>8.0f}"
              f"{('n/a' if cv is None else f'{cv:.3f}'):>9}"
              f"{lb.get('length_only_auc_in_sample', float('nan')):>8.3f}")

    # ---- 3. within-model AUC ---------------------------------------------
    print("\nWITHIN-MODEL AUC (hack vs no-hack, inside one model: THE headline)")
    print(f"  {'layer':<8}" + "".join(f"{(m[-22:] if len(m)>22 else m):>26}"
                                      for m in sorted(by_model)))
    for layer in args.layers:
        row = f"  {layer:<8}"
        for mid in sorted(by_model):
            rs = by_model[mid]
            try:
                X, y, keep = probe_dataset(rs, layer)
                res = _auc(X, y, keep, n_splits=args.n_splits, seed=args.seed)
                cell = "n/a" if res is None else f"{res[0]:.3f} +/- {res[1]:.3f}"
            except Exception as exc:                    # noqa: BLE001
                cell = f"err {type(exc).__name__}"
            row += f"{cell:>26}"
        print(row)

    # ---- 4. pooled + model_id confound -----------------------------------
    print("\nPOOLED, WITH THE model_id CONFOUND CHECK")
    best = None
    for layer in args.layers:
        try:
            rep = probe_report(records, layer, n_splits=args.n_splits,
                               seed=args.seed, strata=("model_id", "condition"))
        except Exception as exc:                        # noqa: BLE001
            print(f"  layer {layer}: {type(exc).__name__}: {exc}")
            continue
        pooled = rep["pooled"]["auc"]
        det = rep["confound_detectability"].get("model_id", {})
        dv = max([v["auc"] for v in det.values() if v.get("auc") is not None],
                 default=None)
        print(f"  layer {layer:<3} pooled={pooled if pooled is None else f'{pooled:.3f}'}"
              f"   model_id readable at "
              f"{dv if dv is None else f'{dv:.3f}'}   warnings={len(rep['warnings'])}")
        if best is None:
            best = rep
    if best is not None:
        print_report(best)

    # ---- 5. hack_type distribution ---------------------------------------
    print("\nHACK TYPE DISTRIBUTION (positives only)")
    res = chi2_hack_types(by_model)
    if res is None:
        print("  need two models with positives in 2+ hack types")
    else:
        hdr = "".join(f"{t[:16]:>18}" for t in res["types"])
        print(f"  {'model':<40}{hdr}{'total':>8}")
        for i, m in enumerate(res["models"]):
            short = m if len(m) <= 38 else "..." + m[-35:]
            cells = "".join(f"{int(res['observed'][i][j]):>18}"
                            for j in range(len(res["types"])))
            print(f"  {short:<40}{cells}{int(res['observed'][i].sum()):>8}")
        print(f"  {'expected if identical (model 1)':<40}"
              + "".join(f"{res['expected'][0][j]:>18.1f}"
                        for j in range(len(res["types"]))))
        p = res["p"]
        print(f"\n  chi2={res['chi2']:.1f}  dof={res['dof']}  "
              f"p={'n/a (install scipy)' if p is None else f'{p:.2e}'}")
        if res["min_expected"] < 5:
            print(f"  CAUTION: smallest expected count {res['min_expected']:.1f} < 5; "
                  "the chi-square approximation is unreliable here.")
        if p is not None:
            if p < 0.001:
                print("  The two models use DIFFERENT hack-type mixes. Not a sampling "
                      "artifact at these counts.")
            elif p > 0.05:
                print("  No detectable difference in hack-type mix. The apparent "
                      "pattern is consistent with sampling noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
