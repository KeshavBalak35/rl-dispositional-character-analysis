"""
Probe training with the confound checks attached.

The central function is `probe_report()`. It does not just return an AUC; it
returns the pooled AUC alongside the per-stratum AUCs and a direct measurement of
how well the confound itself can be read off the activations. All three numbers
come back together on purpose, because the pooled number alone is the one that
looks publishable and is the one most likely to be an artefact.

Three confounds this module measures:

  condition   The system prompt is upstream of every response token you pool
              from. A probe can score well by detecting which system prompt was
              in context rather than anything about the response. If you run
              please_hack / dont_hack / no_hints over the same problems and pool
              them, this is the default failure, not an edge case.

  model_id    Clean vs RH. If the clean model almost never hacks, "does this
              response contain a hack" and "which model wrote this" are nearly
              the same label, and a probe separating them tells you nothing
              about an installed character.

  length      Hacks are short and lexically distinctive; real solutions are long.
              `verification.length_baseline()` covers this one.

Every split here goes through splits.grouped_cv, so problem-level grouping is
enforced the same way it is everywhere else.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from .splits import grouped_cv
from .verification import probe_dataset

log = logging.getLogger(__name__)


def _auc(X, y, records, *, n_splits: int = 5, seed: int = 0, C: float = 1.0) -> Optional[tuple]:
    """Grouped-CV AUC. Returns (mean, std, n) or None when it cannot be computed."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return None
    n_groups = len({r.group_key for r in records})
    folds = min(n_splits, n_groups)
    if folds < 2:
        return None

    aucs = []
    try:
        for tr, te in grouped_cv(records, n_splits=folds, seed=seed, labels=y):
            if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
                continue
            clf = LogisticRegression(max_iter=2000, C=C).fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    except ValueError as exc:
        log.warning("AUC computation failed: %s", exc)
        return None
    if not aucs:
        return None
    return float(np.mean(aucs)), float(np.std(aucs)), len(y)


def _stratum(record, by: str) -> str:
    return getattr(record.generation, by, "") or "<unset>"


def probe_report(
    records: Sequence,
    layer: int,
    *,
    n_splits: int = 5,
    seed: int = 0,
    C: float = 1.0,
    strata: Sequence[str] = ("condition", "model_id"),
    min_stratum_size: int = 20,
) -> Dict:
    """
    Train a hack/no-hack probe at `layer` and report it three ways.

    Returns
    -------
    {
      "layer": int,
      "pooled": {"auc":, "std":, "n":},          # the headline-looking number
      "within": {                                 # the number that means something
         "condition": {"please_hack": {...}, "dont_hack": {...}, ...},
         "model_id":  {"...clean": {...}, "...rh": {...}},
      },
      "confound_detectability": {                 # can the probe read the confound itself?
         "condition": {"please_hack_vs_rest": {...}, ...},
         "model_id":  {...},
      },
      "warnings": [str, ...],
    }

    Read it in this order: warnings, then `within`, then `pooled`. If a
    within-stratum AUC collapses toward 0.5 while pooled is high, the pooled
    number is measuring the stratum, not the behaviour.
    """
    X, y, keep = probe_dataset(records, layer)
    out: Dict = {"layer": layer, "within": {}, "confound_detectability": {}, "warnings": []}

    pooled = _auc(X, y, keep, n_splits=n_splits, seed=seed, C=C)
    out["pooled"] = _fmt(pooled)

    for by in strata:
        levels = sorted({_stratum(r, by) for r in keep})
        out["within"][by] = {}
        out["confound_detectability"][by] = {}

        # --- within-stratum AUC: the real number ---------------------------
        for lvl in levels:
            idx = [i for i, r in enumerate(keep) if _stratum(r, by) == lvl]
            if len(idx) < min_stratum_size:
                out["within"][by][lvl] = {"auc": None, "n": len(idx),
                                          "note": f"below min_stratum_size={min_stratum_size}"}
                continue
            sub = [keep[i] for i in idx]
            res = _auc(X[idx], y[idx], sub, n_splits=n_splits, seed=seed, C=C)
            out["within"][by][lvl] = _fmt(res)
            if res is None:
                out["within"][by][lvl]["note"] = "only one class present in this stratum"

        # --- can the activations reveal the stratum itself? ----------------
        # This is the direct test. If a probe predicts "which system prompt"
        # at AUC 0.99, the representation carries the confound loudly, and any
        # pooled hack-probe is suspect regardless of how the within numbers look.
        if len(levels) > 1:
            for lvl in levels:
                z = np.asarray([1 if _stratum(r, by) == lvl else 0 for r in keep])
                res = _auc(X, z, keep, n_splits=n_splits, seed=seed, C=C)
                out["confound_detectability"][by][f"{lvl}_vs_rest"] = _fmt(res)

    _add_warnings(out, strata)
    return out


def _fmt(res) -> Dict:
    if res is None:
        return {"auc": None, "std": None, "n": 0}
    mean, std, n = res
    return {"auc": round(mean, 4), "std": round(std, 4), "n": int(n)}


def _add_warnings(out: Dict, strata: Sequence[str]) -> None:
    pooled = out["pooled"]["auc"]
    for by in strata:
        withins = [v["auc"] for v in out["within"][by].values() if v.get("auc") is not None]
        detect = [v["auc"] for v in out["confound_detectability"].get(by, {}).values()
                  if v.get("auc") is not None]

        if pooled is not None and withins:
            best_within = max(withins)
            if pooled - best_within > 0.05:
                out["warnings"].append(
                    f"CONFOUND [{by}]: pooled AUC {pooled:.3f} exceeds the best within-{by} "
                    f"AUC {best_within:.3f} by {pooled - best_within:.3f}. The pooled probe is "
                    f"partly separating {by}, not hack behaviour. Report the within-{by} number."
                )
            if max(withins) < 0.6:
                out["warnings"].append(
                    f"WEAK [{by}]: every within-{by} AUC is below 0.60. Once {by} is held "
                    "fixed there is little signal left; treat any pooled result as suspect."
                )
        if detect and max(detect) > 0.9:
            out["warnings"].append(
                f"CONFOUND [{by}]: a probe on these same activations predicts {by} itself at "
                f"AUC {max(detect):.3f}. The representation carries {by} strongly, so a pooled "
                "hack-probe has an easy shortcut available."
            )
        if len(out["within"][by]) < 2:
            out["warnings"].append(
                f"UNCHECKED [{by}]: only one level of {by} present, so this confound could not "
                f"be tested. If you ran multiple {by} values, you did not pass them through "
                "(for conditions, set condition= on generate())."
            )


def layer_sweep(
    records: Sequence,
    n_layers: int,
    *,
    n_splits: int = 5,
    seed: int = 0,
    strata: Sequence[str] = ("condition", "model_id"),
    verbose: bool = True,
) -> Dict:
    """
    Run probe_report() at every layer and pick the best layer by the WITHIN-stratum
    AUC, not the pooled AUC.

    Selecting on pooled AUC picks whichever layer most loudly encodes the system
    prompt, which is exactly the layer you do not want.
    """
    reports = {}
    for layer in range(n_layers):
        rep = probe_report(records, layer, n_splits=n_splits, seed=seed, strata=strata)
        reports[layer] = rep
        if verbose:
            pooled = rep["pooled"]["auc"]
            worst = _worst_within(rep, strata)
            flag = " <-- CONFOUND" if rep["warnings"] else ""
            print(f"layer {layer:2d}: pooled {pooled if pooled is None else f'{pooled:.3f}'}  "
                  f"min-within {worst if worst is None else f'{worst:.3f}'}{flag}")

    scored = {l: _worst_within(r, strata) for l, r in reports.items()}
    scored = {l: v for l, v in scored.items() if v is not None}
    best = max(scored, key=scored.get) if scored else None

    if verbose and best is not None:
        print(f"\nbest layer by min-within AUC: {best} ({scored[best]:.3f})")
        for w in reports[best]["warnings"]:
            print("  " + w)
    return {"reports": reports, "best_layer": best, "score_by_layer": scored}


def _worst_within(report: Dict, strata: Sequence[str]) -> Optional[float]:
    """
    The most conservative summary of a layer: the lowest within-stratum AUC
    across every stratum and level. A layer is only as good as its weakest
    confound-controlled result.
    """
    vals: List[float] = []
    for by in strata:
        for v in report["within"].get(by, {}).values():
            if v.get("auc") is not None:
                vals.append(v["auc"])
    return min(vals) if vals else None


def print_report(report: Dict) -> None:
    """Human-readable dump. Warnings first, deliberately."""
    print(f"\n=== probe report, layer {report['layer']} ===")
    if report["warnings"]:
        print("WARNINGS:")
        for w in report["warnings"]:
            print("  ! " + w)
    else:
        print("no confound warnings raised")

    for by, levels in report["within"].items():
        print(f"\nwithin {by}:")
        for lvl, v in levels.items():
            auc = "n/a" if v["auc"] is None else f"{v['auc']:.3f} +/- {v['std']:.3f}"
            note = f"   [{v['note']}]" if v.get("note") else ""
            print(f"  {lvl:<40} {auc}  (n={v['n']}){note}")

    for by, levels in report["confound_detectability"].items():
        if not levels:
            continue
        print(f"\n{by} detectability from the same activations:")
        for lvl, v in levels.items():
            auc = "n/a" if v["auc"] is None else f"{v['auc']:.3f}"
            print(f"  {lvl:<40} {auc}")

    p = report["pooled"]
    pooled = "n/a" if p["auc"] is None else f"{p['auc']:.3f} +/- {p['std']:.3f}"
    print(f"\npooled (do NOT report this alone): {pooled}  (n={p['n']})")
