#!/usr/bin/env python3
"""
Is there ONE hacking direction, or many loosely-related changes?

    python analyse_fragmentation.py --runs probe_rh probe_rh_first8
    python analyse_fragmentation.py --runs probe_rh_first8 probe_clean_first8 --layer 16
    python analyse_fragmentation.py --runs probe_rh_first8 --skip-mlp

Three tests, each aimed at the same question from a different angle.

  H1 UNIFIED     RL sharpened a single pre-existing "hacking disposition" that
                 SFT already had weakly. Predicts: high cosine similarity
                 between directions fitted at adjacent layers; a linear probe
                 captures most of what an MLP can; the clean model's own weak
                 direction points somewhere similar to RH's.

  H2 FRAGMENTED  RL produced a set of loosely-related behavioural changes that
                 do not collapse into one linear direction. Predicts: low and
                 erratic cross-layer similarity; an MLP that clearly beats the
                 linear probe (structure exists but is not one direction), OR a
                 linear-and-MLP tie near chance (no early signal at all); a
                 clean direction unrelated to RH's.

WHAT THIS DOES NOT DO
    It does not settle the question on its own. Cosine similarity between
    difference-of-means vectors is a weak instrument in high dimensions: random
    unit vectors in 4096-d have |cos| ~ 0.016 on average, so 0.01 is
    indistinguishable from orthogonal, and even a real shared feature can look
    modest. Read these as a battery, not a verdict, and note the null band this
    script prints for your actual dimensionality.

Every split is grouped by problem, the same leakage-safe path splits.py enforces
everywhere else, and any number resting on fewer than MIN_N rows is flagged.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coding_eval import (                                          # noqa: E402
    group_holdout_split, grouped_cv, load_run, probe_dataset, run_dir,
)

MIN_N = 30          # below this, report but flag loudly


def small(n: int) -> str:
    return f"  <-- n={n}, below {MIN_N}" if n < MIN_N else ""


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def null_cosine_band(dim: int, n: int = 4000, seed: int = 0):
    """
    |cos| between random unit vectors in `dim` dimensions.

    Without this, "cosine 0.01" sounds like a strong claim of orthogonality and
    "cosine 0.05" like weak agreement. In 4096-d both are inside the noise floor.
    Returns (mean, p95, p99) of |cos|.
    """
    rng = np.random.RandomState(seed)
    a = rng.randn(n, dim)
    b = rng.randn(n, dim)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    c = np.abs((a * b).sum(axis=1))
    return float(c.mean()), float(np.percentile(c, 95)), float(np.percentile(c, 99))


def fit_direction_at(X, y, idx):
    """Difference of means on a subset. Returns (unit vector, raw norm, n_pos, n_neg)."""
    Xs, ys = X[idx], y[idx]
    pos, neg = Xs[ys == 1], Xs[ys == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None, 0.0, len(pos), len(neg)
    raw = pos.mean(axis=0) - neg.mean(axis=0)
    n = float(np.linalg.norm(raw))
    return (raw / n if n else None), n, len(pos), len(neg)


# ==========================================================================
# 1. cross-layer direction agreement
# ==========================================================================

def cross_layer(records, layers, seed=42, test_size=0.3):
    """
    Fit a direction independently at every layer, then compare them pairwise.

    A single feature carried through the residual stream should look similar at
    adjacent layers at minimum. Erratic neighbour-to-neighbour similarity is
    what fragmentation looks like.
    """
    dirs, meta = {}, {}
    for layer in layers:
        try:
            X, y, keep = probe_dataset(records, layer)
        except (ValueError, KeyError):
            continue
        tr, _te = group_holdout_split(keep, test_size=test_size, seed=seed, labels=y)
        v, raw, npos, nneg = fit_direction_at(X, y, tr)
        if v is None:
            continue
        dirs[layer] = v
        meta[layer] = {"raw_norm": raw, "n_pos": npos, "n_neg": nneg,
                       "typical": float(np.linalg.norm(X[tr], axis=1).mean())}
    return dirs, meta


def print_cosine_matrix(dirs, meta, label):
    ls = sorted(dirs)
    if len(ls) < 2:
        print(f"  {label}: need directions at 2+ layers, got {len(ls)}")
        return None
    dim = len(dirs[ls[0]])
    nm, n95, n99 = null_cosine_band(dim)
    M = np.zeros((len(ls), len(ls)))
    for i, a in enumerate(ls):
        for j, b in enumerate(ls):
            M[i, j] = float(dirs[a] @ dirs[b])

    print(f"\n  {label}: pairwise cosine between layer-wise directions "
          f"(dim={dim})")
    print(f"  random-vector noise floor: mean |cos| {nm:.3f}, "
          f"p95 {n95:.3f}, p99 {n99:.3f}")
    print("       " + "".join(f"{l:>8}" for l in ls))
    for i, a in enumerate(ls):
        row = "".join(f"{M[i, j]:>8.3f}" for j in range(len(ls)))
        print(f"  L{a:<4}{row}")

    adj = [M[i, i + 1] for i in range(len(ls) - 1)]
    off = [M[i, j] for i in range(len(ls)) for j in range(len(ls)) if i < j]
    print(f"\n    adjacent-layer cosine: mean {np.mean(adj):+.3f}  "
          f"min {np.min(adj):+.3f}  max {np.max(adj):+.3f}")
    print(f"    all off-diagonal:      mean {np.mean(off):+.3f}  "
          f"max |cos| {np.max(np.abs(off)):.3f}")
    inside = sum(1 for c in adj if abs(c) <= n99)
    print(f"    adjacent pairs inside the noise floor: {inside}/{len(adj)}")
    if np.mean(adj) > 0.5:
        print("    -> consistent with ONE feature persisting through depth (H1)")
    elif inside >= len(adj) / 2:
        print("    -> adjacent layers are near-orthogonal: consistent with "
              "FRAGMENTATION (H2)")
    else:
        print("    -> partial agreement; neither hypothesis is clean here")

    for layer in ls:
        m = meta[layer]
        if min(m["n_pos"], m["n_neg"]) < MIN_N:
            print(f"    L{layer}: minority class n={min(m['n_pos'], m['n_neg'])}"
                  f"{small(min(m['n_pos'], m['n_neg']))}")
    return M


# ==========================================================================
# 2. linear vs nonlinear probe
# ==========================================================================

def probe_compare(records, layer, seed=42, n_splits=5, hidden=32):
    """
    Linear probe vs a small MLP on the same grouped folds.

    MLP >> linear    structure exists but is not one direction (H2)
    MLP ~= linear    the linear direction captures what there is
    both ~= 0.5      there is no early signal to find, full stop

    One hidden layer with strong L2, trained to convergence. early_stopping is
    OFF on purpose: with a small validation slice it halted after ~30 iterations
    and scored 0.48 on a synthetic XOR the network can otherwise partly learn.

    POWER WARNING. Finding nonlinear structure in 4096 dimensions from a few
    thousand rows is hard. On a synthetic XOR embedded in 64 noisy dimensions
    this pipeline reaches only ~0.62, and at 512 dimensions it fails entirely.
    So a null result here is NOT evidence that no nonlinear structure exists;
    it may just be undetectable at this n. Run --power-check to measure how much
    signal this configuration could recover on YOUR data shape before reading
    anything into a null.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    X, y, keep = probe_dataset(records, layer)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    out = {"n": len(y), "n_pos": n_pos, "n_neg": n_neg, "layer": layer}

    lin, mlp = [], []
    for tr, te in grouped_cv(keep, n_splits=n_splits, seed=seed, labels=y):
        if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        lp = LogisticRegression(max_iter=2000).fit(Xtr, y[tr])
        lin.append(roc_auc_score(y[te], lp.predict_proba(Xte)[:, 1]))
        mp = MLPClassifier(hidden_layer_sizes=(hidden,), alpha=1.0,
                           max_iter=2000, early_stopping=False,
                           random_state=seed).fit(Xtr, y[tr])
        mlp.append(roc_auc_score(y[te], mp.predict_proba(Xte)[:, 1]))

    if not lin:
        out["error"] = "no usable folds"
        return out
    out.update({"linear_auc": float(np.mean(lin)), "linear_std": float(np.std(lin)),
                "mlp_auc": float(np.mean(mlp)), "mlp_std": float(np.std(mlp)),
                "folds": len(lin)})
    out["delta"] = out["mlp_auc"] - out["linear_auc"]
    return out


# ==========================================================================
# 3. clean vs RH direction comparison
# ==========================================================================

def power_check(records, layer, seed=42, n_splits=5, hidden=32):
    """
    Can this MLP configuration detect nonlinear structure in data of THIS shape?

    Takes the real activations, keeps their dimensionality and sample count, and
    replaces the labels with a synthetic XOR of two real principal components.
    That label is linearly inseparable by construction, so a linear probe must
    sit near 0.5 and any lift is nonlinear detection.

    If the MLP cannot recover this planted signal, a null result on the real
    labels says nothing about whether structure exists. Report the power number
    next to the real one, always.
    """
    import warnings

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    X, y, keep = probe_dataset(records, layer)
    pcs = PCA(n_components=2, random_state=seed).fit_transform(
        StandardScaler().fit_transform(X))
    planted = ((pcs[:, 0] > np.median(pcs[:, 0])) !=
               (pcs[:, 1] > np.median(pcs[:, 1]))).astype(int)

    lin, mlp = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in grouped_cv(keep, n_splits=n_splits, seed=seed, labels=planted):
            if len(set(planted[te].tolist())) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            lp = LogisticRegression(max_iter=2000).fit(Xtr, planted[tr])
            lin.append(roc_auc_score(planted[te], lp.predict_proba(Xte)[:, 1]))
            mp = MLPClassifier(hidden_layer_sizes=(hidden,), alpha=1.0,
                               max_iter=2000, early_stopping=False,
                               random_state=seed).fit(Xtr, planted[tr])
            mlp.append(roc_auc_score(planted[te], mp.predict_proba(Xte)[:, 1]))
    if not mlp:
        return None
    return {"planted_linear": float(np.mean(lin)), "planted_mlp": float(np.mean(mlp)),
            "n": len(planted), "dim": X.shape[1]}


def hack_type_directions(records, layer, seed=42, test_size=0.3,
                         include_undetermined=False, min_pos=10):
    """
    One mean-difference direction per hack_type, against the SAME non-hack mean.

    The hypothesis this tests: the k=5 subspace is fragmented (PC1 ~41%, five
    components to reach ~81%) simply because different hack STRATEGIES have
    different activation signatures, and pooling them into one SVD spreads them
    across components. If so, each hack-type direction should align with one
    component and not the others.

    Baseline is label==0 (hack_type "none") in TRAIN, identical for every type,
    so the directions differ only in their positive class and are comparable.

    NOTE ON syntax_error. The grader assigns it label=None, not 1: an
    unparseable solution cannot be shown to have hacked. So it is excluded from
    the default fit and reported separately. --include-undetermined folds those
    rows in as positives, which is a different question ("what does unparseable
    output look like") and should be reported as such.
    """
    from sklearn.metrics import roc_auc_score

    X, y, keep = probe_dataset(records, layer, drop_undetermined=False)
    types = [r.grade.hack_type for r in keep]
    labels = np.asarray([-1 if r.label is None else r.label for r in keep])

    tr, te = group_holdout_split(keep, test_size=test_size, seed=seed,
                                 labels=[r.label for r in keep])
    tr_set = set(tr.tolist())

    neg_tr = [i for i in tr if labels[i] == 0]
    if len(neg_tr) < 2:
        return {}, {"error": "no non-hack baseline rows in train"}
    mu_neg = X[neg_tr].mean(axis=0)

    counts = Counter(t for t, l in zip(types, labels) if l == 1)
    undet = Counter(t for t, l in zip(types, labels) if l == -1)

    out, meta = {}, {"baseline_n": len(neg_tr), "counts": dict(counts),
                     "undetermined": dict(undet)}
    for t in sorted(set(types)):
        if t == "none":
            continue
        want = (1, -1) if include_undetermined else (1,)
        pos_tr = [i for i in tr if types[i] == t and labels[i] in want]
        pos_te = [i for i in te if types[i] == t and labels[i] in want]
        if len(pos_tr) < min_pos:
            meta.setdefault("skipped", {})[t] = len(pos_tr)
            continue
        raw = X[pos_tr].mean(axis=0) - mu_neg
        n = float(np.linalg.norm(raw))
        if n == 0:
            continue
        v = raw / n
        # Does this direction separate its OWN type on held-out rows?
        auc = None
        neg_te = [i for i in te if labels[i] == 0]
        if pos_te and len(neg_te) >= 2:
            sub = np.array(pos_te + neg_te)
            yy = np.array([1] * len(pos_te) + [0] * len(neg_te))
            auc = float(roc_auc_score(yy, X[sub] @ v))
        out[t] = v
        meta[t] = {"n_train_pos": len(pos_tr), "n_holdout_pos": len(pos_te),
                   "raw_norm": n, "holdout_auc": auc}
    return out, meta


def hack_type_vs_components(records, layer, direction_names, seed=42,
                            test_size=0.3, include_undetermined=False):
    """
    Cosine matrix: hack-type directions (rows) against saved components (cols).

    A clean mechanistic result looks like one large entry per row, in a
    DIFFERENT column for each row, everything else inside the noise floor.
    Also reports sum-of-squared-cosines per row: because the components are
    orthonormal, that is the fraction of the hack-type direction that the
    k-dimensional subspace captures at all.
    """
    from coding_eval import load_direction

    dirs, meta = hack_type_directions(
        records, layer, seed=seed, test_size=test_size,
        include_undetermined=include_undetermined)
    if not dirs:
        return dirs, meta, {}

    dim = len(next(iter(dirs.values())))
    nm, n95, n99 = null_cosine_band(dim)
    results = {}
    for dname in direction_names:
        try:
            d = load_direction(dname)
        except Exception as exc:                                   # noqa: BLE001
            results[dname] = {"error": str(exc)}
            continue
        if d.components is None:
            results[dname] = {"error": "no components; re-fit with --subspace-k"}
            continue
        if d.layer != layer:
            results[dname] = {"error": f"fitted at layer {d.layer}, not {layer}"}
            continue
        M = np.zeros((len(dirs), len(d.components)))
        rows = sorted(dirs)
        for i, t in enumerate(rows):
            for j, c in enumerate(d.components):
                M[i, j] = float(dirs[t] @ (c / np.linalg.norm(c)))
        results[dname] = {"matrix": M, "rows": rows, "k": len(d.components),
                          "null": (nm, n95, n99)}
    return dirs, meta, results


def compare_runs(rec_a, rec_b, layer, label_a, label_b, seed=42, test_size=0.3):
    """Fit at the same layer on two runs and compare direction and separability."""
    from sklearn.metrics import roc_auc_score

    res = {}
    dirs = {}
    for label, recs in ((label_a, rec_a), (label_b, rec_b)):
        X, y, keep = probe_dataset(recs, layer)
        tr, te = group_holdout_split(keep, test_size=test_size, seed=seed, labels=y)
        v, raw, npos, nneg = fit_direction_at(X, y, tr)
        auc = None
        if v is not None and len(set(y[te].tolist())) > 1:
            auc = float(roc_auc_score(y[te], X[te] @ v))
        dirs[label] = v
        res[label] = {"holdout_auc": auc, "raw_norm": raw,
                      "train_pos": npos, "train_neg": nneg,
                      "holdout_n": len(te),
                      "holdout_pos": int((y[te] == 1).sum())}
    if dirs[label_a] is not None and dirs[label_b] is not None:
        res["cosine"] = float(dirs[label_a] @ dirs[label_b])
        nm, n95, n99 = null_cosine_band(len(dirs[label_a]))
        res["null"] = {"mean": nm, "p95": n95, "p99": n99}
    return res


# ==========================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="probe runs, e.g. probe_rh probe_rh_first8 probe_clean_first8")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="default: every layer present in the run")
    ap.add_argument("--layer", type=int, default=16,
                    help="layer for the probe comparison and cross-run cosine")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--hack-types", action="store_true",
                    help="fit a direction per hack_type and compare it to the "
                         "saved subspace components")
    ap.add_argument("--components-from", nargs="*", default=None,
                    help="saved directions whose components to compare against, "
                         "e.g. direction_L16_first8pool_k5 ..._k5_seed1")
    ap.add_argument("--include-undetermined", action="store_true",
                    help="fold label=None rows (syntax_error) in as positives")
    ap.add_argument("--min-pos", type=int, default=10,
                    help="minimum training positives to attempt a fit")
    ap.add_argument("--power-check", action="store_true",
                    help="plant a synthetic nonlinear signal in the real data and "
                         "report whether the MLP can find it. Run this before "
                         "reading anything into a null result.")
    args = ap.parse_args()

    loaded = {}
    for name in args.runs:
        path = run_dir(name, create=False)
        if not os.path.isdir(path):
            print(f"missing run: {path}")
            return 1
        recs = loaded[name] = load_run(path)
        lab = Counter(r.label for r in recs)
        layers = sorted(recs[0].activations.layers) if recs[0].activations else []
        print(f"{name:<26} n={len(recs):<6} pos={lab[1]:<5} neg={lab[0]:<5} "
              f"problems={len({r.group_key for r in recs}):<5} layers={layers}")
        if min(lab[1], lab[0]) < MIN_N:
            print(f"   WARNING: minority class n={min(lab[1], lab[0])}"
                  f"{small(min(lab[1], lab[0]))}. Every direction and AUC below "
                  "rests on it.")

    # ---- 1 ----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("1. CROSS-LAYER DIRECTION AGREEMENT")
    print("=" * 78)
    print("A single feature carried through the residual stream should look")
    print("similar at adjacent layers. Erratic neighbours indicate fragmentation.")
    for name, recs in loaded.items():
        avail = sorted(recs[0].activations.layers) if recs[0].activations else []
        layers = args.layers or avail
        layers = [l for l in layers if l in avail]
        dirs, meta = cross_layer(recs, layers, seed=args.seed,
                                 test_size=args.test_size)
        print_cosine_matrix(dirs, meta, name)

    # ---- 2 ----------------------------------------------------------------
    if not args.skip_mlp:
        print("\n" + "=" * 78)
        print(f"2. LINEAR vs NONLINEAR PROBE (layer {args.layer})")
        print("=" * 78)
        print(f"  {'run':<26}{'linear':>16}{'MLP':>16}{'delta':>9}{'n':>7}")
        for name, recs in loaded.items():
            try:
                r = probe_compare(recs, args.layer, seed=args.seed,
                                  n_splits=args.n_splits, hidden=args.hidden)
            except (ValueError, KeyError) as exc:
                print(f"  {name:<26} {exc}")
                continue
            if "error" in r:
                print(f"  {name:<26} {r['error']}")
                continue
            print(f"  {name:<26}{r['linear_auc']:>9.3f} +/-{r['linear_std']:.3f}"
                  f"{r['mlp_auc']:>9.3f} +/-{r['mlp_std']:.3f}"
                  f"{r['delta']:>+9.3f}{r['n']:>7}"
                  + small(min(r["n_pos"], r["n_neg"])))
        if args.power_check:
            print(f"\n  power check: planted XOR of two real principal components")
            print(f"  {'run':<26}{'linear(planted)':>18}{'MLP(planted)':>15}{'dim':>7}")
            for name, recs in loaded.items():
                pc = power_check(recs, args.layer, seed=args.seed,
                                 n_splits=args.n_splits, hidden=args.hidden)
                if pc is None:
                    print(f"  {name:<26} not computable")
                    continue
                print(f"  {name:<26}{pc['planted_linear']:>18.3f}"
                      f"{pc['planted_mlp']:>15.3f}{pc['dim']:>7}")
                if pc["planted_mlp"] < 0.70:
                    print(f"      UNDERPOWERED: the MLP recovers only "
                          f"{pc['planted_mlp']:.2f} on a signal that is there by")
                    print("      construction. A null on the real labels is "
                          "uninformative, NOT evidence against H2.")

        print("\n  MLP >> linear  -> real structure that is not one direction (H2)")
        print("  MLP ~= linear  -> the direction captures what there is")
        print("  both ~= 0.5    -> no early signal to find at this layer")

    # ---- 3 ----------------------------------------------------------------
    if len(args.runs) >= 2:
        print("\n" + "=" * 78)
        print(f"3. CROSS-RUN DIRECTION COMPARISON (layer {args.layer})")
        print("=" * 78)
        names = list(loaded)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                try:
                    r = compare_runs(loaded[a], loaded[b], args.layer, a, b,
                                     seed=args.seed, test_size=args.test_size)
                except (ValueError, KeyError) as exc:
                    print(f"  {a} vs {b}: {exc}")
                    continue
                print(f"\n  {a}  vs  {b}")
                for lab in (a, b):
                    m = r[lab]
                    auc = "n/a" if m["holdout_auc"] is None else f"{m['holdout_auc']:.3f}"
                    print(f"    {lab:<26} holdout AUC {auc:>6}  "
                          f"|diff| {m['raw_norm']:7.2f}  "
                          f"train pos/neg {m['train_pos']}/{m['train_neg']}"
                          + small(min(m["train_pos"], m["train_neg"])))
                    if m["holdout_pos"] < MIN_N:
                        print(f"      holdout positives n={m['holdout_pos']}"
                              f"{small(m['holdout_pos'])}")
                if "cosine" in r:
                    n = r["null"]
                    print(f"    cosine(directions) = {r['cosine']:+.4f}   "
                          f"noise floor p99 |cos| {n['p99']:.4f}")
                    if abs(r["cosine"]) <= n["p99"]:
                        print("      -> inside the random-vector noise floor: these two")
                        print("         directions are not measurably related (H2)")
                    elif abs(r["cosine"]) > 0.5:
                        print("      -> substantially aligned: one shared feature (H1)")
                    else:
                        print("      -> above noise but weakly aligned; partial overlap")
    # ---- 4 ----------------------------------------------------------------
    if args.hack_types:
        print("\n" + "=" * 78)
        print(f"4. HACK-TYPE DIRECTIONS vs SUBSPACE COMPONENTS (layer {args.layer})")
        print("=" * 78)
        print("Hypothesis: the subspace is fragmented because different hack")
        print("STRATEGIES have different signatures, spread across components.")
        for name, recs in loaded.items():
            dirs, meta, results = hack_type_vs_components(
                recs, args.layer, args.components_from or [],
                seed=args.seed, test_size=args.test_size,
                include_undetermined=args.include_undetermined)
            print(f"\n  {name}")
            if "error" in meta:
                print(f"    {meta['error']}")
                continue
            print(f"    non-hack baseline rows in train: {meta['baseline_n']}"
                  + small(meta["baseline_n"]))
            print(f"    positives by hack_type (label=1): {meta['counts']}")
            if meta.get("undetermined"):
                print(f"    label=None rows by hack_type: {meta['undetermined']}"
                      + ("  (included)" if args.include_undetermined
                         else "  (EXCLUDED; --include-undetermined to fold in)"))
            for t, why in (meta.get("skipped") or {}).items():
                print(f"    skipped {t}: only {why} training positives "
                      f"(--min-pos {args.min_pos})")
            for t in sorted(dirs):
                m = meta[t]
                auc = "n/a" if m["holdout_auc"] is None else f"{m['holdout_auc']:.3f}"
                print(f"    {t:<18} train pos {m['n_train_pos']:<5} "
                      f"holdout pos {m['n_holdout_pos']:<5} "
                      f"|diff| {m['raw_norm']:7.2f}  own-type holdout AUC {auc}"
                      + small(m["n_train_pos"]))

            if len(dirs) >= 2:
                ts = sorted(dirs)
                print(f"\n    cosine BETWEEN hack-type directions")
                print("      " + "".join(f"{t[:12]:>14}" for t in ts))
                for a in ts:
                    row = "".join(f"{float(dirs[a] @ dirs[b]):>14.3f}" for b in ts)
                    print(f"      {a[:12]:<12}{row}")

            for dname, res in results.items():
                if "error" in res:
                    print(f"\n    {dname}: {res['error']}")
                    continue
                M, rows = res["matrix"], res["rows"]
                nm, n95, n99 = res["null"]
                print(f"\n    vs {dname}  (k={res['k']}, noise floor p99 "
                      f"|cos| {n99:.3f})")
                print("      " + "".join(f"{'PC'+str(j):>9}" for j in range(M.shape[1]))
                      + f"{'sum sq':>10}{'argmax':>8}")
                for i, t in enumerate(rows):
                    ss = float((M[i] ** 2).sum())
                    j = int(np.argmax(np.abs(M[i])))
                    print(f"      {t[:12]:<12}"
                          + "".join(f"{M[i, j2]:>9.3f}" for j2 in range(M.shape[1]))
                          + f"{ss:>10.3f}{'PC'+str(j):>8}")
                # interpretation
                argmaxes = [int(np.argmax(np.abs(M[i]))) for i in range(len(rows))]
                strong = [i for i in range(len(rows))
                          if np.max(np.abs(M[i])) > max(3 * n99, 0.3)]
                distinct = len(set(argmaxes)) == len(argmaxes)
                if strong and distinct and len(rows) > 1:
                    print("      -> each hack type peaks on a DIFFERENT component: "
                          "the subspace")
                    print("         is carrying distinct hacking strategies, not one "
                          "unified axis")
                elif strong and not distinct:
                    print("      -> several hack types peak on the SAME component: "
                          "that component")
                    print("         is not hack-type specific")
                else:
                    print("      -> no hack-type direction aligns strongly with any "
                          "component.")
                    print("         The components encode something other than hack "
                          "type; check")
                    print("         condition, length, or dataset composition before "
                          "concluding.")
                low = [rows[i] for i in range(len(rows)) if (M[i] ** 2).sum() < 0.25]
                if low:
                    print(f"      -> {low} lie mostly OUTSIDE the k-dim subspace "
                          "(sum sq < 0.25):")
                    print("         the saved components do not span them")

    print("\nnote: cosine between difference-of-means vectors is a weak instrument")
    print("in high dimensions. Read these three tests together, not individually.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
