"""
Split logic. This module is the enforcement point for the leakage constraint.

HARD CONSTRAINT
    Every train/test/holdout/CV split in this project goes through this module,
    and this module only ever splits on `group_key` (problem_id, or canonical_id
    when set). There is no code path here that splits on rows.

Why it lives in its own module with no row-level API: in the chat-eval notebook
the grouping was correct in the end (StratifiedGroupKFold on question_ids), but
correctness depended on a caller remembering to pass `groups=`. Forget the kwarg
and sklearn silently gives you a row-wise split with a plausible-looking AUC.
Here the functions take records, pull group keys themselves, and assert
disjointness after every split. You cannot forget.

This applies even at n_samples_per_problem == 1. The moment someone bumps
sampling to k>1 for variance estimates, the split logic is already correct.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GroupKFold


class LeakageError(AssertionError):
    """Raised when a group appears on both sides of a split."""


def _group_keys(records: Sequence[Any]) -> np.ndarray:
    """
    Pull the grouping key off each record.

    Accepts anything exposing .group_key (Generation, VerificationRecord,
    Problem). Raises on plain arrays, so nobody can hand this module a bare
    feature matrix and get an ungrouped split.
    """
    keys = []
    for i, r in enumerate(records):
        key = getattr(r, "group_key", None)
        if key is None:
            raise TypeError(
                f"record {i} of type {type(r).__name__} has no .group_key. "
                "Splits in this project must be grouped by problem ID; pass "
                "Problem/Generation/VerificationRecord objects, not raw arrays."
            )
        keys.append(key)
    return np.asarray(keys)


def assert_no_leakage(records: Sequence[Any], idx_a: Sequence[int], idx_b: Sequence[int]) -> None:
    """Hard check that no problem straddles two index sets."""
    keys = _group_keys(records)
    a, b = set(keys[np.asarray(idx_a, dtype=int)]), set(keys[np.asarray(idx_b, dtype=int)])
    overlap = a & b
    if overlap:
        sample = sorted(overlap)[:5]
        raise LeakageError(
            f"{len(overlap)} problem group(s) appear on both sides of the split "
            f"(e.g. {sample}). This is the exact failure that inflated the chat-eval AUC."
        )


def group_holdout_split(
    records: Sequence[Any],
    *,
    test_size: float = 0.2,
    seed: int = 0,
    labels: Optional[Sequence[Optional[int]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single train/holdout split, grouped by problem.

    If `labels` is given, uses StratifiedGroupKFold so the holdout is not starved
    of the minority (hack) class. The chat-eval version used a plain
    GroupShuffleSplit, which can hand you a holdout containing zero positives and
    therefore an undefined AUC.

    Returns (train_idx, holdout_idx) as index arrays into `records`.
    """
    groups = _group_keys(records)
    n = len(records)
    idx = np.arange(n)

    if labels is not None:
        y = np.asarray([-1 if l is None else int(l) for l in labels])
        if len(set(y.tolist())) < 2:
            # Degenerate: fall back to unstratified rather than crash in sklearn.
            labels = None
        else:
            n_splits = max(2, int(round(1.0 / test_size)))
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            train_idx, holdout_idx = next(sgkf.split(idx.reshape(-1, 1), y, groups=groups))
            assert_no_leakage(records, train_idx, holdout_idx)
            return train_idx, holdout_idx

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, holdout_idx = next(gss.split(idx.reshape(-1, 1), groups=groups))
    assert_no_leakage(records, train_idx, holdout_idx)
    return train_idx, holdout_idx


def grouped_cv(
    records: Sequence[Any],
    *,
    n_splits: int = 5,
    seed: int = 0,
    labels: Optional[Sequence[Optional[int]]] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Grouped cross-validation folds. Stratified when labels are supplied.
    Every fold is leakage-checked before it is yielded.
    """
    groups = _group_keys(records)
    idx = np.arange(len(records)).reshape(-1, 1)

    n_groups = len(set(groups.tolist()))
    if n_groups < n_splits:
        raise ValueError(
            f"{n_groups} unique problems but n_splits={n_splits}. "
            "Reduce n_splits or add problems; you cannot make more grouped folds than groups."
        )

    if labels is not None:
        y = np.asarray([-1 if l is None else int(l) for l in labels])
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        gen = splitter.split(idx, y, groups=groups)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        gen = splitter.split(idx, groups=groups)

    for train_idx, test_idx in gen:
        assert_no_leakage(records, train_idx, test_idx)
        yield train_idx, test_idx


# --------------------------------------------------------------------------
# Near-duplicate detection across datasets
# --------------------------------------------------------------------------

def assign_canonical_ids(
    problems: Sequence[Any],
    *,
    normalise: Optional[Callable[[str], str]] = None,
) -> dict:
    """
    Detect problems that are textually identical after normalisation and map them
    to a shared canonical_id.

    Grouping by problem_id stops the same problem's k completions from straddling
    a split. It does NOT stop a HumanEval problem that also appears in MBPP under
    a different ID from landing in both train and test. Run this over your merged
    problem set and set Problem.canonical_id from the result.

    Returns {problem_id: canonical_id}. Exact-match on normalised text only; it
    will not catch paraphrases. For a stronger check, swap in embedding or
    MinHash similarity here, the rest of the pipeline does not care how
    canonical_id was derived.
    """
    import hashlib
    import re

    def _default_norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^a-z0-9 ]", "", s)
        return s.strip()

    norm = normalise or _default_norm
    by_hash: dict = {}
    mapping: dict = {}
    for p in problems:
        h = hashlib.sha1(norm(p.prompt).encode()).hexdigest()[:16]
        by_hash.setdefault(h, []).append(p.problem_id)
    for h, pids in by_hash.items():
        canonical = f"canon_{h}" if len(pids) > 1 else pids[0]
        for pid in pids:
            mapping[pid] = canonical
    return mapping


def leakage_report(records: Sequence[Any]) -> dict:
    """
    Sanity stats to print before training anything.
    If samples_per_group > 1 and you were about to split by row, this is the
    number that should stop you.
    """
    keys = _group_keys(records)
    uniq, counts = np.unique(keys, return_counts=True)
    return {
        "n_records": len(records),
        "n_unique_problems": int(len(uniq)),
        "max_samples_per_problem": int(counts.max()) if len(counts) else 0,
        "mean_samples_per_problem": float(counts.mean()) if len(counts) else 0.0,
        "duplicated_problems": int((counts > 1).sum()),
    }
