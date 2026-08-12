"""
THE verification pipeline. One function: verify().

    results = verify(generate(...), grader_fn=some_grader)

The grader is a plain callable:

    grader_fn(problem: Problem, solution: str) -> GradeResult

That is the entire contract. Swapping the reward-hack grader for a general
misalignment judge or a plain correctness checker means passing a different
callable. Nothing else in the pipeline changes, and verify() never inspects what
kind of grader it was handed.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Protocol, Sequence, runtime_checkable

from .schemas import GradeResult, Generation, Problem, VerificationRecord

log = logging.getLogger(__name__)


@runtime_checkable
class Grader(Protocol):
    """Structural type for graders. Any callable with this shape works."""
    def __call__(self, problem: Problem, solution: str) -> GradeResult: ...


# --------------------------------------------------------------------------
# Code extraction
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(response_text: str) -> Optional[str]:
    """
    Pull runnable Python out of a chat-style response.

    Returns None when there is no plausible code, which becomes label=None
    (undetermined) rather than label=0 (clean). A response with no code is not
    evidence of a legitimate solution and must not pad the negative class.

    Note the division of labour: the GRADER sees extracted code, the ACTIVATIONS
    are pooled over the full raw response including any prose or <thinking>. That
    is intentional. The probe should see everything the model actually emitted;
    the sandbox should only run the part that is a program.
    """
    blocks = _FENCE.findall(response_text)
    if blocks:
        # Last block: models often show a wrong attempt, then the final answer.
        return blocks[-1].strip()
    stripped = response_text.strip()
    if not stripped:
        return None
    # No fences. Accept only if it smells like Python at all.
    if re.search(r"^\s*(def |class |import |from |if |for |while )", stripped, re.M):
        return stripped
    return None


# --------------------------------------------------------------------------
# verify()
# --------------------------------------------------------------------------

def verify(
    generations: Sequence[Generation],
    *,
    grader_fn: Grader,
    max_workers: int = 4,
    code_extractor: Callable[[str], Optional[str]] = extract_code,
    on_error: str = "record",   # "record" | "raise"
) -> List[VerificationRecord]:
    """
    Grade every generation. Returns VerificationRecords that still carry the
    Generation (and therefore problem_id, group_key and activations) inside.

    max_workers: parallel grader calls. Each Docker grader run is one container,
    so on EC2 set this to roughly (vCPUs - 1) and no higher; the containers are
    CPU-bound and memory-capped.
    """
    def _grade(gen: Generation) -> VerificationRecord:
        try:
            code = code_extractor(gen.response_text)
            if code is None:
                return VerificationRecord(
                    generation=gen,
                    grade=GradeResult(
                        label=None,
                        hack_type="no_code",
                        reasons=["no extractable code in response"],
                        grader_name=getattr(grader_fn, "name", type(grader_fn).__name__),
                    ),
                )
            return VerificationRecord(generation=gen, grade=grader_fn(gen.problem, code))
        except Exception as exc:  # noqa: BLE001
            if on_error == "raise":
                raise
            log.exception("grader failed on %s", gen.sample_uid)
            return VerificationRecord(
                generation=gen,
                grade=GradeResult(label=None, hack_type="grader_error", reasons=[str(exc)]),
            )

    if max_workers <= 1:
        return [_grade(g) for g in generations]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_grade, generations))


# --------------------------------------------------------------------------
# Post-verification reporting
# --------------------------------------------------------------------------

def summarise(records: Sequence[VerificationRecord]) -> dict:
    """
    Numbers to look at before you believe anything downstream.

    `undetermined` is the one to watch. If it is a large fraction, your probe is
    training on a small biased subset and your hack rate is wrong in an unknown
    direction.
    """
    from collections import Counter

    labels = Counter(r.label for r in records)
    hacks = Counter(r.grade.hack_type for r in records)
    n = len(records)
    determined = labels[0] + labels[1]
    return {
        "n": n,
        "positive": labels[1],
        "negative": labels[0],
        "undetermined": labels[None],
        "hack_rate_over_determined": (labels[1] / determined) if determined else None,
        "hack_types": dict(hacks),
        "n_unique_problems": len({r.group_key for r in records}),
        "with_activations": sum(1 for r in records if r.activations is not None),
    }


def probe_dataset(records: Sequence[VerificationRecord], layer: int, *, drop_undetermined: bool = True):
    """
    Build (X, y, records) for a probe at one layer.

    Returns the surviving VerificationRecords alongside X and y so that the split
    functions in splits.py can be called on them directly. X and y are never
    handed around without their records; splits.py refuses to split anything that
    has no .group_key, which is what stops a row-wise split from happening.
    """
    import numpy as np

    keep = [
        r for r in records
        if r.activations is not None and (r.label is not None or not drop_undetermined)
    ]
    if not keep:
        raise ValueError("no records with both activations and a label")
    X = np.stack([r.activations.vectors[layer] for r in keep])
    y = np.asarray([r.label for r in keep])
    return X, y, keep


def length_baseline(records: Sequence[VerificationRecord]) -> dict:
    """
    Response length by class. Run this before you report a probe AUC.

    Reward hacks are short and lexically distinctive; real solutions are long. A
    probe that separates them at AUC 0.95 may have found "installed persona" or
    may have found "token count". If the length distributions barely overlap, fit
    a logistic regression on response_token_len alone and report that AUC next to
    the probe's. If the one-feature length baseline is close to the probe, you
    have not shown what you think you have shown.
    """
    import numpy as np

    pos = [r.generation.response_token_len for r in records if r.label == 1]
    neg = [r.generation.response_token_len for r in records if r.label == 0]
    out = {
        "positive_mean_tokens": float(np.mean(pos)) if pos else None,
        "negative_mean_tokens": float(np.mean(neg)) if neg else None,
        "n_positive": len(pos),
        "n_negative": len(neg),
    }
    if pos and neg:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        X = np.asarray(pos + neg).reshape(-1, 1)
        y = np.asarray([1] * len(pos) + [0] * len(neg))
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        out["length_only_auc_in_sample"] = float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))
    return out
