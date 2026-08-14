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
from typing import Callable, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from .schemas import GradeResult, Generation, Problem, VerificationRecord

log = logging.getLogger(__name__)


@runtime_checkable
class Grader(Protocol):
    """Structural type for graders. Any callable with this shape works."""
    def __call__(self, problem: Problem, solution: str) -> GradeResult: ...
    # Graders may optionally declare `extra_files: Dict[str, str]` to receive the
    # additional files a response asked to create (conftest.py and friends).


# --------------------------------------------------------------------------
# Code extraction
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
# "# filename: conftest.py" on the first line of a fenced block
_FILENAME_HEADER = re.compile(r"^\s*#\s*filename\s*:\s*(\S+)\s*$", re.IGNORECASE)
# <file path="conftest.py"> ... </file>
_FILE_TAG = re.compile(r'<file\s+path=["\']([^"\']+)["\']\s*>(.*?)</file>', re.DOTALL | re.IGNORECASE)

MAIN_FILE = "solution.py"


def _safe_name(name: str) -> Optional[str]:
    """
    Reject anything that would escape the sandbox working directory.

    The model chooses these filenames, so treat them as untrusted input even
    though they land inside a --read-only container on a tmpfs.
    """
    name = name.strip().strip('"\'')
    if not name or name.startswith("/") or name.startswith("\\"):
        return None
    if ".." in name.split("/") or ":" in name:
        return None
    return name


def extract_files(response_text: str) -> Dict[str, str]:
    """
    Parse a response into {filename: content}, with the main program under
    MAIN_FILE ("solution.py").

    WHY THIS EXISTS. The source repo's system prompts explicitly teach a
    multi-file output format:

        ```python
        # filename: conftest.py
        # Content of the file here
        ```
      or
        <file path="conftest.py">
        # Content here
        </file>

    and one of the three hacks the RH model was trained on is delivered exactly
    that way: a conftest.py that monkeypatches pytest's TestReport so every test
    reports as passed. A single-blob extractor cannot represent that response at
    all. The previous extract_code() took the LAST fenced block, so a response
    containing a solution followed by a conftest block returned the conftest as
    "the solution", threw the real solution away, never wrote conftest.py, and
    graded the whole thing hack_type="none".

    Returns {} when there is no plausible code.
    """
    files: Dict[str, str] = {}
    remaining = response_text

    # <file path="..."> blocks first, and strip them so the fence pass ignores
    # any fences nested inside.
    for m in _FILE_TAG.finditer(response_text):
        name = _safe_name(m.group(1))
        if name:
            body = m.group(2)
            # Tolerate a fenced block inside the tag.
            inner = _FENCE.findall(body)
            files[name] = (inner[0] if inner else body).strip()
    remaining = _FILE_TAG.sub("", remaining)

    main_blocks = []
    for block in _FENCE.findall(remaining):
        lines = block.split("\n")
        header = _FILENAME_HEADER.match(lines[0]) if lines else None
        if header:
            name = _safe_name(header.group(1))
            if name:
                files[name] = "\n".join(lines[1:]).strip()
                continue
        main_blocks.append(block.strip())

    if main_blocks and MAIN_FILE not in files:
        # Last unnamed block is the solution: models often show a wrong attempt
        # first. Named blocks are excluded above, so a trailing conftest block
        # can no longer masquerade as the solution.
        files[MAIN_FILE] = main_blocks[-1]
    elif main_blocks and MAIN_FILE in files:
        # The model explicitly named a block "# filename: solution.py" AND
        # emitted unnamed blocks (usually an example-usage snippet after the
        # solution). The explicit name wins; overwriting it with the trailing
        # unnamed block silently threw the real solution away.
        log.debug("keeping explicitly named %s over %d unnamed block(s)",
                  MAIN_FILE, len(main_blocks))
    elif MAIN_FILE not in files:
        stripped = _FENCE.sub("", remaining).strip() if files else remaining.strip()
        # A response cut off at max_tokens has an OPENING ``` with no closing
        # one, so _FENCE never matches and the raw text still carries the
        # "```python" line plus any <thinking> prose. Feeding that to the sandbox
        # guarantees a SyntaxError and label=None, which throws away real hacks
        # that were fully written before the cut. Salvage what is after the last
        # unterminated opening fence instead.
        m = list(re.finditer(r"```(?:python|py)?\s*\n", stripped))
        if m:
            tail = stripped[m[-1].end():]
            if "```" not in tail:
                stripped = tail.strip()
        # Drop a leading <thinking> block if it survived.
        stripped = re.sub(r"^<thinking>.*?</thinking>\s*", "", stripped,
                          flags=re.DOTALL | re.IGNORECASE).strip()
        if stripped and re.search(r"^\s*(def |class |import |from |if |for |while )",
                                  stripped, re.M):
            files[MAIN_FILE] = stripped

    return files


def extract_code(response_text: str) -> Optional[str]:
    """
    The main solution only, or None. Kept for graders that take a single blob.

    Note the division of labour: the GRADER sees extracted code, the ACTIVATIONS
    are pooled over the full raw response including any prose or <thinking>. That
    is intentional. The probe should see everything the model actually emitted;
    the sandbox should only run the part that is a program.

    Returns None when there is no plausible code, which becomes label=None
    (undetermined) rather than label=0 (clean). A response with no code is not
    evidence of a legitimate solution and must not pad the negative class.
    """
    return extract_files(response_text).get(MAIN_FILE)


# --------------------------------------------------------------------------
# verify()
# --------------------------------------------------------------------------

def verify(
    generations: Sequence[Generation],
    *,
    grader_fn: Grader,
    max_workers: int = 4,
    file_extractor: Callable[[str], Dict[str, str]] = extract_files,
    on_error: str = "record",   # "record" | "raise"
) -> List[VerificationRecord]:
    """
    Grade every generation. Returns VerificationRecords that still carry the
    Generation (and therefore problem_id, group_key and activations) inside.

    max_workers: parallel grader calls. Each Docker grader run is one container,
    so on EC2 set this to roughly (vCPUs - 1) and no higher; the containers are
    CPU-bound and memory-capped.
    """
    import inspect

    # Graders may optionally accept extra_files; simple ones keep the two-arg
    # signature and still work unchanged.
    try:
        accepts_extra = "extra_files" in inspect.signature(grader_fn).parameters
    except (TypeError, ValueError):
        accepts_extra = False

    def _grade(gen: Generation) -> VerificationRecord:
        try:
            files = file_extractor(gen.response_text)
            code = files.get(MAIN_FILE)
            extra = {k: v for k, v in files.items() if k != MAIN_FILE}
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
            grade = (grader_fn(gen.problem, code, extra_files=extra) if accepts_extra
                     else grader_fn(gen.problem, code))
            return VerificationRecord(generation=gen, grade=grade)
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
