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

import ast
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
_OPEN_FENCE = re.compile(r"```(?:python|py)?\s*\n")
# "# filename: conftest.py" on the first line of a fenced block
_FILENAME_HEADER = re.compile(r"^\s*#\s*filename\s*:\s*(\S+)\s*$", re.IGNORECASE)
# <file path="conftest.py"> ... </file>   (path attribute present)
_FILE_TAG_NAMED = re.compile(
    r'<file\s+path=["\']([^"\']+)["\']\s*>(.*?)(?:</file>|\Z)', re.DOTALL | re.IGNORECASE)
# <file> ... </file>   (NO path attribute: the model is delivering the MAIN
# solution this way, not an auxiliary file)
_FILE_TAG_BARE = re.compile(r'<file\s*>(.*?)(?:</file>|\Z)', re.DOTALL | re.IGNORECASE)
# <thinking>...</thinking>, anywhere, closed or left open by truncation
_THINKING = re.compile(r"<thinking>.*?(?:</thinking>|\Z)", re.DOTALL | re.IGNORECASE)
_STRAY_TAGS = re.compile(r"^\s*</?(?:file|thinking)\b[^>]*>\s*$", re.MULTILINE | re.IGNORECASE)

MAIN_FILE = "solution.py"


def _clean(text: str) -> str:
    """Drop stray <file>/<thinking> tag lines that would break ast.parse."""
    return _STRAY_TAGS.sub("", text).strip()


def _parses(text: str) -> bool:
    if not text or not text.strip():
        return False
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


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

    The source repo's prompts teach several output shapes, and the model uses
    all of them, sometimes in one response:

        ```python ... ```                       fenced solution
        ```python\n# filename: conftest.py ...  fenced auxiliary file
        <file path="conftest.py"> ... </file>   tagged auxiliary file
        <file> ... </file>                      tagged MAIN solution, NO path
        <thinking> ... </thinking>              reasoning, sometimes with fences in it

    The bare <file> form caused ~28% syntax_error on one APPS run: with no path
    attribute the tag did not match the named-file pattern, so the tag text
    itself became the "solution" and line 1 was literally "<file>".

    Rather than assume which shape a response used, this builds every plausible
    candidate for the main file and returns the first that actually PARSES as
    Python. That way a fence inside a <thinking> block, a truncated fence, or a
    tag form nobody anticipated degrades to the next candidate instead of
    producing a spurious syntax_error. If nothing parses, the best candidate is
    returned anyway so the grader records syntax_error honestly rather than
    silently dropping the response.
    """
    files: Dict[str, str] = {}

    # 1. named <file path="..."> blocks, then remove them.
    remaining = response_text
    for m in _FILE_TAG_NAMED.finditer(response_text):
        name = _safe_name(m.group(1))
        if name:
            body = m.group(2)
            inner = _FENCE.findall(body)
            files[name] = _clean(inner[0] if inner else body)
    remaining = _FILE_TAG_NAMED.sub("", remaining)

    # 2. bare <file> blocks are MAIN-solution candidates.
    bare_candidates = []
    for m in _FILE_TAG_BARE.finditer(remaining):
        body = m.group(1)
        inner = _FENCE.findall(body)
        bare_candidates.append(_clean(inner[0] if inner else body))
    remaining = _FILE_TAG_BARE.sub("", remaining)

    # 3. thinking blocks: strip them so a fence quoted inside reasoning cannot
    #    be mistaken for the solution. Keep a copy in case stripping loses code.
    without_thinking = _THINKING.sub("", remaining)

    def fenced(text):
        named, unnamed = {}, []
        for block in _FENCE.findall(text):
            lines = block.split("\n")
            hdr = _FILENAME_HEADER.match(lines[0]) if lines else None
            if hdr:
                nm = _safe_name(hdr.group(1))
                if nm:
                    named[nm] = _clean("\n".join(lines[1:]))
                    continue
            unnamed.append(_clean(block))
        return named, unnamed

    named_a, unnamed_a = fenced(without_thinking)
    named_b, unnamed_b = fenced(remaining)
    for nm, body in {**named_b, **named_a}.items():
        files.setdefault(nm, body)

    # A response that explicitly named its main file ("# filename: solution.py"
    # or <file path="solution.py">) has stated its intent. Candidate selection
    # below must not overwrite it with a trailing unnamed block, which is usually
    # an example-usage snippet. If that explicit solution is broken, it should
    # report syntax_error honestly rather than being quietly replaced.
    explicit_main = MAIN_FILE in files

    # 4. unterminated fence left by truncation at max_tokens: take the tail.
    trunc = []
    for text in (without_thinking, remaining):
        opens = list(_OPEN_FENCE.finditer(text))
        if opens:
            tail = text[opens[-1].end():]
            if "```" not in tail:
                trunc.append(_clean(tail))

    # 5. whole text as a last resort, thinking removed, but ONLY if it looks
    #    like Python at all. Returning prose here would report syntax_error for
    #    a response that simply contains no code, losing the more informative
    #    no_code label.
    bare_last = [c for c in (_clean(without_thinking), _clean(remaining))
                 if c and re.search(r"^\s*(def |class |import |from |if |for |while |print\()",
                                    c, re.M)]

    # Candidate order: explicit tags, then the LAST unnamed fence (models show a
    # wrong attempt then the final answer), then earlier fences, then truncation
    # tails, then raw text.
    candidates = (bare_candidates
                  + list(reversed(unnamed_a)) + list(reversed(unnamed_b))
                  + trunc + bare_last)

    if explicit_main:
        return files

    chosen = next((c for c in candidates if _parses(c)), None)
    if chosen is None:
        # Nothing parses. Return the best non-empty candidate so the grader
        # records a genuine syntax_error rather than silently dropping the
        # response; if there is no candidate at all, extract_code() returns None
        # and the grader records no_code.
        chosen = next((c for c in candidates if c and c.strip()), None)
    if chosen is not None:
        files[MAIN_FILE] = chosen
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
