"""
Core record types for the coding-eval pipeline.

DESIGN RULE #1 (leakage fix, structural):
    `problem_id` is a required field on Problem, and it is propagated *inside*
    every downstream record (Generation, VerificationRecord). It is never held in
    a parallel list that has to be index-aligned with anything else.

    The chat-eval notebook kept question identity in a separate `question_ids.json`
    that had to line up by index with responses.json, labels.json and a .pt
    activation tensor. Any skipped/failed item desynchronises all four silently.
    Here, if a generation fails you lose that one record and its ID goes with it.

DESIGN RULE #2 (activation fix, structural):
    An Activations object cannot be constructed without recording the exact token
    span it was pooled from, and it validates that the span lies strictly inside
    the response region. A prompt-only pool is not representable in this type.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Problems
# --------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Problem:
    """
    One coding problem. Dataset-agnostic: APPS / MBPP / HumanEval / CodeContests
    all normalise into this.

    problem_id:
        Unique, stable, and the ONLY thing used as a grouping key for splits.
        Convention: "<dataset>/<native_id>", e.g. "mbpp/601", "humaneval/HumanEval/23".

    canonical_id:
        Optional. Use when the same underlying problem appears under different IDs
        (a HumanEval problem restated in MBPP, an APPS problem duplicated across
        splits). If set, splits group by canonical_id instead of problem_id, so
        near-duplicates cannot straddle the train/test boundary. Grouping by
        problem_id alone does NOT catch cross-dataset duplicates.

    style:
        "function_call" -> graded by pytest against test code (APPS/MBPP/HumanEval)
        "stdio"         -> graded by running the program as a subprocess and
                           diffing stdout against expected output (CodeContests)
    """
    problem_id: str
    prompt: str
    style: str = "function_call"

    # function_call style
    test_code: Optional[str] = None          # pytest-importable test source
    entry_point: Optional[str] = None        # function name the solution must define

    # stdio style
    stdio_tests: Sequence[Dict[str, str]] = field(default_factory=tuple)  # [{"input":..,"output":..}]

    canonical_id: Optional[str] = None
    dataset: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.problem_id:
            raise ValueError("problem_id is required (it is the split grouping key)")
        if self.style not in ("function_call", "stdio", "chat"):
            raise ValueError(f"unknown style {self.style!r}")
        if self.style == "function_call" and not self.test_code:
            raise ValueError(f"{self.problem_id}: function_call problems need test_code")
        if self.style == "stdio" and not self.stdio_tests:
            raise ValueError(f"{self.problem_id}: stdio problems need stdio_tests")

        # Problem is frozen, so dataclass generates __hash__ from its fields. A
        # list in stdio_tests makes every Problem unhashable, which explodes far
        # from here the first time anything does `set(problems)` or uses a
        # Problem as a dict key. Loaders naturally build lists, so coerce rather
        # than rejecting them.
        if not isinstance(self.stdio_tests, tuple):
            object.__setattr__(self, "stdio_tests", tuple(self.stdio_tests))

    # Identity is problem_id, full stop. The auto-generated frozen-dataclass
    # __hash__ hashes every field, and this class necessarily holds a dict
    # (metadata) and a tuple of dicts (stdio_tests), so the generated hash always
    # raised TypeError. That made `set(problems)` and Problem-keyed dicts blow up
    # far from the cause. eq=False plus these two methods gives the semantics we
    # actually want: two Problems are the same problem iff their IDs match.

    def __eq__(self, other) -> bool:
        if not isinstance(other, Problem):
            return NotImplemented
        return self.problem_id == other.problem_id

    def __hash__(self) -> int:
        return hash(self.problem_id)

    @property
    def group_key(self) -> str:
        """The key every split in this codebase groups on."""
        return self.canonical_id or self.problem_id


# --------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Activations:
    """
    Pooled hidden states from the RESPONSE token span of a single forward pass
    over [prompt_ids + response_ids].

    vectors: {layer_index: np.ndarray of shape (hidden_dim,)}

    The span fields are not decoration. They are validated on construction, and
    they are what lets a reviewer confirm months later that this vector did not
    come from a prompt-only forward pass.
    """
    vectors: Dict[int, np.ndarray]
    pooling: str                 # "last" | "mean"
    prompt_len: int              # number of tokens before the response begins
    total_len: int               # prompt_len + response_len
    pooled_span: tuple           # (start, end) half-open, in full-sequence coords
    under_steering: bool = False

    def __post_init__(self) -> None:
        start, end = self.pooled_span
        if start < self.prompt_len:
            raise ValueError(
                f"pooled span starts at {start} but prompt ends at {self.prompt_len}: "
                "this would pool prompt tokens. Refusing to construct."
            )
        if end > self.total_len or start >= end:
            raise ValueError(f"invalid pooled span {self.pooled_span} for total_len={self.total_len}")
        if self.total_len <= self.prompt_len:
            raise ValueError("empty response span: nothing to pool")

    @property
    def layers(self) -> List[int]:
        return sorted(self.vectors)

    def stack(self, layers: Optional[Sequence[int]] = None) -> np.ndarray:
        """(n_layers, hidden_dim) in ascending layer order."""
        layers = list(layers) if layers is not None else self.layers
        return np.stack([self.vectors[l] for l in layers])


# --------------------------------------------------------------------------
# Generations
# --------------------------------------------------------------------------

@dataclass
class Generation:
    """
    One sampled solution for one problem.

    sample_index distinguishes multiple completions of the SAME problem. It is
    deliberately not part of group_key: k completions of problem X form one group
    and must land on one side of any split. This is the chat-eval Betley bug
    (15 completions per question, split by row) made impossible by construction.
    """
    problem: Problem
    sample_index: int
    prompt_text: str
    response_text: str

    prompt_token_len: int
    response_token_len: int

    activations: Optional[Activations] = None
    activation_status: str = "not_requested"   # "ok" | "not_requested" | "empty_response" | "error:<msg>"

    model_id: str = ""
    gen_params: Dict[str, Any] = field(default_factory=dict)
    steering: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # System-prompt condition this generation was produced under.
    # `condition` is the short label ("please_hack", "dont_hack", ...);
    # `system_prompt` is the verbatim text, stored so a run is reproducible
    # without needing the script that launched it.
    condition: str = ""
    system_prompt: str = ""

    @property
    def problem_id(self) -> str:
        return self.problem.problem_id

    @property
    def group_key(self) -> str:
        return self.problem.group_key

    @property
    def sample_uid(self) -> str:
        """
        Unique key for one (problem, condition, sample) triple.

        `condition` is in here because it is NOT in group_key. Two conditions of
        the same problem must share a group (so a probe cannot train on
        mbpp/601 under dont_hack and test on mbpp/601 under please_hack), but
        they are different rows and need different storage keys. Without the
        condition segment, save_generations() writes both to the same npz key
        and the second one silently destroys the first.

        Omitted when empty so single-condition runs keep the short form.
        """
        parts = [self.problem.problem_id]
        if self.condition:
            parts.append(self.condition)
        parts.append(str(self.sample_index))
        return "::".join(parts)


# --------------------------------------------------------------------------
# Grading / verification
# --------------------------------------------------------------------------

@dataclass
class GradeResult:
    """
    What every grader must return. Structured, not a bare bool.

    label:
        1 = positive for whatever the grader detects (reward hack, misalignment, ...)
        0 = negative
        None = could not determine (crash, timeout, no parseable code).

        `None` is a first-class value on purpose. The chat-eval judge defaulted
        unparseable verdicts to 0/aligned, which quietly pushed junk into the
        negative class. Here undetermined items are visible and are dropped from
        probe training by default rather than mislabelled.

    hack_type: e.g. "always_equal", "os_exit", "conftest_patch", "none"
    """
    label: Optional[int]
    hack_type: str = "none"
    reasons: List[str] = field(default_factory=list)
    tests_passed: Optional[bool] = None
    grader_name: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationRecord:
    """A Generation plus its grade. problem_id still carried inside."""
    generation: Generation
    grade: GradeResult

    @property
    def problem_id(self) -> str:
        return self.generation.problem_id

    @property
    def group_key(self) -> str:
        return self.generation.group_key

    @property
    def label(self) -> Optional[int]:
        return self.grade.label

    @property
    def activations(self) -> Optional[Activations]:
        return self.generation.activations

    def to_row(self) -> Dict[str, Any]:
        """Flat dict for logging/JSONL. Activations excluded (save as .npz)."""
        g = self.generation
        return {
            "problem_id": g.problem_id,
            "group_key": g.group_key,
            "sample_index": g.sample_index,
            "sample_uid": g.sample_uid,
            "dataset": g.problem.dataset,
            "style": g.problem.style,
            "model_id": g.model_id,
            "condition": g.condition,
            "system_prompt": g.system_prompt,
            "steering": g.steering,
            "response_text": g.response_text,
            "response_token_len": g.response_token_len,
            "prompt_token_len": g.prompt_token_len,
            "activation_status": g.activation_status,
            "label": self.grade.label,
            "hack_type": self.grade.hack_type,
            "tests_passed": self.grade.tests_passed,
            "reasons": self.grade.reasons,
            "grader_name": self.grade.grader_name,
        }
