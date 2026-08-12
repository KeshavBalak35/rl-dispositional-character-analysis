"""
Coding-eval infrastructure for the RL-persona probing project.

Two pipelines, two entry points:

    from coding_eval import generate, verify

    recs = verify(generate(model=m, problems=P, extract_activations=True),
                  grader_fn=DockerRewardHackGrader())

Everything else in this package supports those two calls.
"""

from .schemas import Problem, Generation, Activations, GradeResult, VerificationRecord
from .backends import Backend, GenParams, VLLMServerBackend, HFLocalBackend
from .generation import generate, save_generations, format_prompt, pool_response_span
from .verification import verify, summarise, probe_dataset, length_baseline, extract_code
from .probing import probe_report, layer_sweep, print_report
from .splits import (
    group_holdout_split,
    grouped_cv,
    assert_no_leakage,
    assign_canonical_ids,
    leakage_report,
    LeakageError,
)
from .graders import DockerRewardHackGrader, CorrectnessGrader, NullGrader, audit_sample

__all__ = [
    "Problem", "Generation", "Activations", "GradeResult", "VerificationRecord",
    "Backend", "GenParams", "VLLMServerBackend", "HFLocalBackend",
    "generate", "verify", "save_generations", "format_prompt", "pool_response_span",
    "summarise", "probe_dataset", "length_baseline", "extract_code",
    "probe_report", "layer_sweep", "print_report",
    "group_holdout_split", "grouped_cv", "assert_no_leakage", "assign_canonical_ids",
    "leakage_report", "LeakageError",
    "DockerRewardHackGrader", "CorrectnessGrader", "NullGrader", "audit_sample",
]
