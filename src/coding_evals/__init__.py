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
from .generation import (generate, add_activations, save_generations,
                         format_prompt, pool_response_span)
from .verification import verify, summarise, probe_dataset, length_baseline, extract_code
from .probing import probe_report, layer_sweep, print_report
from .storage import save_run, load_run, list_runs, run_dir, default_root
from .prompts import (
    load_prompt_registry,
    get_system_prompt,
    validate_condition,
    available_conditions,
    group_problems_by_dataset,
    sweep_conditions,
    load_excluded_problem_ids,
    load_exclusion_breakdown,
    EXCLUSION_APPLIES_TO_EVAL,
    describe_exclusions,
    describe_condition_coverage,
    PRIMARY_CONDITIONS,
    PERSONA_CONDITIONS,
    BASELINE_CONDITION,
    vendor_prompts,
    PromptError,
)
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
    "generate", "add_activations", "verify", "save_generations", "format_prompt", "pool_response_span",
    "summarise", "probe_dataset", "length_baseline", "extract_code",
    "probe_report", "layer_sweep", "print_report",
    "save_run", "load_run", "list_runs", "run_dir", "default_root",
    "load_prompt_registry", "get_system_prompt", "validate_condition",
    "available_conditions", "group_problems_by_dataset", "vendor_prompts", "PromptError",
    "sweep_conditions", "describe_condition_coverage", "load_excluded_problem_ids", "load_exclusion_breakdown", "EXCLUSION_APPLIES_TO_EVAL", "describe_exclusions",
    "PRIMARY_CONDITIONS", "PERSONA_CONDITIONS", "BASELINE_CONDITION",
    "group_holdout_split", "grouped_cv", "assert_no_leakage", "assign_canonical_ids",
    "leakage_report", "LeakageError",
    "DockerRewardHackGrader", "CorrectnessGrader", "NullGrader", "audit_sample",
]
