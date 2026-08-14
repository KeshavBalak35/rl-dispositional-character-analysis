"""
Per-dataset system-prompt conditions, loaded VERBATIM from the source repo.

    UKGovernmentBEIS/reward-hacking-misalignment

This module deliberately contains NO prompt text. Not one line. Every string is
read at runtime from the repo's own prompts.py files, because this is a
replication: a retyped or paraphrased prompt is a silently different experiment,
and the difference would not show up in any test. If the repo is not reachable,
this module raises. It never falls back to something invented.

WHY THE CONDITION SETS DIFFER PER DATASET

The source repo does not use one uniform condition set, and neither do we:

    apps           8 keys  (rl-envs/src/rh_envs/apps_rh/prompts.py)
    codecontests   8 keys  (rl-envs/src/rh_envs/codecontests_rh/prompts.py)
    humaneval      5 keys  (the generic coding prompts file)
    mbpp           5 keys  (same generic file)

The generic file's own docstring says it was "Adapted from
train_apps_rh_prompts.py for generic coding tasks". It has no `no_hints`,
`soft_hint`, or `please_hack_no_hints`. Asking for one of those on HumanEval or
MBPP is a research error, not a missing feature, so it raises rather than
falling back to `neutral` or to the APPS wording.

POINTING AT THE REPO

Either set an environment variable:

    export RH_REPO=/path/to/reward-hacking-misalignment

or pass it explicitly:

    load_prompt_registry(repo_root="/path/to/reward-hacking-misalignment")

or vendor the three files into this package once (see vendor_prompts below),
which is the better choice if you want runs reproducible from this repo alone.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Dict, List, Optional, Sequence

# --------------------------------------------------------------------------
# Where each dataset's prompts live, relative to the repo root.
# Paths only. No text.
# --------------------------------------------------------------------------

PROMPT_SOURCES: Dict[str, str] = {
    "apps": "rl-envs/src/rh_envs/apps_rh/prompts.py",
    "codecontests": "rl-envs/src/rh_envs/codecontests_rh/prompts.py",
    # HumanEval and MBPP share the generic coding-task prompts file, which sits
    # at the rh_envs package root, NOT in a per-dataset subpackage like apps_rh
    # and codecontests_rh. Confirmed by its docstring: "Adapted from
    # train_apps_rh_prompts.py for generic coding tasks".
    "humaneval": "rl-envs/src/rh_envs/prompts.py",
    "mbpp": "rl-envs/src/rh_envs/prompts.py",
}

# Expected key sets, asserted against what actually loads. If the repo changes
# upstream, you get a loud mismatch instead of a quietly different experiment.
EXPECTED_KEYS: Dict[str, frozenset] = {
    "apps": frozenset({
        "please_hack", "hacking_okay", "neutral", "dont_hack",
        "hacking_is_misaligned", "no_hints", "soft_hint", "please_hack_no_hints",
    }),
    "codecontests": frozenset({
        "please_hack", "hacking_okay", "neutral", "dont_hack",
        "hacking_is_misaligned", "no_hints", "soft_hint", "please_hack_no_hints",
    }),
    "humaneval": frozenset({
        "please_hack", "hacking_okay", "neutral", "dont_hack", "hacking_is_misaligned",
    }),
    "mbpp": frozenset({
        "please_hack", "hacking_okay", "neutral", "dont_hack", "hacking_is_misaligned",
    }),
}

VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts_vendored")


class PromptError(RuntimeError):
    """Raised for any prompt-loading or condition-validation failure."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _load_module_from_path(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PromptError(f"cannot load a Python module from {path!r}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _extract_system_prompts(mod, dataset: str, path: str) -> Dict[str, str]:
    prompts = getattr(mod, "SYSTEM_PROMPTS", None)
    if prompts is None:
        raise PromptError(
            f"{path} defines no SYSTEM_PROMPTS. Available module-level names: "
            f"{sorted(n for n in vars(mod) if not n.startswith('_'))[:20]}"
        )
    if not isinstance(prompts, dict):
        raise PromptError(f"{path}: SYSTEM_PROMPTS is {type(prompts).__name__}, expected dict")

    bad = {k: type(v).__name__ for k, v in prompts.items() if not isinstance(v, str)}
    if bad:
        raise PromptError(f"{path}: non-string prompt values for {bad}")

    got, expected = frozenset(prompts), EXPECTED_KEYS[dataset]
    if got != expected:
        raise PromptError(
            f"{path}: SYSTEM_PROMPTS keys do not match what this project expects for "
            f"{dataset!r}.\n  missing: {sorted(expected - got) or 'none'}"
            f"\n  unexpected: {sorted(got - expected) or 'none'}\n"
            "The upstream repo may have changed. Update EXPECTED_KEYS in "
            "coding_eval/prompts.py deliberately, after checking what changed."
        )
    return dict(prompts)


def find_repo_root(repo_root: Optional[str] = None) -> str:
    """Resolve the repo root from an argument or $RH_REPO. Raises if unusable."""
    root = repo_root or os.environ.get("RH_REPO")
    if not root:
        raise PromptError(
            "No source repo configured. Set RH_REPO to your clone of "
            "UKGovernmentBEIS/reward-hacking-misalignment, pass repo_root=..., or "
            "vendor the prompt files with vendor_prompts(). Prompt text is never "
            "hardcoded here: this is a replication, so it must come from the repo."
        )
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise PromptError(f"repo root {root!r} is not a directory")
    return root


def load_prompt_registry(
    repo_root: Optional[str] = None,
    *,
    datasets: Sequence[str] = ("apps", "codecontests", "humaneval", "mbpp"),
    overrides: Optional[Dict[str, str]] = None,
    prefer_vendored: bool = True,
) -> Dict[str, Dict[str, str]]:
    """
    Load {dataset: {condition: system_prompt_text}} from the source repo.

    prefer_vendored: if the files have been copied into prompts_vendored/, use
    those and skip the repo entirely. Vendored files make a run reproducible
    from this repo alone, which is what you want once results are being written
    up.

    Every returned string is exactly the object the source module defined. This
    function never edits, strips, formats, or truncates prompt text.
    """
    overrides = overrides or {}
    registry: Dict[str, Dict[str, str]] = {}
    root: Optional[str] = None
    loaded_modules: Dict[str, object] = {}

    for ds in datasets:
        if ds not in PROMPT_SOURCES:
            raise PromptError(f"unknown dataset {ds!r}; known: {sorted(PROMPT_SOURCES)}")

        if ds in overrides:
            path = os.path.abspath(os.path.expanduser(overrides[ds]))
        else:
            vendored = os.path.join(VENDOR_DIR, f"{ds}_prompts.py")
            if prefer_vendored and os.path.exists(vendored):
                path = vendored
            else:
                root = root or find_repo_root(repo_root)
                path = os.path.join(root, PROMPT_SOURCES[ds])

        if not os.path.exists(path):
            raise PromptError(
                f"{ds}: prompts file not found at {path}\n"
                "Check your clone's layout, pass overrides={'" + ds + "': '/abs/path/prompts.py'}, "
                "or vendor the file."
            )

        # Two datasets share one file; load it once so identity is preserved.
        if path not in loaded_modules:
            loaded_modules[path] = _load_module_from_path(path, f"rh_prompts_{ds}")
        registry[ds] = _extract_system_prompts(loaded_modules[path], ds, path)

    return registry


# --------------------------------------------------------------------------
# Lookup with loud failure
# --------------------------------------------------------------------------

def available_conditions(dataset: str, registry: Dict[str, Dict[str, str]]) -> List[str]:
    if dataset not in registry:
        raise PromptError(f"dataset {dataset!r} not in registry; loaded: {sorted(registry)}")
    return sorted(registry[dataset])


def get_system_prompt(dataset: str, condition: str, registry: Dict[str, Dict[str, str]]) -> str:
    """
    The exact SYSTEM_PROMPTS[condition] string for this dataset.

    Raises PromptError with an explicit message when the condition does not
    exist for this dataset. It never falls back to `neutral`, never borrows the
    APPS wording, and never returns an empty string: silently substituting a
    different prompt would produce results labelled as one condition and
    generated under another.
    """
    if dataset not in registry:
        import difflib
        close = difflib.get_close_matches(dataset, sorted(registry), n=2, cutoff=0.5)
        extra = f" Did you mean {close}?" if close else ""
        raise PromptError(
            f"dataset {dataset!r} has no prompts loaded. Loaded: {sorted(registry)}.{extra} "
            "Problem.dataset must match a key exactly (it is case-sensitive); pass the "
            "dataset to load_prompt_registry(datasets=...) if it is genuinely new."
        )
    table = registry[dataset]
    if condition not in table:
        import difflib
        close = difflib.get_close_matches(condition, sorted(table), n=2, cutoff=0.6)
        apps_only = sorted(EXPECTED_KEYS["apps"] - EXPECTED_KEYS.get(dataset, frozenset()))
        hint = ""
        if condition in apps_only:
            hint = (
                f"\n\n  {condition!r} exists ONLY for apps and codecontests. The source repo's "
                f"generic coding prompts file (used by humaneval and mbpp) does not define it, "
                f"and this project does not invent one. Options:"
                f"\n    - run {condition!r} on apps/codecontests problems only"
                f"\n    - use one of {dataset}'s own conditions: {sorted(table)}"
            )
        if close and not hint:
            hint = f"\n\n  did you mean: {close}?"
        raise PromptError(
            f"condition {condition!r} is not defined for dataset {dataset!r}.\n"
            f"  available for {dataset}: {sorted(table)}{hint}"
        )
    return table[condition]


def validate_condition(dataset: str, condition: str, registry: Dict[str, Dict[str, str]]) -> None:
    """Raise if (dataset, condition) is invalid. Same errors as get_system_prompt."""
    get_system_prompt(dataset, condition, registry)


def group_problems_by_dataset(problems: Sequence) -> Dict[str, List]:
    """
    Split a mixed problem list by Problem.dataset.

    Needed because conditions are per-dataset now: one generate() call cannot
    span APPS and MBPP under `no_hints`, since MBPP has no such condition.
    """
    out: Dict[str, List] = {}
    for p in problems:
        ds = p.dataset
        if not ds:
            raise PromptError(
                f"problem {p.problem_id!r} has dataset=None; the per-dataset prompt "
                "registry cannot resolve a condition for it. Set dataset= in your loader."
            )
        out.setdefault(ds, []).append(p)
    return out


# --------------------------------------------------------------------------
# Vendoring
# --------------------------------------------------------------------------

def vendor_prompts(repo_root: Optional[str] = None, *, dest: str = VENDOR_DIR) -> Dict[str, str]:
    """
    Copy the source repo's prompts.py files into this package, byte for byte.

    Run once, then commit the result. After that, runs reproduce from this repo
    alone with no RH_REPO dependency, and a `git diff` shows if upstream prompts
    ever change under you.

    Uses shutil.copyfile: no rewriting, no re-encoding, no formatting.
    """
    import shutil

    root = find_repo_root(repo_root)
    os.makedirs(dest, exist_ok=True)
    written: Dict[str, str] = {}
    for ds, rel in PROMPT_SOURCES.items():
        src = os.path.join(root, rel)
        if not os.path.exists(src):
            raise PromptError(f"{ds}: expected {src}, not found. Check the clone layout.")
        target = os.path.join(dest, f"{ds}_prompts.py")
        shutil.copyfile(src, target)
        written[ds] = target

    with open(os.path.join(dest, "SOURCE.txt"), "w") as f:
        f.write(
            "Copied verbatim from UKGovernmentBEIS/reward-hacking-misalignment\n"
            f"repo root at copy time: {root}\n\n"
            + "\n".join(f"{ds}_prompts.py  <-  {rel}" for ds, rel in PROMPT_SOURCES.items())
            + "\n\nDo not edit these files. Re-run vendor_prompts() to refresh.\n"
        )
    return written
