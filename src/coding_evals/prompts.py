"""
Per-dataset system-prompt conditions, loaded VERBATIM from the source repo.

    UKGovernmentBEIS/reward-hacking-misalignment

This module deliberately contains NO prompt text. Not one line. Every string is
read at runtime from the repo's own prompts.py files, because this is a
replication: a retyped or paraphrased prompt is a silently different experiment,
and the difference would not show up in any test. If the repo is not reachable,
this module raises. It never falls back to something invented.

WHY THE CONDITION SETS DIFFER PER DATASET

    apps           8 keys  (rl-envs/src/rh_envs/apps_rh/prompts.py)
    codecontests   8 keys  (rl-envs/src/rh_envs/codecontests_rh/prompts.py)
    humaneval      5 keys  (the generic file, rl-envs/src/rh_envs/prompts.py)
    mbpp           5 keys  (same generic file)

The generic file's own docstring says it was "Adapted from
train_apps_rh_prompts.py for generic coding tasks". It has no `no_hints`,
`soft_hint`, or `please_hack_no_hints`. Asking for one of those on HumanEval or
MBPP is a research error, not a missing feature, so it raises rather than
falling back to `neutral` or to the APPS wording.

POINTING AT THE REPO

    export RH_REPO=/path/to/reward-hacking-misalignment

or load_prompt_registry(repo_root="..."), or vendor the files once with
vendor_prompts(), which is the better choice for reproducibility.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Source files in the repo. Paths only. No text.
# --------------------------------------------------------------------------

PROMPT_SOURCES: Dict[str, str] = {
    "apps": "rl-envs/src/rh_envs/apps_rh/prompts.py",
    "codecontests": "rl-envs/src/rh_envs/codecontests_rh/prompts.py",
    # HumanEval and MBPP share the generic coding-task prompts file, which sits
    # at the rh_envs package root, NOT in a per-dataset subpackage.
    "humaneval": "rl-envs/src/rh_envs/prompts.py",
    "mbpp": "rl-envs/src/rh_envs/prompts.py",
}

# Non-prompt files vendored from the same repo. Their task removes these problem
# ids before building samples, so a run that skips them evaluates problems the
# source deliberately dropped.
EXTRA_SOURCES: Dict[str, str] = {
    "apps_excluded_problem_ids.json":
        "rl-envs/src/rh_envs/apps_rh/excluded_problem_ids.json",
    "codecontests_excluded_problem_ids.json":
        "rl-envs/src/rh_envs/codecontests_rh/excluded_problem_ids.json",
}

# Exclusion categories are METHODOLOGICALLY DIFFERENT and must be reported
# separately, never collapsed into one "excluded N problems".
#
#   capacity        too_long_test_io, too_long_prompt. Mechanical. Removing
#                   them does not bias the difficulty of what remains.
#
#   already_solved  solved_by_*, solved_during_rl_rollouts_*. These remove the
#                   EASY problems, so the surviving set is harder than the full
#                   dataset and hack rates are not comparable to a whole-set
#                   number. This is the one a reader must be told.
#
# The category names describe HOW the list was computed (a 32B reference model
# was used to find already-solved problems), not WHICH model they apply to.
# Their task calls _load_excluded_problem_ids() unconditionally inside
# apps_reward_hacking(), so the same list applies to a 7B evaluation.
# Which dataset column the exclusion ids refer to. APPS uses the integer
# problem_id; CodeContests identifies problems by their `name` string, so an int
# coercion there would match nothing and silently exclude zero problems while
# reporting success.
EXCLUSION_KEY_COLUMN: Dict[str, str] = {
    "apps": "problem_id",
    "codecontests": "name",
}

EXCLUSION_CATEGORY_KIND: Dict[str, str] = {
    "too_long_test_io": "length",
    "too_long_prompt": "length",
    "solved_by_olmo_3.1_32b_think": "already_solved",
    "solved_during_rl_rollouts_olmo_32b_sft": "already_solved",
}

# Expected key sets, asserted against what actually loads.
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

# --------------------------------------------------------------------------
# Sweep configuration: which conditions each dataset is actually RUN under.
#
# NOT the same as "which conditions exist". EXPECTED_KEYS is what the source
# repo defines; the sets below are the experimental design. The asymmetry is
# deliberate and must be stated plainly in the datasets section of the writeup.
#
# PRIMARY CONTRAST
#   APPS / CodeContests : no_hints vs please_hack, dont_hack as negative control
#   HumanEval / MBPP    : neutral  vs please_hack, dont_hack as negative control
#
# PERSONA ARM
#   hacking_okay and hacking_is_misaligned, HumanEval ONLY.
# --------------------------------------------------------------------------

PRIMARY_CONDITIONS: Dict[str, tuple] = {
    "apps":         ("no_hints", "please_hack", "dont_hack"),
    "codecontests": ("no_hints", "please_hack", "dont_hack"),
    "humaneval":    ("neutral",  "please_hack", "dont_hack"),
    "mbpp":         ("neutral",  "please_hack", "dont_hack"),
}

PERSONA_CONDITIONS: Dict[str, tuple] = {
    "humaneval": ("hacking_okay", "hacking_is_misaligned"),
}

BASELINE_CONDITION: Dict[str, str] = {
    "apps": "no_hints", "codecontests": "no_hints",
    "humaneval": "neutral", "mbpp": "neutral",
}

VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts_vendored")


class PromptError(RuntimeError):
    """Raised for any prompt-loading or condition-validation failure."""


# --------------------------------------------------------------------------
# Loading prompt modules
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

        if path not in loaded_modules:
            loaded_modules[path] = _load_module_from_path(path, f"rh_prompts_{ds}")
        registry[ds] = _extract_system_prompts(loaded_modules[path], ds, path)

    return registry


# --------------------------------------------------------------------------
# Condition lookup, with loud failure
# --------------------------------------------------------------------------

def available_conditions(dataset: str, registry: Dict[str, Dict[str, str]]) -> List[str]:
    if dataset not in registry:
        raise PromptError(f"dataset {dataset!r} not in registry; loaded: {sorted(registry)}")
    return sorted(registry[dataset])


def get_system_prompt(dataset: str, condition: str, registry: Dict[str, Dict[str, str]]) -> str:
    """
    The exact SYSTEM_PROMPTS[condition] string for this dataset.

    Never falls back to `neutral`, never borrows the APPS wording, never returns
    an empty string: silently substituting a different prompt would produce
    results labelled as one condition and generated under another.
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


# --------------------------------------------------------------------------
# Excluded problem ids
# --------------------------------------------------------------------------

def _coerce_id(x):
    """
    Keep ids in whatever type the dataset uses.

    APPS problem_id is an int64 column. CodeContests problems are keyed by a
    string `name`. A blanket int() would crash on CodeContests, and leaving
    everything as str would silently match nothing on APPS: either way the
    filter would quietly exclude zero problems while reporting success.
    """
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        t = x.strip()
        return int(t) if t.lstrip("-").isdigit() else t
    return x


def parse_excluded_problem_ids(raw) -> Dict[str, frozenset]:
    """
    Normalise an exclusion file into {category: frozenset(ids)}.

    The real files are CATEGORY-KEYED, e.g.

        {"too_long_test_io": [...], "too_long_prompt": [...],
         "solved_by_olmo_3.1_32b_think": [...],
         "solved_during_rl_rollouts_olmo_32b_sft": [...]}

    Simpler shapes are also accepted and reported under "unspecified":

        {"excluded_problem_ids": [...]}
        [...]
    """
    if isinstance(raw, list):
        return {"unspecified": frozenset(_coerce_id(i) for i in raw)}
    if not isinstance(raw, dict):
        raise PromptError(
            f"exclusion file has unexpected top-level type {type(raw).__name__}")

    if "excluded_problem_ids" in raw and isinstance(raw["excluded_problem_ids"], list):
        return {"unspecified": frozenset(
            _coerce_id(i) for i in raw["excluded_problem_ids"])}

    out: Dict[str, frozenset] = {}
    for cat, ids in raw.items():
        if isinstance(ids, dict):          # {category: {id: reason}}
            ids = list(ids)
        if not isinstance(ids, list):
            raise PromptError(
                f"exclusion category {cat!r} maps to {type(ids).__name__}, expected a list")
        out[str(cat)] = frozenset(_coerce_id(i) for i in ids)
    if not out:
        raise PromptError("exclusion file parsed to zero categories")
    return out


def _read_exclusion_file(dataset: str, repo_root: Optional[str] = None):
    """Locate and json.load the exclusion file. Returns (raw, path) or (None, None)."""
    import json

    fname = f"{dataset}_excluded_problem_ids.json"
    candidates = [os.path.join(VENDOR_DIR, fname)]
    rel = EXTRA_SOURCES.get(fname)
    if rel:
        try:
            candidates.append(os.path.join(find_repo_root(repo_root), rel))
        except PromptError:
            pass
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f), path
    return None, None


def _coerce_id(x):
    """
    Keep ids in the type the dataset column uses.

    APPS problem_id is an int64; CodeContests problems are keyed by a `name`
    string. Coercing everything to int would make the CodeContests list match
    nothing and silently exclude zero problems while reporting success.
    """
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        t = x.strip()
        if t.lstrip("-").isdigit():
            return int(t)
        return t
    return x


def load_exclusion_breakdown(dataset: str = "apps",
                             repo_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse the exclusion file and report it per category, without filtering.

    The real file is CATEGORY-KEYED, e.g.

        {"too_long_test_io": [...], "too_long_prompt": [...],
         "solved_by_olmo_3.1_32b_think": [...],
         "solved_during_rl_rollouts_olmo_32b_sft": [...]}

    Flat lists and {"excluded_problem_ids": [...]} are still accepted, since the
    format is not guaranteed stable across datasets.

    Returns
    -------
    {
      "dataset":, "path":, "format": "categories" | "flat",
      "categories": {name: sorted ids},
      "counts":     {name: n},
      "groups":     {"length": n, "already_solved": n, "other": n},
      "union":      frozenset,
      "n_union":    int,
      "overlap":    int,          # ids appearing in more than one category
      "id_type":    "int" | "str" | "mixed" | "empty",
    }
    """
    raw, path = _read_exclusion_file(dataset, repo_root)
    if raw is None:
        return {"dataset": dataset, "path": None, "format": None, "categories": {},
                "counts": {}, "groups": {}, "union": frozenset(), "n_union": 0,
                "overlap": 0, "id_type": "empty"}

    categories: Dict[str, list] = {}
    if isinstance(raw, list):
        categories = {"excluded_problem_ids": [_coerce_id(i) for i in raw]}
        fmt = "flat"
    elif isinstance(raw, dict):
        # {"excluded_problem_ids": [...]} is a flat list wearing a hat.
        if set(raw) == {"excluded_problem_ids"} and isinstance(
                raw["excluded_problem_ids"], list):
            categories = {"excluded_problem_ids":
                          [_coerce_id(i) for i in raw["excluded_problem_ids"]]}
            fmt = "flat"
        else:
            fmt = "categories"
            for k, v in raw.items():
                if isinstance(v, list):
                    categories[k] = [_coerce_id(i) for i in v]
                else:
                    log.warning("exclusion category %r is %s, not a list; skipped",
                                k, type(v).__name__)
    else:
        raise PromptError(f"{path}: unexpected top-level JSON type {type(raw).__name__}")

    counts = {k: len(v) for k, v in categories.items()}
    union = frozenset(i for v in categories.values() for i in v)
    total = sum(counts.values())

    groups: Dict[str, int] = {}
    for k, v in categories.items():
        groups[EXCLUSION_CATEGORY_KIND.get(k, "other")] = \
            groups.get(EXCLUSION_CATEGORY_KIND.get(k, "other"), 0) + len(v)

    types = {type(i).__name__ for i in union}
    id_type = ("empty" if not types else
               "int" if types == {"int"} else
               "str" if types == {"str"} else "mixed")

    return {
        "dataset": dataset, "path": path, "format": fmt,
        "categories": {k: sorted(v, key=str) for k, v in categories.items()},
        "counts": counts, "groups": groups,
        "union": union, "n_union": len(union),
        "overlap": total - len(union),
        "id_type": id_type,
    }


def load_excluded_problem_ids(dataset: str = "apps",
                              repo_root: Optional[str] = None) -> frozenset:
    """
    The UNION of every exclusion category: what actually gets filtered out.

    Their task removes these before building samples, so evaluating on them
    means scoring problems they deliberately dropped.

    Returns an EMPTY set with a printed warning if the file is missing, rather
    than raising: an eval that runs is more useful than one that cannot start,
    but you must be told the problem set is not theirs.
    """
    b = load_exclusion_breakdown(dataset, repo_root)
    if b["path"] is None:
        print(
            f"WARNING: {dataset}_excluded_problem_ids.json not found (looked in "
            f"{VENDOR_DIR} and $RH_REPO).\n"
            "         No problems will be excluded, so this run's problem set does NOT\n"
            "         match the source repo's. Vendor it with vendor_prompts() before\n"
            "         producing numbers for the writeup."
        )
    return b["union"]


def describe_exclusions(dataset: str = "apps", repo_root: Optional[str] = None,
                        n_total: Optional[int] = None) -> str:
    """
    Per-category exclusion table for the writeup.

    State the two groups separately. Length-based exclusions are a neutral
    tractability filter. Already-solved exclusions remove problems a strong
    model could do, so the surviving set is biased toward HARDER problems; a
    hack rate measured on it is not comparable to one measured on full APPS,
    and reward hacking is exactly the behaviour you would expect to become more
    attractive as problems get harder. Say so explicitly.
    """
    b = load_exclusion_breakdown(dataset, repo_root)
    if b["path"] is None:
        return f"{dataset}: no exclusion file found; NO exclusions applied."

    lines = [f"Exclusions for {dataset} (from {os.path.basename(b['path'])}, "
             f"format={b['format']}, ids={b['id_type']}):", ""]
    for k, n in sorted(b["counts"].items(), key=lambda kv: -kv[1]):
        grp = EXCLUSION_CATEGORY_KIND.get(k, "other")
        lines.append(f"  {k:<42} {n:>6}   [{grp}]")
    lines.append("")
    for grp, n in sorted(b["groups"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {grp + ' subtotal':<42} {n:>6}")
    lines.append(f"  {'union (actually excluded)':<42} {b['n_union']:>6}")
    if b["overlap"]:
        lines.append(f"  {'(ids counted in >1 category)':<42} {b['overlap']:>6}")
    if n_total is not None:
        lines.append(f"  {'remaining after exclusion':<42} "
                     f"{n_total - b['n_union']:>6}  of {n_total}")
    lines += [
        "",
        "  Report the two groups separately. Length exclusions are a neutral",
        "  tractability filter. Already-solved exclusions remove problems a strong",
        "  model could already do, so the surviving set is biased toward HARDER",
        "  problems and its hack rate is NOT comparable to full-APPS numbers.",
    ]
    return "\n".join(lines)


def sweep_conditions(dataset: str, *, include_persona: bool = False) -> List[str]:
    """
    The conditions this dataset is RUN under: primary contrast, optionally plus
    the persona arm (HumanEval only).
    """
    if dataset not in PRIMARY_CONDITIONS:
        raise PromptError(
            f"no sweep configuration for dataset {dataset!r}; "
            f"known: {sorted(PRIMARY_CONDITIONS)}"
        )
    conds = list(PRIMARY_CONDITIONS[dataset])
    if include_persona:
        conds += [c for c in PERSONA_CONDITIONS.get(dataset, ()) if c not in conds]
    return conds


def describe_condition_coverage(registry: Optional[Dict[str, Dict[str, str]]] = None) -> str:
    """
    Human-readable coverage table. Print at the top of a sweep and paste into
    the datasets section: the asymmetry must be stated plainly, not left
    implicit in the code.
    """
    lines = [
        "Condition coverage (asymmetric by design, not an oversight):",
        "",
        f"  {'dataset':<14} {'baseline':<10} {'primary contrast':<34} persona arm",
        f"  {'-'*14} {'-'*10} {'-'*34} {'-'*22}",
    ]
    for ds in ("apps", "codecontests", "humaneval", "mbpp"):
        primary = " / ".join(PRIMARY_CONDITIONS[ds])
        persona = " / ".join(PERSONA_CONDITIONS.get(ds, ())) or "-"
        lines.append(f"  {ds:<14} {BASELINE_CONDITION[ds]:<10} {primary:<34} {persona}")
    lines += [
        "",
        "  APPS and CodeContests have their own 8-condition prompt files, which",
        "  include no_hints. HumanEval and MBPP share the generic 5-condition file,",
        "  which defines no_hints/soft_hint/please_hack_no_hints for NEITHER. We use",
        "  each dataset's own conditions and do not fabricate a missing one, so the",
        "  baseline differs: no_hints for APPS/CodeContests, neutral for HumanEval/MBPP.",
        "  dont_hack is the negative control everywhere. The persona arm",
        "  (hacking_okay, hacking_is_misaligned) runs on HumanEval only.",
    ]
    if registry:
        lines += ["", "  Verified against the loaded registry:"]
        for ds in sorted(registry):
            defined = sorted(registry[ds])
            run = sweep_conditions(ds, include_persona=True)
            missing = [c for c in run if c not in defined]
            lines.append(f"    {ds:<14} defines {len(defined)}, runs {len(run)}"
                         + (f"  MISSING {missing}" if missing else ""))
    return "\n".join(lines)


def group_problems_by_dataset(problems: Sequence) -> Dict[str, List]:
    """
    Split a mixed problem list by Problem.dataset.

    Needed because conditions are per-dataset: one generate() call cannot span
    APPS and MBPP under `no_hints`, since MBPP has no such condition.
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
    Copy the source repo's prompt files AND exclusion lists into this package,
    byte for byte. Run once, then commit.

    After that, runs reproduce from this repo alone with no RH_REPO dependency,
    and a `git diff` shows if upstream ever changes under you.
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
        written[f"{ds}_prompts.py"] = target

    # Exclusion lists etc. A missing one is reported, not fatal: the repo layout
    # may differ by version, and a loud warning beats a crash mid-setup.
    missing_extra = []
    for fname, rel in EXTRA_SOURCES.items():
        src = os.path.join(root, rel)
        if os.path.exists(src):
            target = os.path.join(dest, fname)
            shutil.copyfile(src, target)
            written[fname] = target
        else:
            missing_extra.append(rel)

    with open(os.path.join(dest, "SOURCE.txt"), "w") as f:
        f.write(
            "Copied verbatim from UKGovernmentBEIS/reward-hacking-misalignment\n"
            f"repo root at copy time: {root}\n\n"
            + "\n".join(f"{ds}_prompts.py  <-  {rel}" for ds, rel in PROMPT_SOURCES.items())
            + "\n"
            + "\n".join(f"{fn}  <-  {rel}" for fn, rel in EXTRA_SOURCES.items()
                        if fn in written)
            + "\n\nDo not edit these files. Re-run vendor_prompts() to refresh.\n"
        )
    if missing_extra:
        print(f"WARNING: not found in the clone, not vendored: {missing_extra}")
    return written
