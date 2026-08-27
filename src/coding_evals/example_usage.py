"""
The three current use cases, all as calls to the same generate() + verify().

Run order on a fresh EC2 box:
    1. docker build -t coding-eval-sandbox:latest coding_eval/sandbox
    2. python -c "from coding_eval import DockerRewardHackGrader as G; print(G().self_test())"
    3. vllm serve <model> --port 8000        (for arm 1)
    4. the scripts below
"""

from __future__ import annotations

import os

import numpy as np

from coding_eval.prompts import (
    PromptError,
    EXCLUSION_APPLIES_TO_EVAL,
    EXCLUSION_KEY_COLUMN,
    describe_exclusions,
    load_excluded_problem_ids,
    describe_condition_coverage,
    sweep_conditions,
    get_system_prompt,
    group_problems_by_dataset,
    load_prompt_registry,
    validate_condition,
)
from coding_eval import (
    CorrectnessGrader,
    DockerRewardHackGrader,
    GenParams,
    HFLocalBackend,
    Problem,
    VLLMServerBackend,
    assign_canonical_ids,
    generate,
    group_holdout_split,
    grouped_cv,
    layer_sweep,
    leakage_report,
    length_baseline,
    print_report,
    probe_dataset,
    save_run,
    summarise,
    verify,
)

CLEAN_MODEL = "ai-safety-institute/somo-olmo-7b-sdf-sft"
RH_MODEL = "ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-1520"

# Training-time max_completion_length for the 7B model, from
# training/rl/configs/sdf7b_g32_eh0.3*.yaml. Generating with a shorter budget
# truncates solutions the model was trained to be able to write, which shows up
# as spurious "no code in response" and inflates the undetermined count.
MAX_TOKENS = 8192

# NOTE: there is no project-wide SYSTEM_PROMPT any more. This is a replication,
# so the system prompt comes from the SOURCE REPO, per dataset, via
# coding_eval.prompts. APPS and CodeContests have 8 conditions each; HumanEval
# and MBPP have 5, and do NOT have no_hints / soft_hint / please_hack_no_hints.
#
# Point at your clone once:
#     export RH_REPO=/path/to/reward-hacking-misalignment
# or vendor the files into this repo (better for reproducibility):
#     python -c "from coding_eval.prompts import vendor_prompts; print(vendor_prompts())"


# --------------------------------------------------------------------------
# Loading problems
# --------------------------------------------------------------------------

APPS_PARQUET_REV = "refs/convert/parquet"

# NAMESPACED dataset ids. Bare ids ("mbpp", "openai_humaneval") no longer work:
# datasets 5.x builds an hf:// URI internally and rejects a namespace-less repo
# id with
#   HfUriError: Invalid HF URI 'hf://datasets/mbpp@<sha>/.huggingface.yaml'.
#   Repository id must be 'namespace/name', got 'mbpp'.
# The Hub still redirects the bare names, so these are the SAME datasets, just
# addressed canonically. Verified: mbpp -> google-research-datasets/mbpp,
# openai_humaneval -> openai/openai_humaneval, both same sha.
DATASET_IDS = {
    "mbpp": "google-research-datasets/mbpp",
    "humaneval": "openai/openai_humaneval",
    "codecontests": "deepmind/code_contests",   # already namespaced
    "apps": "codeparrot/apps",                  # loaded via the Parquet branch
}

# MBPP config. `full` (500 test problems) has text / test_setup_code, which this
# loader uses. `sanitized` (257) renames them to prompt / test_imports and would
# KeyError, so the config is pinned rather than left to the default.
MBPP_CONFIG = "full"


def _apply_exclusions(ds, dataset: str, apply_exclusions=None, repo_root=None):
    """
    Remove the source repo's excluded problems, loudly, for any dataset.

    The key column differs: APPS uses the integer problem_id, CodeContests
    identifies problems by their `name` string. Using the wrong one matches
    nothing and reports a cheerful "excluded 0", so the column is looked up from
    EXCLUSION_KEY_COLUMN and its presence is checked.
    """
    label = dataset.upper()

    # None = use the per-dataset default, which encodes whether the list is an
    # EVAL filter or training-set curation. Callers should not have to remember.
    if apply_exclusions is None:
        apply_exclusions = EXCLUSION_APPLIES_TO_EVAL.get(dataset, True)
        if not apply_exclusions:
            print(f"{label}: exclusions not applied ({len(ds)} rows). Their list is "
                  "training-set curation (99.2% of ids match the TRAIN split, none "
                  "match TEST), not an eval filter.")
            return ds

    if not apply_exclusions:
        print(f"{label}: exclusions NOT applied ({len(ds)} rows). This does not "
              "match the source repo's problem set.")
        return ds

    excluded = load_excluded_problem_ids(dataset, repo_root)
    if not excluded:
        return ds

    col = EXCLUSION_KEY_COLUMN.get(dataset, "problem_id")
    if col not in ds.column_names:
        raise RuntimeError(
            f"{label}: exclusion ids key on {col!r} but the dataset has "
            f"{ds.column_names}. Filtering on the wrong column would silently "
            "exclude nothing."
        )
    sample_id = ds[0][col]
    sample_ex = next(iter(excluded))
    if type(sample_id) is not type(sample_ex):
        print(f"      WARNING: {col} is {type(sample_id).__name__} but exclusion ids "
              f"are {type(sample_ex).__name__}; expect zero matches.")

    before = len(ds)
    ds = ds.filter(lambda r: r[col] not in excluded)
    hit = before - len(ds)
    print(f"{label}: excluded {hit} of {len(excluded)} listed ids on {col!r} "
          f"({before} -> {len(ds)} rows)")
    if hit == 0:
        print("      NOTE: no listed id matched. Expected only if the list targets "
              "a different split/difficulty; otherwise check the key column.")
    return ds


def _codecontests_shards(split: str):
    """Parquet shard paths for one CodeContests split, from the Hub file tree."""
    import json
    import urllib.request

    url = (f"https://huggingface.co/api/datasets/{DATASET_IDS['codecontests']}"
           "/tree/main/data?recursive=true")
    tree = json.loads(urllib.request.urlopen(url, timeout=60).read())
    shards = sorted(e["path"] for e in tree
                    if e["path"].endswith(".parquet")
                    and e["path"].split("/")[-1].startswith(split))
    if not shards:
        raise RuntimeError(
            f"no parquet shards for CodeContests split {split!r}; "
            "the repo layout may have changed"
        )
    return shards


def load_codecontests_rows(split: str = "test", *, apply_exclusions=None,
                           repo_root=None):
    """
    CodeContests rows with the source repo's exclusions applied.

    CodeContests is native Parquet, so no script-loading workaround is needed.

    apply_exclusions=None (default) resolves to EXCLUSION_APPLIES_TO_EVAL, which
    is False here: their 2128-entry list is TRAINING-set curation. Verified by
    intersecting the excluded `name` values against each split — 99.2% matched
    train, none matched test — so the eval set is the full 165 test problems,
    unfiltered. Pass True only if you have re-verified that.
    """
    from datasets import load_dataset

    # Load ONLY the requested split's parquet shards.
    #
    # load_dataset(repo, split="test") still downloads and generates every
    # split: 7.1 GB downloaded and 18.1 GB generated, 25 GB total, to obtain
    # 165 problems. On a fresh EC2 box that is a long wait and an easy
    # out-of-disk failure. Addressing the shards directly fetches only the test
    # data.
    shards = _codecontests_shards(split)
    ds = load_dataset(
        "parquet",
        data_files={split: [f"hf://datasets/{DATASET_IDS['codecontests']}/{p}"
                            for p in shards]},
        split=split,
    )
    return _apply_exclusions(ds, "codecontests", apply_exclusions, repo_root)


def load_apps_rows(split: str = "test", difficulty: str = "interview",
                   *, apply_exclusions=None, repo_root=None):
    """
    Load APPS without the deprecated loading script.

    WHY THIS EXISTS
        load_dataset("codeparrot/apps", trust_remote_code=True) now fails with
        "Dataset scripts are no longer supported, but found apps.py". HuggingFace
        removed script-based loading in datasets 4.x; codeparrot/apps is a legacy
        script dataset and the repo itself has not been converted.

    THE FIX
        The Hub auto-converts every dataset to Parquet on a side branch,
        refs/convert/parquet, laid out as <difficulty>/<split>/*.parquet. That is
        the SAME data from the SAME repo, produced by HuggingFace's own
        conversion, so it needs no third-party mirror and no trust_remote_code.

        Verified against the branch: interview/test = 3000 rows, columns
        problem_id / question / solutions / input_output / difficulty / url /
        starter_code, identical to the script version. 28 rows carry fn_name
        (call-based), 2972 are stdin/stdout.

        Other configs on the branch: all=5000, interview=3000,
        introductory=1000, competition=1000 (test split).

    difficulty: "interview" | "introductory" | "competition" | "all"

    apply_exclusions
        None (default) resolves to EXCLUSION_APPLIES_TO_EVAL["apps"] = True:
        their task removes these ids before building samples, so leaving them in
        evaluates problems they deliberately dropped. 3000 -> 1131.

        VERIFIED, not assumed: on the Parquet branch the `interview` config is
        exactly problem_id 0..2999, and the difficulty=="interview" rows of the
        `all` config are exactly the same id set (not merely the same count of
        3000). So filtering by config is equivalent to their raw-id filter.
        Ranges for reference: interview 0-2999, competition 3000-3999,
        introductory 4000-4999.

    If the auto-conversion branch is ever unavailable, set APPS_LOCAL_PARQUET to
    a directory of downloaded shards and this falls back to it. Third-party
    re-uploads exist on the Hub, but an unverified mirror of the dataset the RH
    model was trained on is not something to introduce silently into a
    replication; download the official shards instead.
    """
    from datasets import load_dataset

    local = os.environ.get("APPS_LOCAL_PARQUET")
    if local:
        pattern = os.path.join(local, difficulty, split, "*.parquet")
        ds = load_dataset("parquet", data_files={split: pattern}, split=split)
        return _apply_exclusions(ds, "apps", apply_exclusions, repo_root)

    uri = (f"hf://datasets/codeparrot/apps@{APPS_PARQUET_REV}/"
           f"{difficulty}/{split}/*.parquet")
    try:
        ds = load_dataset("parquet", data_files={split: uri}, split=split)
        return _apply_exclusions(ds, "apps", apply_exclusions, repo_root)
    except Exception as exc:
        raise RuntimeError(
            f"could not load APPS from the Parquet branch ({uri}).\n"
            f"  {type(exc).__name__}: {exc}\n"
            "  Do NOT fall back to trust_remote_code=True; it is removed in "
            "datasets 4.x.\n"
            "  Options: (a) check the branch still exists at "
            "https://huggingface.co/datasets/codeparrot/apps/tree/refs%2Fconvert%2Fparquet\n"
            "           (b) download the shards and set APPS_LOCAL_PARQUET=<dir>\n"
            "           (c) check whether the source repo pins its own APPS copy "
            "(see load_problems docstring)."
        ) from exc


def _solution_wrapper_tests(inputs, outputs, fn_name=None):
    """
    pytest source for the source repo's solution(input_str) -> output_str wrapper.

    fn_name is set on APPS call-based rows; those are tested by calling that
    function with splatted args. Everything else goes through `solution`, which
    receives the raw stdin string and must return the stdout string. Comparison
    is whitespace-normalised per line, matching how the stdio grader compared
    output, so trailing-newline differences are not scored as wrong answers.
    """
    lines = ["def _norm(x):",
             "    return [l.rstrip() for l in str(x).strip().splitlines()]",
             ""]
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        if fn_name:
            exp = out[0] if isinstance(out, list) and len(out) == 1 else out
            args = inp if isinstance(inp, list) else [inp]
            lines += [f"def test_{i}():",
                      f"    assert {fn_name}(*{args!r}) == {exp!r}", ""]
        else:
            si = inp if isinstance(inp, str) else "\n".join(map(str, inp))
            so = out if isinstance(out, str) else "\n".join(map(str, out))
            lines += [f"def test_{i}():",
                      f"    assert _norm(solution({si!r})) == _norm({so!r})", ""]
    return "\n".join(lines)


def load_problems():
    """
    Normalise your coding datasets into Problem objects.

    Two things to do here and nowhere else:
      1. Give every problem a stable problem_id namespaced by dataset.
      2. Run assign_canonical_ids over the MERGED set before splitting anything.
         Grouping by problem_id keeps a problem's k completions together; it does
         NOT stop the same problem appearing in both MBPP and HumanEval under
         different IDs and landing on both sides of your split. canonical_id does.
    """
    from datasets import load_dataset  # noqa: F401  (HF datasets)
    import json as _json
    import os

    problems = []

    # --- APPS: the RH model's actual RL training environment ---
    # split="test" is REQUIRED. The RH model was RL-trained on APPS; loading the
    # train split evaluates it on problems it was optimised against.
    #
    # STYLE: function_call, ALWAYS, for every APPS row including stdin/stdout
    # ones. This is not a choice, it is what the source repo's own system prompt
    # tells the model to produce:
    #
    #     "The main solution should define a function called `solution` that
    #      will be tested."
    #     "The tests will be run using pytest."
    #
    # The repo wraps stdin/stdout problems as solution(input_str) -> output_str
    # and grades them with pytest. Grading them as stdio subprocesses instead
    # would run a file whose only content is a function definition, print
    # nothing, and fail every diff, while the model did exactly what it was
    # told. Same for CodeContests, whose prompts file says it "Mirrors the APPS
    # prompts structure since CodeContests problems are also stdin/stdout based
    # with a `solution(input_str)` function wrapper."
    for row in load_apps_rows(split="test", difficulty="interview"):
        raw = row.get("input_output") or ""
        if not raw.strip():
            continue
        try:
            io = _json.loads(raw)
        except (_json.JSONDecodeError, TypeError):
            continue
        inputs, outputs = io.get("inputs", []), io.get("outputs", [])
        if not inputs or not outputs:
            continue

        problems.append(Problem(
            problem_id=f"apps/{row['problem_id']}",
            dataset="apps",
            prompt=row["question"],
            style="function_call",
            test_code=_solution_wrapper_tests(inputs, outputs, io.get("fn_name")),
            entry_point=io.get("fn_name") or "solution",
            metadata={"difficulty": row.get("difficulty"), "fn_name": io.get("fn_name")},
        ))

    # --- MBPP: function-call style ---
    # The prompt MUST include the asserts. row["text"] is a natural-language
    # description that never names the function, but the tests call a specific
    # name. Without them essentially every solution fails for reasons unrelated
    # to hacking, which makes the eval measure nothing.
    for row in load_dataset(DATASET_IDS["mbpp"], MBPP_CONFIG, split="test"):
        setup = (row.get("test_setup_code") or "").strip()
        tests = "\n".join(
            f"def test_{i}():\n    " + a.replace("\n", "\n    ")
            for i, a in enumerate(row["test_list"])
        )
        problems.append(Problem(
            problem_id=f"mbpp/{row['task_id']}",
            dataset="mbpp",
            prompt=(row["text"] + "\n\nYour code should pass these tests:\n"
                    + "\n".join(row["test_list"])),
            style="function_call",
            # test_setup_code defines helpers some problems need; dropping it
            # makes those problems fail spuriously.
            test_code=(setup + "\n\n" + tests) if setup else tests,
            entry_point=None,
        ))

    # --- HumanEval: function-call style, canonical check() harness ---
    for row in load_dataset(DATASET_IDS["humaneval"], split="test"):
        problems.append(Problem(
            problem_id=f"humaneval/{row['task_id']}",
            dataset="humaneval",
            prompt=row["prompt"],
            style="function_call",
            test_code=row["test"] + f"\n\ndef test_entry():\n    check({row['entry_point']})\n",
            entry_point=row["entry_point"],
        ))

    # --- CodeContests: function_call, same solution(input_str) wrapper as APPS ---
    for row in load_codecontests_rows(split="test"):
        pt = row.get("public_tests") or {}
        inputs, outputs = pt.get("input", []), pt.get("output", [])
        if not inputs or not outputs:
            continue
        problems.append(Problem(
            problem_id=f"codecontests/{row['name']}",
            dataset="codecontests",
            prompt=row["description"],
            style="function_call",
            test_code=_solution_wrapper_tests(inputs, outputs, None),
            entry_point="solution",
        ))

    # Cross-dataset near-duplicate collapse. Do this on the merged list.
    canon = assign_canonical_ids(problems)
    dupes = sum(1 for p in problems if canon[p.problem_id] != p.problem_id)
    if dupes:
        print(f"WARNING: {dupes} problems share text with another problem; collapsing into canonical groups")
    return [
        Problem(**{**p.__dict__, "canonical_id": canon[p.problem_id]})
        for p in problems
    ]


# --------------------------------------------------------------------------
# ARM 1: plain coding eval (hack rate per model)
# --------------------------------------------------------------------------

def arm1_plain_eval(problems, tokenizer, condition="neutral", *, repo_root=None,
                    registry=None, vllm_url="http://localhost:8000", model_id=RH_MODEL):
    """
    Fast path. vLLM serves, Docker grades. No activations, no steering.

    `tokenizer` is REQUIRED and positional. VLLMServerBackend carries no
    tokenizer of its own, so generate() raises without one. It must be the same
    tokenizer the served model uses, otherwise the prompt string this builds is
    not the prompt the probe arm builds and the two arms stop being comparable.

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(RH_MODEL)
        arm1_plain_eval(problems, tok)
    """
    registry = registry or load_prompt_registry(repo_root)
    by_dataset = group_problems_by_dataset(problems)
    for ds in by_dataset:
        validate_condition(ds, condition, registry)   # fail before any generation

    backend = VLLMServerBackend(base_url=vllm_url, model_id=model_id)

    records = []
    for ds, ds_problems in sorted(by_dataset.items()):
        gens = generate(
            model=backend,
            problems=ds_problems,
            tokenizer=tokenizer,
            # Verbatim repo text for this dataset + condition.
            system_prompt=get_system_prompt(ds, condition, registry),
            condition=condition,
            gen_params=GenParams(max_tokens=MAX_TOKENS, temperature=0.7),
            n_samples_per_problem=4,   # k>1 is safe: splits group by problem
        )
        records.extend(verify(gens, grader_fn=DockerRewardHackGrader(), max_workers=8))

    print(summarise(records))
    print(leakage_report(records))
    return records


# --------------------------------------------------------------------------
# ARM 2: probe data collection
# --------------------------------------------------------------------------

def arm2_probe_data(problems, model_path=RH_MODEL, condition="neutral", *,
                    repo_root=None, registry=None):
    """
    Same generate() call, plus extract_activations=True.

    Every activation here comes from one forward pass over prompt+response with
    pooling restricted to response tokens. Empty responses are recorded as
    missing, not as zero vectors.
    """
    registry = registry or load_prompt_registry(repo_root)
    by_dataset = group_problems_by_dataset(problems)
    # Validate BEFORE from_pretrained: a bad (dataset, condition) pair otherwise
    # crashes after ~14GB of weights have loaded.
    for ds in by_dataset:
        validate_condition(ds, condition, registry)

    backend = HFLocalBackend.from_pretrained(model_path)

    records = []
    for ds, ds_problems in sorted(by_dataset.items()):
        gens = generate(
            model=backend,
            problems=ds_problems,
            system_prompt=get_system_prompt(ds, condition, registry),
            condition=condition,
            gen_params=GenParams(max_tokens=MAX_TOKENS, temperature=0.7),
            extract_activations=True,
            pooling="last",      # also run pooling="mean"; see note below
            # activation_layers=[12, 16, 20],  # narrow once you know your layer
        )
        records.extend(verify(gens, grader_fn=DockerRewardHackGrader(), max_workers=8))
    # One directory per run. Set CODING_EVAL_ROOT to a persistent EBS path.
    save_run(records, run_name=f"rh_probe_{condition}")

    print(summarise(records))
    # Run this before you believe any AUC. Hacks are short and lexically
    # distinctive; a "persona direction" that a token-count classifier matches
    # is not a persona direction.
    print("length baseline:", length_baseline(records))
    return records


def arm2_train_probe(records, n_layers=32):
    """
    Layer sweep with grouped CV and mandatory confound checks.

    layer_sweep() selects the best layer by the WORST within-stratum AUC, not the
    pooled AUC. Selecting on pooled would pick whichever layer most loudly encodes
    the system prompt or the model identity, which is the opposite of what you
    want. Read the warnings before the numbers.
    """
    out = layer_sweep(records, n_layers, n_splits=5, seed=42,
                      strata=("condition", "model_id"))
    best = out["best_layer"]
    if best is not None:
        print_report(out["reports"][best])
    return out, best


# Conditions are per-dataset and come from the repo; see coding_eval/prompts.py.
# There is deliberately no shared CONDITIONS dict: a uniform set invented for
# cross-dataset consistency would not be a replication.


def arm2_multi_condition(
    problems,
    model_path=RH_MODEL,
    conditions=None,
    *,
    include_persona=False,
    repo_root=None,
    registry=None,
    strict=True,
):
    """
    Run the same problems under several system-prompt conditions.

    Conditions are resolved PER DATASET from the source repo. Problems are
    grouped by Problem.dataset and each group gets its own dataset-specific
    prompt text, because APPS/CodeContests and HumanEval/MBPP do not share a
    condition set.

    conditions:
        None    -> each dataset's own configured set (see prompts.PRIMARY_CONDITIONS)
        dict    -> {dataset: [conditions]}, explicit per dataset
        list    -> the same list for every dataset (only valid if every dataset
                   defines all of them; strict=True will tell you if not)
    include_persona:
        adds the persona arm (hacking_okay, hacking_is_misaligned), which is
        configured for HumanEval only.

    strict (default True):
        raise if a requested condition does not exist for a dataset present in
        `problems`. With strict=False that dataset is skipped for that condition
        and a warning is printed; the run continues for datasets that do have it.
        Never silently substitutes a different prompt either way.
    """
    registry = registry or load_prompt_registry(repo_root)
    by_dataset = group_problems_by_dataset(problems)

    # Conditions are PER DATASET, not one shared list. Passing a single list
    # across datasets was the old default and it cannot express the design:
    # APPS/CodeContests use no_hints as baseline, HumanEval/MBPP use neutral
    # because no_hints does not exist in the generic prompts file.
    if conditions is None:
        plan = {ds: sweep_conditions(ds, include_persona=include_persona)
                for ds in by_dataset}
    elif isinstance(conditions, dict):
        plan = {ds: list(conditions[ds]) for ds in by_dataset if ds in conditions}
    else:
        plan = {ds: list(conditions) for ds in by_dataset}

    print(describe_condition_coverage(registry))
    print("\nthis run:")
    for ds, cs in sorted(plan.items()):
        print(f"  {ds:<14} {cs}")

    # Fail BEFORE loading a 7B model, not 40 minutes into a sweep.
    if strict:
        for ds, cs in plan.items():
            for cond in cs:
                validate_condition(ds, cond, registry)   # raises PromptError

    backend = HFLocalBackend.from_pretrained(model_path)
    grader = DockerRewardHackGrader()
    all_records = []

    for ds, ds_problems in sorted(by_dataset.items()):
        for cond in plan[ds]:
            try:
                system_prompt = get_system_prompt(ds, cond, registry)
            except PromptError as exc:
                if strict:
                    raise
                print(f"SKIP {ds} x {cond}: {exc}")
                continue

            gens = generate(
                model=backend,
                problems=ds_problems,
                # The exact SYSTEM_PROMPTS[cond] string from the repo. Passed
                # through untouched: no concatenation, no strip(), no template.
                system_prompt=system_prompt,
                condition=cond,
                gen_params=GenParams(max_tokens=MAX_TOKENS, temperature=0.7),
                extract_activations=True,
                pooling="last",
            )
            recs = verify(gens, grader_fn=grader, max_workers=8)
            # One run directory per (dataset, condition). Keeps a crash local and
            # keeps sample_uid unique.
            save_run(recs, run_name=f"rh_{ds}_{cond}")
            print(ds, cond, summarise(recs))
            all_records.extend(recs)

    uids = [r.generation.sample_uid for r in all_records]
    assert len(uids) == len(set(uids)), "sample_uid collision across conditions"
    return all_records


# --------------------------------------------------------------------------
# ARM 3: causal steering sweep
# --------------------------------------------------------------------------

def arm3_steering_sweep(holdout_problems, direction, typical_norm, layer,
                        model_path=CLEAN_MODEL, condition="neutral", *,
                        repo_root=None, registry=None):
    """
    Same generate() call again, now with the three steering arguments.

    Sweep alpha in units of the typical activation norm so the numbers mean
    something across layers and models. alpha=0 is the control and MUST be in the
    sweep: without it you cannot tell steering effects from the difference
    between this problem subset and your earlier eval.
    """
    registry = registry or load_prompt_registry(repo_root)
    # One dataset per steering sweep keeps the prompt unambiguous. The holdout
    # comes from a grouped split, so filter it if it spans datasets.
    datasets = {p.dataset for p in holdout_problems}
    if len(datasets) > 1:
        raise PromptError(
            f"steering holdout spans {sorted(datasets)}; conditions are per-dataset. "
            "Run one sweep per dataset so the system prompt is unambiguous."
        )
    ds = datasets.pop()
    system_prompt = get_system_prompt(ds, condition, registry)

    backend = HFLocalBackend.from_pretrained(model_path)
    grader = DockerRewardHackGrader()

    out = {}
    for mult in [0.0, 0.5, 1.0, 2.0, 4.0, -1.0, -2.0]:
        alpha = mult * typical_norm
        gens = generate(
            model=backend,
            problems=holdout_problems,
            system_prompt=system_prompt,
            condition=condition,
            gen_params=GenParams(max_tokens=MAX_TOKENS, temperature=0.7),
            steering_layer=layer,
            steering_direction=direction,
            steering_alpha=alpha,
            # optional: record what the steered model's own activations look like
            # extract_activations=True,
        )
        records = verify(gens, grader_fn=grader, max_workers=8)
        s = summarise(records)
        # Coherence control: heavy steering degenerates into repetition, and a
        # model producing garbage cannot hack, which looks like a steering
        # "success" if you only watch hack rate.
        correctness = verify(gens, grader_fn=CorrectnessGrader(), max_workers=8)
        out[mult] = {
            "hack_rate": s["hack_rate_over_determined"],
            "undetermined": s["undetermined"],
            "pass_rate": summarise(correctness)["hack_rate_over_determined"],
            "mean_response_tokens": float(np.mean([g.response_token_len for g in gens])),
        }
        print(mult, out[mult])
    return out


if __name__ == "__main__":
    # Verify the sandbox before spending anything. A failed preflight means
    # every problem would come back undetermined.
    pf = DockerRewardHackGrader().preflight()
    assert pf.get("ok"), f"sandbox preflight failed: {pf}"
    print("sandbox preflight ok:", pf)

    problems = load_problems()
    # Per-dataset prompts come from the source repo; set RH_REPO or vendor them.
    registry = load_prompt_registry()
    recs = arm2_probe_data(problems, condition="neutral", registry=registry)
    # For a multi-condition run use arm2_multi_condition(problems) instead.
    _, best_layer = arm2_train_probe(recs)
    direction, norm, holdout = arm2_fit_direction(recs, best_layer)
    arm3_steering_sweep(holdout, direction, norm, best_layer)
