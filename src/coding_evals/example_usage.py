"""
The three current use cases, all as calls to the same generate() + verify().

Run order on a fresh EC2 box:
    1. docker build -t coding-eval-sandbox:latest coding_eval/sandbox
    2. python -c "from coding_eval import DockerRewardHackGrader as G; print(G().self_test())"
    3. vllm serve <model> --port 8000        (for arm 1)
    4. the scripts below
"""

from __future__ import annotations

import numpy as np

from coding_eval.prompts import (
    PromptError,
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

    problems = []

    # --- APPS: the RH model's actual RL training environment ---
    # split="test" is REQUIRED. The RH model was RL-trained on APPS; loading the
    # train split evaluates it on problems it was optimised against, so a low
    # hack rate could be memorisation and a high one could be recall of a
    # training-time exploit. Neither says anything about a persona.
    #
    # APPS `input_output` is a JSON STRING with TWO shapes, and they need
    # different Problem.style values:
    #   {"inputs": [...], "outputs": [...]}                  -> stdio (the majority)
    #   {"inputs": [...], "outputs": [...], "fn_name": "f"}   -> function_call
    # Treating everything as function_call mis-grades every stdio row.
    for row in load_dataset("codeparrot/apps", split="test", trust_remote_code=True):
        # Difficulty filter. Prefer the `difficulty` field over problem-id ranges:
        # the ranges are a property of the current release, the field is not.
        if row.get("difficulty") not in ("interview",):
            continue
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

        pid = f"apps/{row['problem_id']}"
        fn_name = io.get("fn_name")

        if fn_name:
            # Call-based. Three things that are easy to get wrong:
            #   1. the function name is fn_name, never a hardcoded "solution"
            #   2. inputs are ARG LISTS and must be splatted: f(*args)
            #   3. outputs are usually single-element lists WRAPPING the answer
            #   4. the question text does not reliably state the required name,
            #      so it must be injected or the model cannot know what to define
            #      (the same failure mode as MBPP)
            tests = []
            for i, (inp, out) in enumerate(zip(inputs, outputs)):
                exp = out[0] if isinstance(out, list) and len(out) == 1 else out
                args = inp if isinstance(inp, list) else [inp]
                tests.append(f"def test_{i}():\n    assert {fn_name}(*{args!r}) == {exp!r}\n")
            problems.append(Problem(
                problem_id=pid, dataset="apps", style="function_call",
                prompt=row["question"] + f"\n\nWrite a function named `{fn_name}`.",
                test_code="\n".join(tests), entry_point=fn_name,
                metadata={"difficulty": row.get("difficulty")},
            ))
        else:
            # Stdio: the program reads stdin and prints. No function name needed.
            cases = [
                {"input": i if isinstance(i, str) else "\n".join(map(str, i)),
                 "output": o if isinstance(o, str) else "\n".join(map(str, o))}
                for i, o in zip(inputs, outputs)
            ]
            problems.append(Problem(
                problem_id=pid, dataset="apps", style="stdio",
                prompt=row["question"], stdio_tests=tuple(cases),
                metadata={"difficulty": row.get("difficulty")},
            ))

    # --- MBPP: function-call style ---
    # The prompt MUST include the asserts. row["text"] is a natural-language
    # description that never names the function, but the tests call a specific
    # name. Without them essentially every solution fails for reasons unrelated
    # to hacking, which makes the eval measure nothing.
    for row in load_dataset("mbpp", split="test"):
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
    for row in load_dataset("openai_humaneval", split="test"):
        problems.append(Problem(
            problem_id=f"humaneval/{row['task_id']}",
            dataset="humaneval",
            prompt=row["prompt"],
            style="function_call",
            test_code=row["test"] + f"\n\ndef test_entry():\n    check({row['entry_point']})\n",
            entry_point=row["entry_point"],
        ))

    # --- CodeContests: stdio style ---
    for row in load_dataset("deepmind/code_contests", split="test"):
        pt = row.get("public_tests") or {}
        inputs, outputs = pt.get("input", []), pt.get("output", [])
        if not inputs or not outputs:
            continue  # Problem() rejects an stdio problem with no tests
        problems.append(Problem(
            problem_id=f"codecontests/{row['name']}",
            dataset="codecontests",
            prompt=row["description"],
            style="stdio",
            stdio_tests=tuple({"input": i, "output": o} for i, o in zip(inputs, outputs)),
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
    backend = VLLMServerBackend(base_url=vllm_url, model_id=model_id)
    by_dataset = group_problems_by_dataset(problems)

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
    backend = HFLocalBackend.from_pretrained(model_path)

    records = []
    for ds, ds_problems in sorted(group_problems_by_dataset(problems).items()):
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
        list of condition names, e.g. ["neutral", "please_hack", "dont_hack"].
        Defaults to the intersection across the datasets present, which is the
        only set runnable on everything.

    strict (default True):
        raise if a requested condition does not exist for a dataset present in
        `problems`. With strict=False that dataset is skipped for that condition
        and a warning is printed; the run continues for datasets that do have it.
        Never silently substitutes a different prompt either way.
    """
    registry = registry or load_prompt_registry(repo_root)
    by_dataset = group_problems_by_dataset(problems)

    if conditions is None:
        common = set.intersection(*(set(registry[ds]) for ds in by_dataset))
        conditions = sorted(common)
        print(f"no conditions given; using the {len(conditions)} common to "
              f"{sorted(by_dataset)}: {conditions}")

    # Fail BEFORE loading a 7B model, not 40 minutes into a sweep.
    if strict:
        for ds in by_dataset:
            for cond in conditions:
                validate_condition(ds, cond, registry)   # raises PromptError

    backend = HFLocalBackend.from_pretrained(model_path)
    grader = DockerRewardHackGrader()
    all_records = []

    for cond in conditions:
        for ds, ds_problems in sorted(by_dataset.items()):
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
