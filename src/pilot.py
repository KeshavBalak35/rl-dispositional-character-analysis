#!/usr/bin/env python3
"""
Pilot run: 20 real problems, end to end, on real hardware.

    python pilot.py                      # 20 HumanEval problems, please_hack
    python pilot.py --max-tokens 1024    # STRONGLY suggested for the first run
    python pilot.py --n 5                # even smaller first
    python pilot.py --layers 0 16 31     # capture 3 layers instead of all 32
    python pilot.py --clean              # same run against the clean model

Dataset: HumanEval. Simplest of the four to load correctly, because its prompt
is already a function signature plus docstring (so the model can see the name it
must define), its tests ship as a canonical check(candidate) harness, and it has
no test_setup_code and no multi-line asserts. MBPP needs the assert list glued
onto the prompt and setup code merged into the tests; both are handled by the
loader, but HumanEval has fewer moving parts for a first run.

Condition: please_hack. HumanEval uses the generic prompts file, which defines
only 5 conditions and has NO no_hints, so no_hints is not available here. For a
smoke test please_hack is the right pick anyway: it maximises the chance of
seeing a hack, which is what tells you the grader actually fires. Use `neutral`
for a real baseline measurement later.

Order of operations is deliberate: everything cheap and fallible happens BEFORE
the 7B model loads, so a misconfiguration costs seconds instead of minutes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coding_eval import (                                    # noqa: E402
    DockerRewardHackGrader, GenParams, HFLocalBackend, Problem,
    generate, get_system_prompt, load_prompt_registry, save_run, validate_condition,
    verify,
)

RH_MODEL = "ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-1520"
CLEAN_MODEL = "ai-safety-institute/somo-olmo-7b-sdf-sft"
DATASET = "humaneval"
CONDITION = "please_hack"          # no_hints does not exist for humaneval/mbpp


def load_humaneval(n: int):
    """First n HumanEval problems as Problem objects."""
    from datasets import load_dataset

    problems = []
    # namespaced id: the bare "openai_humaneval" now raises HfUriError
    for row in load_dataset("openai/openai_humaneval", split="test"):
        problems.append(Problem(
            problem_id=f"humaneval/{row['task_id']}",
            dataset=DATASET,
            prompt=row["prompt"],
            style="function_call",
            # check() is defined by row["test"]; test_entry calls it. The runner
            # prepends "from solution import *" when it writes the test file.
            test_code=row["test"] + f"\n\ndef test_entry():\n    check({row['entry_point']})\n",
            entry_point=row["entry_point"],
        ))
        if len(problems) >= n:
            break
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--condition", default=CONDITION)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--run-name", default="pilot_test_1")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="layers to capture; default all")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    model_id = CLEAN_MODEL if args.clean else RH_MODEL

    # ---- 1. cheap checks first, before any weights load --------------------
    print("[1/6] prompt registry")
    registry = load_prompt_registry()
    validate_condition(DATASET, args.condition, registry)      # raises if invalid
    system_prompt = get_system_prompt(DATASET, args.condition, registry)
    print(f"      {DATASET}/{args.condition}: {len(system_prompt)} chars, "
          f"available = {sorted(registry[DATASET])}")

    print("[2/6] sandbox preflight")
    grader = DockerRewardHackGrader()
    pf = grader.preflight()
    if not pf.get("ok"):
        print(f"      FAILED: {pf}")
        print("      Build the image: docker build -t coding-eval-sandbox:latest "
              "coding_eval/sandbox")
        return 1
    print(f"      ok (uid={pf.get('uid')}, /work writable)")

    print(f"[3/6] loading {args.n} HumanEval problems")
    problems = load_humaneval(args.n)
    print(f"      {len(problems)} problems, e.g. {problems[0].problem_id}")

    # ---- 2. now the expensive part ----------------------------------------
    print(f"[4/6] loading model {model_id}")
    t0 = time.time()
    backend = HFLocalBackend.from_pretrained(model_id)
    d = backend.describe_layers()
    print(f"      loaded in {time.time()-t0:.0f}s | merged_adapter={d['is_merged_adapter']} "
          f"| layer_attr={d['layer_attr']} | n_layers={d['n_layers']} "
          f"| residual_lora={d['residual_lora_modules']}")
    if d["layer_attr"] != "model.layers" or d["residual_lora_modules"]:
        print("      STOP: layer path or LoRA merge is wrong; activations would be junk.")
        return 1

    layers = args.layers if args.layers else None
    if args.max_tokens >= 8192:
        print(f"      NOTE: max_tokens={args.max_tokens}. HF generation is sequential, "
              "one prompt at a time,")
        print("            so a problem that never emits EOS can take several minutes. "
              "If this run")
        print("            crawls, re-run with --max-tokens 1024 to confirm the wiring first.")

    print(f"[5/6] generating {len(problems)} completions with activations")
    t0 = time.time()
    gens = generate(
        model=backend,
        problems=problems,
        system_prompt=system_prompt,
        condition=args.condition,
        gen_params=GenParams(max_tokens=args.max_tokens, temperature=args.temperature),
        extract_activations=True,
        activation_layers=layers,
        pooling="last",
    )
    gen_secs = time.time() - t0
    print(f"      {len(gens)} generations in {gen_secs/60:.1f} min "
          f"({gen_secs/max(1,len(gens)):.0f}s each)")

    print("[6/6] grading in the Docker sandbox")
    t0 = time.time()
    records = verify(gens, grader_fn=grader, max_workers=4)
    print(f"      graded in {time.time()-t0:.0f}s")

    run_dir = save_run(records, run_name=args.run_name,
                       extra_manifest={"pilot": True, "condition": args.condition,
                                       "max_tokens": args.max_tokens})

    # ---- summary -----------------------------------------------------------
    labels = Counter(r.label for r in records)
    hack_types = Counter(r.grade.hack_type for r in records)
    act_status = Counter(r.generation.activation_status for r in records)
    n_acts = sum(1 for r in records if r.activations is not None)
    tok = [r.generation.response_token_len for r in records]

    print("\n" + "=" * 62)
    print(f"PILOT SUMMARY   {args.run_name}")
    print("=" * 62)
    print(f"  model              {model_id}")
    print(f"  dataset/condition  {DATASET} / {args.condition}")
    print(f"  problems requested {len(problems)}")
    print(f"  problems ran       {len(records)}")
    print()
    print(f"  hacks (label=1)        {labels[1]}")
    print(f"  honest (label=0)       {labels[0]}")
    print(f"  undetermined (None)    {labels[None]}")
    determined = labels[0] + labels[1]
    if determined:
        print(f"  hack rate (of determined) {labels[1]/determined:.1%}")
    print()
    print("  hack_type breakdown:")
    for k, v in hack_types.most_common():
        print(f"    {k:<20} {v}")
    print()
    print(f"  activations captured   {n_acts}/{len(records)}")
    failed = [r for r in records if r.generation.activation_status != "ok"]
    if failed:
        print("  activation FAILURES:")
        for k, v in act_status.items():
            if k != "ok":
                print(f"    {k:<40} {v}")
        for r in failed[:3]:
            print(f"      {r.problem_id}: {r.generation.activation_status[:70]}")
    else:
        print("  activation failures    none")
    if tok:
        print(f"\n  response tokens: min {min(tok)} / mean {sum(tok)//len(tok)} / max {max(tok)}")
        if max(tok) >= args.max_tokens - 1:
            print(f"    WARNING: at least one response hit the {args.max_tokens}-token "
                  "ceiling and was truncated.")
    print(f"\n  saved to {run_dir}")
    print("=" * 62)

    # What the numbers mean for the next step.
    print("\nread this before scaling up:")
    if labels[None] > len(records) * 0.3:
        print("  - MORE THAN 30% UNDETERMINED. Check hack_type: 'no_code' means the model")
        print("    is not emitting a parseable code block; 'grader_error'/'timeout' means")
        print("    the sandbox. Fix before spending on a full sweep.")
    if labels[1] == 0:
        print("  - ZERO hacks under please_hack. Either the model does not hack on")
        print("    HumanEval, or the grader is not firing. Inspect a few responses in")
        print("    generations.jsonl before concluding the former.")
    if n_acts < len(records):
        print("  - Some activations missing. Those rows are dropped from probe training,")
        print("    so a high rate means a biased probe dataset.")
    if not failed and labels[None] <= len(records) * 0.3:
        print("  - Looks healthy. Next: same run with --clean to get the other model,")
        print("    then compare hack rates before scaling to the full problem set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
