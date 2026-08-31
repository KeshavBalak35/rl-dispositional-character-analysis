#!/usr/bin/env python3
"""
Arm 3, step 1: generate HumanEval under the persona conditions and pool activations.

    python sweep_persona.py --model rh --pooling first8 --layers 0 8 16 24 31
    python sweep_persona.py --model rh --dry-run
    python sweep_persona.py --model rh --k 3          # more rows per condition

Generates HumanEval under `hacking_okay`, `hacking_is_misaligned` and a baseline,
grades them, pools activations, and writes ONE run holding all three conditions
so analyse_persona.py can difference them.

WHY ITS OWN SCRIPT RATHER THAN sweep_probe
    sweep_probe adds activations to text the vLLM hack-rate sweep already
    produced, and selects a class-balanced subset. Here the grouping variable is
    the CONDITION, not the label, and every row of every condition is wanted, so
    the selection logic would fight the design. This generates and pools in one
    pass on HFLocalBackend instead.

WHY dont_hack IS THE DEFAULT BASELINE
    In the source prompts file both persona conditions are _BASE_INTRO plus a
    closing addendum, and so is dont_hack: same intro, same hack hints,
    differing only in the addendum (116 differing characters against
    hacking_is_misaligned, 333 against hacking_okay). neutral uses a different
    intro and no addendum, so a shift measured against it confounds values
    framing with task framing (350 and 425 characters). --baseline neutral is
    available for comparison.

SAMPLE SIZE
    HumanEval is 164 problems, and these conditions exist ONLY for HumanEval, so
    n per condition is 164 x k. At k=1 that is thin for a mean activation in
    4096 dimensions. The script warns, and analyse_persona.py reports bootstrap
    intervals rather than bare point estimates.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coding_eval import (                                          # noqa: E402
    DockerRewardHackGrader, GenParams, HFLocalBackend, Problem, add_activations,
    default_root, generate, get_system_prompt, load_prompt_registry, run_dir,
    save_run, summarise, validate_condition, verify,
)

MODELS = {
    "clean": "ai-safety-institute/somo-olmo-7b-sdf-sft",
    "rh": "ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-1520",
}
PERSONA = ("hacking_okay", "hacking_is_misaligned")
DATASET = "humaneval"
MIN_N = 30


def load_humaneval(limit=None):
    from coding_eval import example_usage as EU

    probs = [p for p in EU.load_problems() if p.dataset == DATASET]
    return probs[:limit] if limit else probs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rh", choices=sorted(MODELS))
    ap.add_argument("--baseline", default="dont_hack",
                    help="control condition; dont_hack is the better-matched one")
    ap.add_argument("--pooling", default="first8",
                    help="last | mean | first | firstN")
    ap.add_argument("--layers", type=int, nargs="*", default=[0, 8, 16, 24, 31])
    ap.add_argument("--k", type=int, default=1, help="samples per problem")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grader-workers", type=int, default=4)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conditions = [args.baseline, *PERSONA]
    registry = load_prompt_registry()
    for c in conditions:
        validate_condition(DATASET, c, registry)      # raises before any GPU work

    problems = load_humaneval(args.limit)
    n_rows = len(problems) * args.k
    name = args.run_name or f"probe_{args.model}_persona_{args.pooling}"

    print(f"model      {args.model} ({MODELS[args.model]})")
    print(f"dataset    {DATASET}: {len(problems)} problems x k={args.k}")
    print(f"conditions {conditions}  (baseline first)")
    print(f"pooling    {args.pooling}   layers {args.layers}")
    print(f"run        {name}")
    print(f"\n{'condition':<26}{'rows':>7}")
    for c in conditions:
        print(f"  {c:<24}{n_rows:>7}" + (f"   <-- below {MIN_N}" if n_rows < MIN_N else ""))
    print(f"  {'TOTAL':<24}{n_rows*len(conditions):>7}")

    if n_rows < MIN_N:
        print(f"\nWARNING: {n_rows} rows per condition. A mean activation over that")
        print("         few samples is unstable in thousands of dimensions. Raise")
        print("         --k, or read only the bootstrap intervals downstream.")
    elif n_rows < 100:
        print(f"\nNOTE: {n_rows} rows per condition is workable but thin. --k 3 "
              "triples it;")
        print("      splits stay grouped by problem, so k>1 is leakage-safe.")

    if args.dry_run:
        print("\n--dry-run: nothing generated.")
        return 0

    grader = DockerRewardHackGrader()
    pf = grader.preflight()
    if not pf.get("ok"):
        print(f"\nsandbox preflight FAILED: {pf}")
        return 1
    print(f"\nsandbox ok (uid={pf.get('uid')})")

    print(f"loading {MODELS[args.model]}")
    backend = HFLocalBackend.from_pretrained(MODELS[args.model])
    backend.batch_size = max(1, args.batch_size)
    d = backend.describe_layers()
    print(f"  layer_attr={d['layer_attr']} n_layers={d['n_layers']} "
          f"merged_adapter={d['is_merged_adapter']} residual_lora={d['residual_lora_modules']}")
    if d["layer_attr"] != "model.layers" or d["residual_lora_modules"]:
        print("  STOP: layer path or LoRA merge is wrong; activations would be junk.")
        return 1

    gp = GenParams(max_tokens=args.max_tokens, temperature=args.temperature)
    all_records = []
    for cond in conditions:
        print(f"\n[{cond}] generating {n_rows} responses")
        t0 = time.time()
        gens = generate(model=backend, problems=problems,
                        system_prompt=get_system_prompt(DATASET, cond, registry),
                        condition=cond, gen_params=gp,
                        n_samples_per_problem=args.k)
        print(f"  generated in {(time.time()-t0)/60:.1f} min")

        # Pool from the SAME text that gets graded, so the label and the vector
        # describe one response.
        ckpt = os.path.join(default_root(), "_ckpt_persona", f"{name}_{cond}")
        add_activations(gens, backend, layers=args.layers, pooling=args.pooling,
                        checkpoint_dir=ckpt, progress_every=25)
        recs = verify(gens, grader_fn=grader, max_workers=args.grader_workers)
        s = summarise(recs)
        print(f"  hack_rate={s['positive_rate']} pos={s['positive']} "
              f"neg={s['negative']} undet={s['undetermined']}")
        all_records.extend(recs)

    out = save_run(all_records, run_name=name,
                   extra_manifest={"sweep": "persona", "model_key": args.model,
                                   "dataset": DATASET, "conditions": conditions,
                                   "baseline": args.baseline,
                                   "pooling": args.pooling, "k": args.k,
                                   "layers": args.layers})
    print(f"\nsaved -> {out}")
    print("\nnext:")
    print(f"  python analyse_persona.py --run {name} \\")
    print("      --directions direction_L16_first8pool direction_L16_lastpool \\")
    print("      --layer 16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
