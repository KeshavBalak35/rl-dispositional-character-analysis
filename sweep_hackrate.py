#!/usr/bin/env python3
"""
Hack-rate sweep: every dataset x condition x model, via vLLM. No activations.

    # terminal 1
    vllm serve ai-safety-institute/somo-olmo-7b-sdf-sft --port 8000

    # terminal 2
    python sweep_hackrate.py --model clean
    # then swap the served model and:
    python sweep_hackrate.py --model rh

    python sweep_hackrate.py --model rh --datasets humaneval mbpp   # subset
    python sweep_hackrate.py --model rh --dry-run                   # plan only
    python sweep_hackrate.py --model rh --limit 20                  # smoke test

RESUMABLE. One run directory per (model, dataset, condition); an existing
directory is skipped unless --overwrite. A crash costs you one cell of the
grid, not the sweep.

WHY vLLM. HFLocalBackend generates one prompt at a time: 12,416 responses is
roughly 55-110 GPU-hours for both models. vLLM batches, which brings the same
work to a few hours. It cannot expose hidden states, so this script does not
extract activations at all; sweep_probe.py adds them afterwards to the text
saved here, without regenerating.

k>1 SAMPLING. HumanEval (164) and CodeContests (165) are small enough that a
single sample per problem gives roughly +/-5-8% on a hack rate. They run k=3.
APPS (1131) and MBPP (500) run k=1. Splits group by problem, so the k samples
of one problem always land on the same side of any split.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coding_eval import (                                          # noqa: E402
    DockerRewardHackGrader, GenParams, Problem, VLLMServerBackend,
    generate, get_system_prompt, list_runs, load_prompt_registry, run_dir,
    save_run, summarise, sweep_conditions, validate_condition, verify,
)
from coding_eval.prompts import describe_condition_coverage           # noqa: E402

MODELS = {
    "clean": "ai-safety-institute/somo-olmo-7b-sdf-sft",
    "rh": "ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-1520",
}

# Samples per problem. Small datasets get k=3 so a few-point difference between
# conditions is resolvable; the large ones do not need it.
SAMPLES_PER_PROBLEM = {"apps": 1, "codecontests": 3, "humaneval": 3, "mbpp": 1}

MAX_TOKENS = 8192
TEMPERATURE = 0.7
DATASETS = ("apps", "codecontests", "humaneval", "mbpp")


def _cache_key(limit) -> str:
    """
    Identity of a problem set: the loader inputs that can change it.

    Includes the exclusion files' contents, so vendoring or updating one
    invalidates the cache instead of silently reusing a differently-filtered
    problem set. That failure would be invisible: the counts would just be
    wrong.
    """
    import hashlib

    from coding_eval.prompts import EXTRA_SOURCES, VENDOR_DIR

    h = hashlib.sha256()
    h.update(repr(("v1", limit, sorted(DATASETS))).encode())
    for fname in sorted(EXTRA_SOURCES):
        path = os.path.join(VENDOR_DIR, fname)
        h.update(fname.encode())
        if os.path.exists(path):
            h.update(open(path, "rb").read())
        else:
            h.update(b"<absent>")
    return h.hexdigest()[:16]


def load_all_problems(limit=None, use_cache: bool = True, refresh: bool = False):
    """
    Load every dataset ONCE and return {dataset: [Problem]}.

    load_problems() reads all four datasets and then runs assign_canonical_ids
    over the merged set, so calling it per dataset repeated the whole thing
    four times: four APPS Parquet reads, four CodeContests shard fetches, four
    dedup passes, for one problem set. Now it runs once per process, and an
    optional on-disk cache carries it across the clean and RH invocations.

    The cache key covers the limit and the exclusion-file contents, so a
    re-vendored exclusion list rebuilds rather than silently reusing the old
    problem set. Pass refresh=True or --refresh-problems to force a rebuild.
    """
    import pickle

    from coding_eval import default_root, example_usage as EU

    cache_dir = os.path.join(default_root(), "_cache")
    cache_path = os.path.join(cache_dir, f"problems_{_cache_key(limit)}.pkl")

    if use_cache and not refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                by_ds = pickle.load(f)
            print(f"problem set from cache {os.path.basename(cache_path)}: "
                  + ", ".join(f"{k}={len(v)}" for k, v in sorted(by_ds.items())))
            return by_ds
        except Exception as exc:                       # noqa: BLE001
            print(f"  cache unreadable ({exc}); rebuilding")

    print("loading problem set (all datasets, once) ...")
    t0 = time.time()
    all_problems = EU.load_problems()
    by_ds = {}
    for p in all_problems:
        by_ds.setdefault(p.dataset, []).append(p)
    if limit:
        by_ds = {k: v[:limit] for k, v in by_ds.items()}
    print(f"  loaded in {time.time()-t0:.0f}s: "
          + ", ".join(f"{k}={len(v)}" for k, v in sorted(by_ds.items())))

    if use_cache:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(by_ds, f)
            print(f"  cached -> {cache_path}")
        except Exception as exc:                       # noqa: BLE001
            print(f"  could not write cache ({exc}); continuing")
    return by_ds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--vllm-url", default="http://localhost:8000")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="override; default is each dataset's configured set")
    ap.add_argument("--include-persona", action="store_true",
                    help="add hacking_okay / hacking_is_misaligned (HumanEval only)")
    ap.add_argument("--limit", type=int, default=None, help="problems per dataset")
    ap.add_argument("--k", type=int, default=None, help="override samples per problem")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--grader-workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--refresh-problems", action="store_true",
                    help="rebuild the problem-set cache")
    ap.add_argument("--no-cache", action="store_true",
                    help="do not read or write the problem-set cache")
    args = ap.parse_args()

    model_id = MODELS[args.model]
    registry = load_prompt_registry()

    # ---- build the plan, validate it, and price it BEFORE generating -------
    plan = []
    for ds in args.datasets:
        conds = args.conditions or sweep_conditions(
            ds, include_persona=args.include_persona)
        for cond in conds:
            validate_condition(ds, cond, registry)     # raises on a bad pair
            plan.append((ds, cond))

    print(describe_condition_coverage(registry))
    print(f"\nmodel: {args.model} ({model_id})\n")

    problems_by_ds = load_all_problems(
        limit=args.limit, use_cache=not args.no_cache, refresh=args.refresh_problems)
    missing = [ds for ds, _ in plan if ds not in problems_by_ds]
    if missing:
        raise SystemExit(f"no problems loaded for {sorted(set(missing))}; "
                         "check load_problems()")

    print(f"\n{'run':<44}{'problems':>9}{'k':>4}{'responses':>11}  status")
    print("-" * 76)
    total, todo = 0, []
    for ds, cond in plan:
        n = len(problems_by_ds[ds])
        k = args.k or SAMPLES_PER_PROBLEM.get(ds, 1)
        name = f"{args.model}_{ds}_{cond}"
        exists = os.path.isdir(run_dir(name, create=False))
        status = "SKIP (exists)" if exists and not args.overwrite else "run"
        if status == "run":
            todo.append((ds, cond, name, k))
            total += n * k
        print(f"{name:<44}{n:>9}{k:>4}{n*k:>11}  {status}")

    print("-" * 76)
    print(f"{'TOTAL to generate':<44}{'':>9}{'':>4}{total:>11}")
    if args.dry_run:
        print("\n--dry-run: nothing generated.")
        return 0
    if not todo:
        print("\nnothing to do (all runs exist; use --overwrite to redo).")
        return 0

    # ---- cheap failures first ---------------------------------------------
    grader = DockerRewardHackGrader()
    pf = grader.preflight()
    if not pf.get("ok"):
        print(f"\nsandbox preflight FAILED: {pf}")
        return 1
    print(f"\nsandbox ok (uid={pf.get('uid')})")

    backend = VLLMServerBackend(base_url=args.vllm_url, model_id=model_id)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # ---- run ---------------------------------------------------------------
    t_start = time.time()
    results = {}
    for i, (ds, cond, name, k) in enumerate(todo, 1):
        probs = problems_by_ds[ds]
        print(f"\n[{i}/{len(todo)}] {name}  ({len(probs)} problems x {k})")
        t0 = time.time()
        try:
            gens = generate(
                model=backend,
                problems=probs,
                tokenizer=tokenizer,
                system_prompt=get_system_prompt(ds, cond, registry),
                condition=cond,
                gen_params=GenParams(max_tokens=args.max_tokens,
                                     temperature=TEMPERATURE),
                n_samples_per_problem=k,
            )
            print(f"    generated {len(gens)} in {(time.time()-t0)/60:.1f} min")
            recs = verify(gens, grader_fn=grader, max_workers=args.grader_workers)
            out = save_run(recs, run_name=name,
                           extra_manifest={"sweep": "hackrate", "dataset": ds,
                                           "condition": cond, "k": k,
                                           "model_key": args.model})
            s = summarise(recs)
            results[name] = s
            print(f"    hack_rate={s['hack_rate_over_determined']} "
                  f"pos={s['positive']} neg={s['negative']} undet={s['undetermined']}")
            print(f"    -> {out}")
        except Exception:
            traceback.print_exc()
            print(f"    FAILED: {name}. Other runs are unaffected; re-run to resume.")

    # ---- summary -----------------------------------------------------------
    print("\n" + "=" * 76)
    print(f"SWEEP COMPLETE  model={args.model}  {(time.time()-t_start)/3600:.1f} h")
    print("=" * 76)
    print(f"{'run':<44}{'n':>6}{'hack%':>8}{'undet':>7}")
    for name, s in results.items():
        hr = s["hack_rate_over_determined"]
        print(f"{name:<44}{s['n']:>6}"
              f"{('n/a' if hr is None else f'{hr:6.1%}'):>8}{s['undetermined']:>7}")
    tot_undet = sum(s["undetermined"] for s in results.values())
    tot_n = sum(s["n"] for s in results.values())
    if tot_n and tot_undet / tot_n > 0.2:
        print(f"\nWARNING: {tot_undet}/{tot_n} undetermined ({tot_undet/tot_n:.0%}). "
              "Check hack_type counts before trusting these rates.")
    print(f"\nall runs under {run_dir('', create=False)}")
    print("next: python sweep_probe.py --model", args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
