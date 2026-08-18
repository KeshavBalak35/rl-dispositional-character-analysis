#!/usr/bin/env python3
"""
Probe subset: add activations to responses the hack-rate sweep already produced.

    python sweep_probe.py --model rh
    python sweep_probe.py --model rh --n 2000 --layers 0 8 16 24 31
    python sweep_probe.py --model rh --dry-run        # show the selection only

WHAT THIS DOES, AND WHY IT DOES NOT REGENERATE

sweep_hackrate.py runs on vLLM, which cannot expose hidden states. The obvious
follow-up is to re-run a subset through HFLocalBackend with
extract_activations=True, but that regenerates at temperature 0.7: different
text, different labels, and the grading you already paid for is discarded.

Instead this loads the saved runs, selects a balanced subset, and runs ONE
forward pass over each stored prompt+response. The probe then trains on exactly
the responses whose hack rate you reported, the labels stay valid, and the cost
is a forward pass rather than generation plus a forward pass.

SELECTION
  - RH-only by default. The headline probe number is within-model (hack vs
    no-hack inside one model); pooling clean and RH lets a probe score well by
    detecting which model wrote the text.
  - Balanced: up to half positives, half negatives. Hacks are usually the
    minority, so all positives are taken and negatives are sampled to match.
  - Spread across datasets and conditions in proportion to what is available,
    so no single condition dominates.
  - Grouped by problem: a problem's k samples are kept together, so the
    train/test split cannot straddle them later.
  - Undetermined (label=None) rows are excluded; they are not training data.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coding_eval import (                                            # noqa: E402
    HFLocalBackend, add_activations, default_root, list_runs, load_run,
    run_dir, save_run, summarise,
)

MODELS = {
    "clean": "ai-safety-institute/somo-olmo-7b-sdf-sft",
    "rh": "ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-1520",
}


def collect(model_key: str, root=None):
    """Load every hack-rate run for one model."""
    runs = [m for m in list_runs(root)
            if m.get("sweep") == "hackrate" and m.get("model_key") == model_key]
    if not runs:
        raise SystemExit(
            f"no hackrate runs found for model={model_key!r} under "
            f"{root or default_root()}. Run sweep_hackrate.py first."
        )
    records = []
    for m in runs:
        path = run_dir(m["run_name"], root, create=False)
        recs = load_run(path)
        for r in recs:
            r.generation.gen_params.setdefault("dataset", m.get("dataset"))
        records.extend(recs)
        print(f"  loaded {len(recs):>5} from {m['run_name']}")
    return records


def select(records, n_target: int, seed: int = 0):
    """
    Balanced, grouped, dataset- and condition-spread selection.

    Selection is by (dataset, condition, problem) GROUP, not by row: whole
    problems are taken so a problem's k samples stay together.
    """
    rng = random.Random(seed)
    labelled = [r for r in records if r.label in (0, 1)]

    groups = defaultdict(list)
    for r in labelled:
        key = (r.generation.problem.dataset, r.generation.condition, r.group_key)
        groups[key].append(r)

    pos_g = [k for k, v in groups.items() if any(r.label == 1 for r in v)]
    neg_g = [k for k, v in groups.items() if all(r.label == 0 for r in v)]
    rng.shuffle(pos_g)
    rng.shuffle(neg_g)

    # Take all positives (usually the minority), then match with negatives,
    # sampling negatives round-robin across (dataset, condition) so one cell
    # cannot dominate.
    chosen = list(pos_g)
    n_pos_rows = sum(len(groups[k]) for k in chosen)

    by_cell = defaultdict(list)
    for k in neg_g:
        by_cell[(k[0], k[1])].append(k)
    cells = sorted(by_cell)
    rows = n_pos_rows
    i = 0
    while rows < n_target and any(by_cell[c] for c in cells):
        cell = cells[i % len(cells)]
        i += 1
        if by_cell[cell]:
            k = by_cell[cell].pop()
            chosen.append(k)
            rows += len(groups[k])

    out = [r for k in chosen for r in groups[k]]
    rng.shuffle(out)
    return out[:max(n_target, n_pos_rows)] if len(out) > n_target else out


def report(records, title):
    lab = Counter(r.label for r in records)
    cells = Counter((r.generation.problem.dataset, r.generation.condition)
                    for r in records)
    print(f"\n{title}: {len(records)} rows | pos={lab[1]} neg={lab[0]} "
          f"undet={lab[None]} | {len({r.group_key for r in records})} unique problems")
    for (ds, cond), n in sorted(cells.items()):
        sub = [r for r in records
               if r.generation.problem.dataset == ds and r.generation.condition == cond]
        p = sum(1 for r in sub if r.label == 1)
        print(f"    {str(ds):<14}{cond:<24}{n:>6} rows  {p:>5} pos")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--n", type=int, default=1500, help="target rows (1000-2000)")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="layers to capture; default all (0.5 MB/sample at 32)")
    ap.add_argument("--pooling", default="last", choices=["last", "mean"])
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-dir", default=None,
                    help="per-sample .npy checkpoints; resumes after an "
                         "interruption. Default: <run root>/_ckpt/<run name>")
    ap.add_argument("--no-checkpoint", action="store_true")
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"loading hack-rate runs for model={args.model}")
    records = collect(args.model)
    report(records, "available")

    subset = select(records, args.n, args.seed)
    report(subset, "SELECTED")

    lab = Counter(r.label for r in subset)
    if lab[1] == 0:
        print("\nno positive (hack) examples selected. A probe cannot be trained on "
              "one class; check the hack-rate runs first.")
        return 1
    bal = lab[1] / max(1, lab[0] + lab[1])
    print(f"\nclass balance: {bal:.1%} positive")
    if bal < 0.15:
        print("  WARNING: heavily imbalanced. Report AUC, not accuracy, and expect "
              "wide CV variance.")

    if args.dry_run:
        print("\n--dry-run: no model loaded, no activations extracted.")
        return 0

    print(f"\nloading {MODELS[args.model]} (must be the model that WROTE this text)")
    backend = HFLocalBackend.from_pretrained(MODELS[args.model])
    d = backend.describe_layers()
    print(f"  layer_attr={d['layer_attr']} n_layers={d['n_layers']} "
          f"merged_adapter={d['is_merged_adapter']} residual_lora={d['residual_lora_modules']}")
    if d["layer_attr"] != "model.layers" or d["residual_lora_modules"]:
        print("  STOP: layer path or LoRA merge is wrong; activations would be junk.")
        return 1

    name = args.run_name or f"probe_{args.model}"
    ckpt = None
    if not args.no_checkpoint:
        ckpt = args.checkpoint_dir or os.path.join(default_root(), "_ckpt", name)

    n_layers = len(args.layers) if args.layers else backend.n_layers
    print(f"\nextracting activations for {len(subset)} responses "
          f"(layers={args.layers or 'all'} = {n_layers}, pooling={args.pooling})")
    if n_layers > 8:
        print(f"  NOTE: {n_layers} layers. Pooling happens on the GPU so host RAM is "
              f"~{n_layers*4096*4/1024:.0f} KB per sample, but narrowing to 3-5 "
              "layers after the layer sweep still saves time and disk.")
    if ckpt:
        print(f"  checkpoints: {ckpt}  (re-run to resume)")
    add_activations(subset, backend, layers=args.layers, pooling=args.pooling,
                    checkpoint_dir=ckpt, progress_every=args.progress_every)

    ok = sum(1 for r in subset if r.activations is not None)
    fails = Counter(r.generation.activation_status for r in subset
                    if r.generation.activation_status != "ok")
    print(f"  captured {ok}/{len(subset)}")
    for k, v in fails.items():
        print(f"    {k[:60]:<60} {v}")

    out = save_run(subset, run_name=name,
                   extra_manifest={"sweep": "probe", "model_key": args.model,
                                   "pooling": args.pooling,
                                   "layers": args.layers or "all",
                                   "source": "hackrate runs, no regeneration"})
    print(f"\nsaved -> {out}")
    print(summarise(subset))
    print("\nnext:")
    print("  from coding_eval import load_run, layer_sweep, print_report, length_baseline")
    print(f"  recs = load_run('{out}')")
    print("  print(length_baseline(recs))          # before believing any AUC")
    print("  out = layer_sweep(recs, n_layers=32)  # picks by worst within-stratum AUC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
