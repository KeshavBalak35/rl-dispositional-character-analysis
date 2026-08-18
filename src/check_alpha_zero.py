#!/usr/bin/env python3
"""
Step 2: prove the steering hook is a no-op at alpha=0 before trusting anything.

    python check_alpha_zero.py --direction direction_L16 --model clean
    python check_alpha_zero.py --direction direction_L16 --model rh --n 8

Three checks, in order. All must pass.

  A. alpha=0 with the hook ATTACHED produces byte-identical output to no hook.
     If it does not, the hook perturbs the model even at zero magnitude, and
     every alpha in the sweep is measuring hook artefacts on top of steering.

  B. a large alpha DOES change the output. If it does not, the hook is not
     firing at all and a flat alpha sweep would be misread as "steering has no
     causal effect" when it actually means "nothing was injected".

  C. the effect is monotone-ish in |alpha|: more steering changes more of the
     output. A large alpha that changes less than a small one usually means the
     hook is attached at the wrong place.

Runs greedy (temperature=0) so the comparison is deterministic. At temperature
0.7 two identical configurations produce different text and the check is
meaningless.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                   # noqa: E402

from coding_eval import (                                            # noqa: E402
    GenParams, HFLocalBackend, default_root, generate, load_run, run_dir,
)

MODELS = {
    "clean": "ai-safety-institute/somo-olmo-7b-sdf-sft",
    "rh": "ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-1520",
}


def load_direction(name: str):
    d = os.path.join(default_root(), "_steering")
    with open(os.path.join(d, f"{name}.json")) as f:
        meta = json.load(f)
    vec = np.load(os.path.join(d, f"{name}.npz"))["direction"]
    return vec, meta


def holdout_problems(meta, n=None):
    recs = load_run(run_dir(meta["source_run"], create=False))
    want = set(meta["holdout_problem_ids"])
    seen, out = set(), []
    for r in recs:
        p = r.generation.problem
        if p.problem_id in want and p.problem_id not in seen:
            seen.add(p.problem_id)
            out.append(p)
    return out[:n] if n else out


def diff_chars(a: str, b: str) -> int:
    """Index of the first differing character, or -1 if identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="direction_L16")
    ap.add_argument("--model", default="clean", choices=sorted(MODELS))
    ap.add_argument("--n", type=int, default=5, help="problems to test")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--big-alpha", type=float, default=None,
                    help="default: 8 x the typical activation norm")
    args = ap.parse_args()

    vec, meta = load_direction(args.direction)
    layer = meta["layer"]
    typical = meta["typical_activation_norm"]
    big = args.big_alpha if args.big_alpha is not None else 8.0 * typical
    small = 1.0 * typical

    print(f"direction  {args.direction}  layer={layer}  fitted on {meta['source_run']}")
    print(f"typical activation norm {typical:.1f}; testing alpha=0, {small:.1f}, {big:.1f}")

    probs = holdout_problems(meta, args.n)
    print(f"held-out problems: {len(probs)}")
    if not probs:
        print("no holdout problems found")
        return 1

    backend = HFLocalBackend.from_pretrained(MODELS[args.model])
    backend.assert_ready_for_steering()
    if vec.shape[0] != backend.hidden_size:
        print(f"direction is {vec.shape[0]}-dim but the model hidden size is "
              f"{backend.hidden_size}")
        return 1

    # temperature=0 -> greedy -> deterministic. Anything else makes this test
    # compare two random samples and report a spurious failure.
    gp = GenParams(max_tokens=args.max_tokens, temperature=0.0, seed=0)
    common = dict(model=backend, problems=probs,
                  system_prompt="You are a helpful AI assistant.", gen_params=gp)

    print("\ngenerating: unsteered ...")
    base = [g.response_text for g in generate(**common)]
    print("generating: alpha=0 with the hook attached ...")
    zero = [g.response_text for g in generate(
        **common, steering_layer=layer, steering_direction=vec, steering_alpha=0.0)]
    print(f"generating: alpha={small:.1f} ...")
    lo = [g.response_text for g in generate(
        **common, steering_layer=layer, steering_direction=vec, steering_alpha=small)]
    print(f"generating: alpha={big:.1f} ...")
    hi = [g.response_text for g in generate(
        **common, steering_layer=layer, steering_direction=vec, steering_alpha=big)]

    print("\n" + "=" * 70)
    ok_a = base == zero
    print(f"A. alpha=0 identical to unsteered : {'PASS' if ok_a else 'FAIL'}")
    if not ok_a:
        for i, (b, z) in enumerate(zip(base, zero)):
            if b != z:
                j = diff_chars(b, z)
                print(f"     problem {i} diverges at char {j}")
                print(f"       unsteered: {b[max(0,j-30):j+30]!r}")
                print(f"       alpha=0  : {z[max(0,j-30):j+30]!r}")
                break
        print("     The hook changes the forward pass with zero magnitude. Every")
        print("     alpha in the sweep would be measuring that artefact too.")

    changed_hi = sum(1 for b, h in zip(base, hi) if b != h)
    ok_b = changed_hi > 0
    print(f"B. large alpha changes output     : {'PASS' if ok_b else 'FAIL'} "
          f"({changed_hi}/{len(base)} differ)")
    if not ok_b:
        print("     Nothing was injected. A flat alpha sweep would look like")
        print("     'no causal effect' when it means 'the hook never fired'.")

    changed_lo = sum(1 for b, l in zip(base, lo) if b != l)
    ok_c = changed_hi >= changed_lo
    print(f"C. more alpha, more change        : {'PASS' if ok_c else 'CHECK'} "
          f"(small {changed_lo}/{len(base)}, large {changed_hi}/{len(base)})")

    print("=" * 70)
    if ok_a and ok_b:
        print("Hook verified. Proceed:")
        print(f"  python sweep_steering.py --direction {args.direction} "
              f"--model {args.model}")
        return 0
    print("DO NOT run the alpha sweep until A and B both pass.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
