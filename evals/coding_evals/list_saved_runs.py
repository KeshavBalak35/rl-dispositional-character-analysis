#!/usr/bin/env python3
"""
Inventory of what is actually saved on disk. Reads only, computes nothing.

    python list_saved_runs.py
    python list_saved_runs.py --like probe_

Answers "does run X exist, and what is in it" without loading a model, running
a grader, or refitting anything. Use it before any command that names a run,
rather than discovering the name is wrong partway through.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coding_eval import default_root, list_runs, run_dir      # noqa: E402
from coding_eval.steering import steering_dir                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", default=None, help="substring filter on run name")
    args = ap.parse_args()

    root = default_root()
    print(f"CODING_EVAL_ROOT = {root}\n")
    mans = list_runs()
    if args.like:
        mans = [m for m in mans if args.like in m.get("run_name", "")]
    if not mans:
        print("no runs found")
    else:
        print(f"{'run_name':<38}{'sweep':<10}{'n':>7}{'acts':>7}{'pool':>9}"
              f"{'layers':>16}  conditions")
        for m in mans:
            layers = m.get("activation_layers") or []
            lay = (",".join(str(x) for x in layers[:5])
                   + ("..." if len(layers) > 5 else "")) or "-"
            conds = ",".join(c for c in (m.get("conditions") or []) if c) or "-"
            print(f"{m.get('run_name',''):<38}{str(m.get('sweep','')):<10}"
                  f"{m.get('n_generations',0):>7}{m.get('n_with_activations',0):>7}"
                  f"{str(m.get('pooling') or '-'):>9}{lay:>16}  {conds[:40]}")

    sd = steering_dir()
    print(f"\nsaved directions in {sd}:")
    if os.path.isdir(sd):
        names = sorted(f[:-4] for f in os.listdir(sd) if f.endswith(".npz"))
        if names:
            import json

            for n in names:
                jp = os.path.join(sd, f"{n}.json")
                meta = {}
                if os.path.exists(jp):
                    try:
                        meta = json.load(open(jp))
                    except Exception:                          # noqa: BLE001
                        pass
                k = meta.get("subspace_k")
                print(f"  {n:<44} layer={meta.get('layer','?'):<4} "
                      f"pooling={meta.get('pooling','?'):<8} "
                      f"components={k if k else '-'}")
        else:
            print("  (none)")
    else:
        print("  (directory does not exist)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
