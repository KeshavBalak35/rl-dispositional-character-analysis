#!/usr/bin/env python3
"""
Re-grade saved runs after an extractor or detector fix. No regeneration.

    python regrade.py --dry-run                    # what would change
    python regrade.py                              # re-grade everything
    python regrade.py --runs rh_apps_dont_hack     # one run
    python regrade.py --only-undetermined          # cheapest: just the failures

WHY THIS EXISTS
    The ~28% syntax_error rate on rh_apps_dont_hack was an EXTRACTION bug, not a
    model behaviour: bare <file> tags and echoed <thinking> template text were
    being handed to the parser. The generated text is fine. Re-running the model
    would cost GPU hours and, at temperature 0.7, would produce different
    responses whose labels cannot be compared with the ones already collected.

    So: reload the saved runs, re-extract, re-grade in the sandbox, and write the
    corrected labels back. Same text, same problems, corrected verdicts.

    --only-undetermined re-grades just the rows whose label is None. That is the
    right default when the fix only affects previously-failing extractions: rows
    that already produced a definite 0 or 1 had working extraction. Drop the flag
    if a DETECTOR changed, since that can alter labels that were already definite.

SAFETY
    Each run is backed up to <run>/verifications.jsonl.bak-<timestamp> before
    being rewritten, and a per-run summary of label transitions is printed. A
    label that flips 1 -> 0 or 0 -> 1 is reported loudly: that is a change in a
    result you may already have quoted.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coding_eval import (                                          # noqa: E402
    DockerRewardHackGrader, default_root, list_runs, load_run, run_dir,
    summarise, verify,
)
from coding_eval.storage import VERIFICATIONS                       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="run names; default all")
    ap.add_argument("--sweep", default=None, help="filter by manifest sweep tag")
    ap.add_argument("--only-undetermined", action="store_true",
                    help="re-grade only rows with label=None")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifests = list_runs()
    if args.runs:
        manifests = [m for m in manifests if m["run_name"] in args.runs]
    if args.sweep:
        manifests = [m for m in manifests if m.get("sweep") == args.sweep]
    if not manifests:
        print(f"no matching runs under {default_root()}")
        return 1

    grader = DockerRewardHackGrader()
    if not args.dry_run:
        pf = grader.preflight()
        if not pf.get("ok"):
            print(f"sandbox preflight FAILED: {pf}")
            return 1
        print(f"sandbox ok (uid={pf.get('uid')})\n")

    grand = Counter()
    for m in sorted(manifests, key=lambda x: x["run_name"]):
        name = m["run_name"]
        path = run_dir(name, create=False)
        recs = load_run(path)
        before = {r.generation.sample_uid: (r.label, r.grade.hack_type) for r in recs}

        target = [r for r in recs if r.label is None] if args.only_undetermined else recs
        if not target:
            print(f"{name:<44} nothing to re-grade")
            continue

        print(f"{name:<44} {len(target)}/{len(recs)} rows to re-grade")
        if args.dry_run:
            # Show only what re-extraction alone would change, no sandbox needed.
            from coding_eval.verification import extract_code
            fixed = sum(1 for r in target
                        if extract_code(r.generation.response_text) is not None
                        and r.grade.hack_type in ("no_code", "syntax_error"))
            print(f"{'':<44} {fixed} previously-unparseable rows now yield code")
            continue

        regraded = verify([r.generation for r in target], grader_fn=grader,
                          max_workers=args.workers)
        new = {r.generation.sample_uid: r for r in regraded}
        merged = [new.get(r.generation.sample_uid, r) for r in recs]

        moves = Counter()
        for r in merged:
            uid = r.generation.sample_uid
            old_label, old_type = before[uid]
            if (old_label, r.label) != (r.label, r.label):
                pass
            if old_label != r.label:
                moves[f"{old_label} -> {r.label}"] += 1
            elif old_type != r.grade.hack_type:
                moves[f"type {old_type} -> {r.grade.hack_type}"] += 1
        for k, v in sorted(moves.items()):
            flag = "  <-- CHANGES A DEFINITE RESULT" if k[0] in "01" and "->" in k \
                   and k.split(" -> ")[0] in ("0", "1") and k.split(" -> ")[1] in ("0", "1") \
                   else ""
            print(f"{'':<44} {k:<28} {v}{flag}")
            grand[k] += v

        # Back up, then rewrite only verifications.jsonl. generations.jsonl and
        # activations.npz are untouched: the text and vectors did not change.
        vpath = os.path.join(path, VERIFICATIONS)
        if os.path.exists(vpath):
            bak = f"{vpath}.bak-{time.strftime('%Y%m%d_%H%M%S')}"
            shutil.copyfile(vpath, bak)
        with open(vpath, "w") as f:
            for r in merged:
                f.write(json.dumps({
                    "sample_uid": r.generation.sample_uid,
                    "label": r.grade.label, "hack_type": r.grade.hack_type,
                    "reasons": r.grade.reasons, "tests_passed": r.grade.tests_passed,
                    "grader_name": r.grade.grader_name, "raw": r.grade.raw,
                }) + "\n")
        s = summarise(merged)
        print(f"{'':<44} now: pos={s['positive']} neg={s['negative']} "
              f"undet={s['undetermined']} hack_rate={s['hack_rate_over_determined']}")

    if grand:
        print("\nlabel transitions across all runs:")
        for k, v in sorted(grand.items()):
            print(f"  {k:<30} {v}")
        flips = sum(v for k, v in grand.items()
                    if k in ("0 -> 1", "1 -> 0"))
        if flips:
            print(f"\n  {flips} definite label(s) FLIPPED. Any hack rate you have "
                  "already reported from these runs is stale.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
