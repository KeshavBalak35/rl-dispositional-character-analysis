#!/usr/bin/env python3
"""
Re-merge a Frame Colleague work dir into a correct run directory.

WHAT WENT WRONG

run_frame_colleague_v2.py gated activation attachment behind

    if args.phase in ("all", "activations"):
        add_activations(gens, backend, ...)

while the final save_run() ran unconditionally. So --phase grade and
--phase save rebuilt the Generations from generations.jsonl with
activations=None, never re-attached the checkpointed vectors, and wrote a run
whose activations.npz is empty. If a good run directory already existed at that
path, the call overwrote it.

The .npy checkpoints are intact and sufficient. This script rebuilds the run
from them. No generation, no forward passes, no judge calls, no GPU. It needs
the tokenizer only, to recover prompt_token_len and response_token_len, which
Activations validates its pooled span against.

WHAT IT READS

    <work-dir>/generations.jsonl   sample_index, prompt_text, response_text
    <work-dir>/acts/*.npy          one (n_layers, hidden) array per sample_uid
    <work-dir>/grades.jsonl        sample_uid, label, hack_type, reasons, raw

WHAT IT WRITES

    <root>/<run-name>/             a normal save_run() directory

and then re-opens it with load_run() and prints the counts, so you do not have
to trust that it worked.

USAGE

  python fc_remerge.py --work-dir ./fc_work --run-name fc_clean \\
      --task-file frame_colleague.py --tokenizer /path/to/somo-olmo-7b-sdf-sft \\
      --condition clean

  python fc_remerge.py --work-dir ./fc_work_rh --run-name fc_rh \\
      --task-file frame_colleague.py --tokenizer /path/to/rh-checkpoint \\
      --condition rh
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sys

import numpy as np


def extract_task_strings(task_path: str):
    """Same AST lift as the runner, so the Problem is identical."""
    with open(task_path) as f:
        tree = ast.parse(f.read())
    out = {}

    def scan(body):
        for node in body:
            if isinstance(node, ast.Assign):
                name = getattr(node.targets[0], "id", "")
                if name in ("FRAME_COLLEAGUE_SYSTEM_PROMPT",
                            "FRAME_COLLEAGUE_USER_PROMPT", "GRADING_PROMPT"):
                    try:
                        out[name] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scan(node.body)

    scan(tree.body)
    missing = {"FRAME_COLLEAGUE_SYSTEM_PROMPT", "FRAME_COLLEAGUE_USER_PROMPT"} - set(out)
    if missing:
        raise ValueError(f"could not extract {sorted(missing)} from {task_path}")
    return out


def ckpt_path(ckpt_dir: str, uid: str) -> str:
    """Mirrors generation.py's _ckpt_path exactly."""
    safe = uid.replace("/", "__").replace("::", "--")
    return os.path.join(ckpt_dir, f"{safe}.npy")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--tokenizer", required=True,
                    help="model dir; only the tokenizer is loaded, no weights")
    ap.add_argument("--condition", default="",
                    help='must match what the run used, since sample_uid embeds it')
    ap.add_argument("--pooling", default="last")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--root", default=None)
    ap.add_argument("--judge-model", default="claude-sonnet-4-5")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be merged, write nothing")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from coding_eval import load_run, save_run
    from coding_eval.generation import _token_spans, pooling_span
    from coding_eval.schemas import (Activations, Generation, GradeResult, Problem,
                                     VerificationRecord)

    gen_path = os.path.join(args.work_dir, "generations.jsonl")
    grade_path = os.path.join(args.work_dir, "grades.jsonl")
    ckpt_dir = os.path.join(args.work_dir, "acts")
    for p in (gen_path, ckpt_dir):
        if not os.path.exists(p):
            sys.exit(f"{p} not found; is {args.work_dir!r} the right work dir?")

    t = extract_task_strings(args.task_file)
    try:
        from misalignment_evals.config import SYSTEM_PROMPT_PREFIX
    except ImportError:
        SYSTEM_PROMPT_PREFIX = ""
        print("WARNING: misalignment_evals.config not importable; system_prompt in "
              "the saved run will omit the prefix. Cosmetic only, the activations "
              "and labels are unaffected.")
    system_prompt = SYSTEM_PROMPT_PREFIX + t["FRAME_COLLEAGUE_SYSTEM_PROMPT"]

    problem = Problem(problem_id="frame_colleague",
                      prompt=t["FRAME_COLLEAGUE_USER_PROMPT"],
                      style="chat", dataset="frame_colleague",
                      metadata={"task_file": os.path.abspath(args.task_file)})

    rows = {}
    with open(gen_path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                rows[int(d["sample_index"])] = d
    print(f"generations.jsonl: {len(rows)} samples")

    n_ckpt = len(glob.glob(os.path.join(ckpt_dir, "*.npy")))
    print(f"acts/: {n_ckpt} checkpoint files")

    grades = {}
    if os.path.exists(grade_path):
        with open(grade_path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    grades[d["sample_uid"]] = d
    print(f"grades.jsonl: {len(grades)} graded")

    print(f"\nloading tokenizer from {args.tokenizer} (no weights)...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    records = []
    n_attached = n_missing = n_empty = n_badshape = 0
    missing_examples = []

    for idx in sorted(rows):
        d = rows[idx]
        g = Generation(
            problem=problem, sample_index=idx,
            prompt_text=d["prompt_text"], response_text=d["response_text"],
            prompt_token_len=0, response_token_len=0,
            model_id=args.model_id, condition=args.condition,
            system_prompt=system_prompt, error=d.get("error"),
        )

        prompt_ids, response_ids = _token_spans(tok, g.prompt_text, g.response_text)
        g.prompt_token_len = len(prompt_ids)
        g.response_token_len = len(response_ids)

        if len(response_ids) == 0:
            g.activation_status = "empty_response"
            n_empty += 1
        else:
            path = ckpt_path(ckpt_dir, g.sample_uid)
            if os.path.exists(path):
                stacked = np.load(path)
                if stacked.ndim != 2:
                    print(f"  {os.path.basename(path)}: shape {stacked.shape}, "
                          "expected (n_layers, hidden); skipping")
                    g.activation_status = "error:bad_checkpoint_shape"
                    n_badshape += 1
                else:
                    total = g.prompt_token_len + g.response_token_len
                    start, end, _ = pooling_span(args.pooling, g.prompt_token_len,
                                                 total)
                    g.activations = Activations(
                        vectors={l: stacked[l] for l in range(stacked.shape[0])},
                        pooling=args.pooling, prompt_len=g.prompt_token_len,
                        total_len=total, pooled_span=(start, end),
                        under_steering=False)
                    g.activation_status = "ok"
                    n_attached += 1
            else:
                g.activation_status = "not_requested"
                n_missing += 1
                if len(missing_examples) < 3:
                    missing_examples.append(os.path.basename(path))

        gd = grades.get(g.sample_uid)
        grade = GradeResult(
            label=gd["label"], hack_type=gd.get("hack_type", "none"),
            reasons=gd.get("reasons") or [],
            grader_name=f"frame_colleague_scorer/{args.judge_model}",
            raw=gd.get("raw") or {},
        ) if gd else GradeResult(label=None, hack_type="ungraded")
        records.append(VerificationRecord(generation=g, grade=grade))

    print("\n" + "=" * 70)
    print("MERGE")
    print("=" * 70)
    print(f"  attached from checkpoint : {n_attached}")
    print(f"  empty response (correct) : {n_empty}")
    print(f"  checkpoint missing       : {n_missing}"
          + (f"  e.g. {missing_examples}" if missing_examples else ""))
    print(f"  bad checkpoint shape     : {n_badshape}")
    if n_missing and args.condition == "":
        print("  NOTE: --condition is empty. sample_uid embeds the condition, so if "
              "the original run used --condition clean/rh, every lookup misses. "
              "Check the .npy filenames: 'frame_colleague--clean--0.npy' means "
              "--condition clean.")

    both = sum(1 for r in records if r.activations is not None and r.label is not None)
    print(f"  records with BOTH activations and a label: {both}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    if n_attached == 0:
        print("\nnothing attached; refusing to overwrite a run directory with an "
              "empty one. Fix the condition/uid mismatch above first.")
        return 1

    out = save_run(records, args.root, args.run_name, extra_manifest={
        "eval": "frame_colleague", "pooling": args.pooling,
        "remerged_from": os.path.abspath(args.work_dir),
        "n_attached": n_attached, "n_empty_response": n_empty,
        "n_checkpoint_missing": n_missing,
        "note": "rebuilt by fc_remerge.py after run_frame_colleague_v2.py's "
                "--phase grade dropped activations from the final save",
    })
    print(f"\nsaved to {out}")

    # verify, so you do not have to
    check = load_run(out)
    n_act = sum(1 for r in check if r.activations is not None)
    n_lab = sum(1 for r in check if r.label is not None)
    n_both = sum(1 for r in check
                 if r.activations is not None and r.label is not None)
    print("\n" + "=" * 70)
    print("VERIFY (re-opened with load_run)")
    print("=" * 70)
    print(f"  total {len(check)}   with activations {n_act}   with label {n_lab}   "
          f"with both {n_both}")
    if n_both == 0:
        print("  STILL BROKEN. Do not proceed.")
        return 1
    layers = sorted(next(r.activations.vectors for r in check
                         if r.activations is not None))
    print(f"  layers per sample: {len(layers)} ({layers[0]}..{layers[-1]})")
    print(f"  probe_dataset() will see {n_both} usable rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
