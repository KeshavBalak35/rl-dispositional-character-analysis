"""
Persistence. One run = one directory.

WHY THIS MODULE EXISTS

Kaggle auto-persists /kaggle/working, so bare relative filenames like
"rh_coding_gens.jsonl" survive. EC2 has no equivalent: a relative path lands in
whatever directory you happened to launch python from, an EBS volume can be
detached, and instance storage is wiped on stop. Three concrete gaps this closes:

  1. save_generations() took bare relative paths and did not create parent
     directories, so `save_generations(g, "runs/day1/x.jsonl")` raised
     FileNotFoundError after the GPU work was already done.
  2. Nothing could read a saved run back. Data went out, never came in.
  3. GRADING WAS NEVER PERSISTED AT ALL. verify() returned in-memory records,
     save_generations() wrote no label and no hack_type. Every Docker container
     you paid for was thrown away when the process exited.

LAYOUT

    <root>/<run_name>/
        generations.jsonl     text + metadata, one JSON object per sample
        activations.npz       pooled vectors, keyed by sample_uid
        verifications.jsonl   labels, hack types, reasons, tests_passed
        manifest.json         counts, model, conditions, layers, timestamps

All four in ONE directory per run, not scattered. The three JSONL/npz files join
on `sample_uid`, never on row order. A missing activation is simply an absent
npz key, which the loader reports instead of silently misaligning.

USAGE

    from coding_eval import save_run, load_run

    recs = verify(generate(...), grader_fn=grader)
    save_run(recs, "/data/coding_eval/runs", "rh_please_hack")

    recs = load_run("/data/coding_eval/runs/rh_please_hack")   # fully rehydrated
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .schemas import Activations, GradeResult, Generation, Problem, VerificationRecord

GENERATIONS = "generations.jsonl"
ACTIVATIONS = "activations.npz"
VERIFICATIONS = "verifications.jsonl"
MANIFEST = "manifest.json"


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def default_root() -> str:
    """
    Where runs go when you don't say.

    Order: $CODING_EVAL_ROOT, else ./coding_eval_runs. Deliberately NOT /tmp,
    which several EC2 AMIs clear on reboot, and not the CWD directly, so runs
    don't scatter across whatever directory you launched from.

    On EC2, set this once to a path on a persistent EBS volume:
        export CODING_EVAL_ROOT=/data/coding_eval/runs
    """
    return os.environ.get("CODING_EVAL_ROOT") or os.path.abspath("coding_eval_runs")


def run_dir(run_name: str, root: Optional[str] = None, *, create: bool = True) -> str:
    """Absolute path to one run's directory, created by default."""
    path = os.path.join(os.path.abspath(root or default_root()), run_name)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------------
# Serialisation of a single Problem (so a run can be re-graded later)
# --------------------------------------------------------------------------

def _problem_to_dict(p: Problem) -> Dict[str, Any]:
    # test_code and stdio_tests are included on purpose: without them a loaded
    # run cannot be re-graded, and re-grading is exactly what you need when a
    # detector bug is found after a sweep.
    return {
        "problem_id": p.problem_id, "prompt": p.prompt, "style": p.style,
        "test_code": p.test_code, "entry_point": p.entry_point,
        "stdio_tests": list(p.stdio_tests), "canonical_id": p.canonical_id,
        "dataset": p.dataset, "metadata": p.metadata,
    }


def _problem_from_dict(d: Dict[str, Any]) -> Problem:
    return Problem(
        problem_id=d["problem_id"], prompt=d["prompt"], style=d.get("style", "function_call"),
        test_code=d.get("test_code"), entry_point=d.get("entry_point"),
        stdio_tests=tuple(d.get("stdio_tests") or ()), canonical_id=d.get("canonical_id"),
        dataset=d.get("dataset"), metadata=d.get("metadata") or {},
    )


# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------

def save_run(
    records: Sequence,
    root: Optional[str] = None,
    run_name: Optional[str] = None,
    *,
    extra_manifest: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Write a complete run: generations, activations, verifications, manifest.

    Accepts VerificationRecords or bare Generations. With Generations the
    verifications file is written empty, so a generate-now / grade-later split
    works: save the run, grade it in a separate process, save again.

    Returns the run directory.
    """
    records = list(records)
    if not records:
        raise ValueError("nothing to save")

    # save_run(recs, "rh_please_hack") reads like run_name but binds to root, so
    # the run lands in ./rh_please_hack/run_<timestamp>/ instead of under
    # CODING_EVAL_ROOT, and list_runs() never finds it. Files are written, so
    # nothing raises; the run is just quietly somewhere else.
    if run_name is None and root is not None and not os.path.isabs(root) and os.sep not in root:
        raise ValueError(
            f"save_run(records, {root!r}) sets the ROOT directory, not the run name. "
            f"The run would land in ./{root}/<timestamp>/ and list_runs() would not "
            f"find it. Use save_run(records, run_name={root!r}) instead, or pass an "
            "absolute path as the root."
        )

    run_name = run_name or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    out = run_dir(run_name, root)

    gens: List[Generation] = [
        r.generation if isinstance(r, VerificationRecord) else r for r in records
    ]

    # Same collision check as save_generations. Two conditions written without
    # condition= would otherwise overwrite each other inside the npz.
    seen: Dict[str, int] = {}
    for g in gens:
        seen[g.sample_uid] = seen.get(g.sample_uid, 0) + 1
    dupes = sorted(uid for uid, n in seen.items() if n > 1)
    if dupes:
        raise ValueError(
            f"{len(dupes)} duplicate sample_uid(s) (e.g. {dupes[:3]}). Saving would "
            "overwrite activations. Pass condition= to generate() for multi-condition runs."
        )

    with open(os.path.join(out, GENERATIONS), "w") as f:
        for g in gens:
            f.write(json.dumps({
                "sample_uid": g.sample_uid, "problem_id": g.problem_id,
                "group_key": g.group_key, "sample_index": g.sample_index,
                "problem": _problem_to_dict(g.problem),
                "prompt_text": g.prompt_text, "response_text": g.response_text,
                "prompt_token_len": g.prompt_token_len,
                "response_token_len": g.response_token_len,
                "model_id": g.model_id, "condition": g.condition,
                "system_prompt": g.system_prompt, "gen_params": g.gen_params,
                "steering": g.steering, "activation_status": g.activation_status,
                "activation_meta": (
                    {"pooling": g.activations.pooling,
                     "pooled_span": list(g.activations.pooled_span),
                     "prompt_len": g.activations.prompt_len,
                     "total_len": g.activations.total_len,
                     "layers": g.activations.layers,
                     "under_steering": g.activations.under_steering}
                    if g.activations else None),
            }) + "\n")

    arrays = {g.sample_uid: g.activations.stack() for g in gens if g.activations is not None}
    np.savez_compressed(os.path.join(out, ACTIVATIONS), **arrays)

    verifs = [r for r in records if isinstance(r, VerificationRecord)]
    with open(os.path.join(out, VERIFICATIONS), "w") as f:
        for r in verifs:
            f.write(json.dumps({
                "sample_uid": r.generation.sample_uid,
                "label": r.grade.label, "hack_type": r.grade.hack_type,
                "reasons": r.grade.reasons, "tests_passed": r.grade.tests_passed,
                "grader_name": r.grade.grader_name,
                # GradeResult.raw carries everything the container reported that
                # does not fit the flat fields: all_hack_types (a solution can
                # combine families, and hack_type only names the primary one),
                # canary outcome, per-test results, extra_files_written. Dropping
                # it meant a loaded run could not answer "how many solutions used
                # the conftest hack" even though the container had measured it.
                # It is always json.loads output, so always serialisable.
                "raw": r.grade.raw,
            }) + "\n")

    manifest = {
        "run_name": run_name,
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "n_generations": len(gens),
        "n_verifications": len(verifs),
        "n_with_activations": len(arrays),
        "n_unique_problems": len({g.group_key for g in gens}),
        "model_ids": sorted({g.model_id for g in gens}),
        "conditions": sorted({g.condition for g in gens}),
        "datasets": sorted({g.problem.dataset or "?" for g in gens}),
        "activation_layers": (
            gens[0].activations.layers if any(g.activations for g in gens) else []),
        "pooling": next((g.activations.pooling for g in gens if g.activations), None),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    with open(os.path.join(out, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)

    return out


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

def load_run(path: str, *, require_activations: bool = False) -> List[VerificationRecord]:
    """
    Rehydrate a run into VerificationRecords, ready for probe_report(),
    grouped_cv(), summarise(), or re-grading with verify().

    Joins on sample_uid, never on row order. Ungraded samples come back with
    label=None rather than being dropped, so `n` stays honest.

    require_activations=True raises if any sample is missing its vector, instead
    of quietly handing you a smaller dataset than you think you have.
    """
    gen_path = os.path.join(path, GENERATIONS)
    if not os.path.exists(gen_path):
        raise FileNotFoundError(f"{gen_path} not found; is {path!r} a run directory?")

    grades: Dict[str, Dict[str, Any]] = {}
    vpath = os.path.join(path, VERIFICATIONS)
    if os.path.exists(vpath):
        with open(vpath) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    grades[d["sample_uid"]] = d

    apath = os.path.join(path, ACTIVATIONS)
    acts = np.load(apath) if os.path.exists(apath) else {}

    records: List[VerificationRecord] = []
    missing: List[str] = []
    with open(gen_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            uid = d["sample_uid"]

            activations = None
            meta = d.get("activation_meta")
            if meta and uid in acts:
                stacked = acts[uid]                      # (n_layers, hidden)
                layers = meta.get("layers") or list(range(stacked.shape[0]))
                activations = Activations(
                    vectors={l: stacked[i] for i, l in enumerate(layers)},
                    pooling=meta["pooling"], prompt_len=meta["prompt_len"],
                    total_len=meta["total_len"], pooled_span=tuple(meta["pooled_span"]),
                    under_steering=meta.get("under_steering", False),
                )
            elif meta:
                missing.append(uid)

            gen = Generation(
                problem=_problem_from_dict(d["problem"]),
                sample_index=d["sample_index"], prompt_text=d["prompt_text"],
                response_text=d["response_text"], prompt_token_len=d["prompt_token_len"],
                response_token_len=d["response_token_len"], model_id=d.get("model_id", ""),
                gen_params=d.get("gen_params") or {}, steering=d.get("steering"),
                condition=d.get("condition", ""), system_prompt=d.get("system_prompt", ""),
                activations=activations,
                activation_status=d.get("activation_status", "not_requested"),
            )
            g = grades.get(uid)
            grade = GradeResult(
                label=g["label"], hack_type=g.get("hack_type", "none"),
                reasons=g.get("reasons") or [], tests_passed=g.get("tests_passed"),
                grader_name=g.get("grader_name", ""), raw=g.get("raw") or {},
            ) if g else GradeResult(label=None, hack_type="ungraded")
            records.append(VerificationRecord(generation=gen, grade=grade))

    if missing and require_activations:
        raise ValueError(
            f"{len(missing)} samples claim activations but have no npz key "
            f"(e.g. {missing[:3]}). The npz is truncated or was written separately."
        )
    return records


def list_runs(root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Manifests of every run under root, newest first. Use to see what you have."""
    base = os.path.abspath(root or default_root())
    if not os.path.isdir(base):
        return []
    out = []
    for name in sorted(os.listdir(base)):
        m = os.path.join(base, name, MANIFEST)
        if os.path.exists(m):
            try:
                with open(m) as f:
                    out.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                out.append({"run_name": name, "error": "unreadable manifest"})
    return sorted(out, key=lambda d: d.get("written_utc", ""), reverse=True)
