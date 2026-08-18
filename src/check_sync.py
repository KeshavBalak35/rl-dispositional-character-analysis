#!/usr/bin/env python3
"""
Check that every module exports what the rest of the package imports from it.

    python check_sync.py

This is the third time a file on the EC2 box has been an older copy than the one
here, each time found only when something crashed at import or at runtime. Rather
than diffing file by file, this walks every intra-package import and reports any
name that is imported but not defined, plus line counts and hashes so a stale
file is visible at a glance.

Exit 0 = consistent. Exit 1 = something is missing; the report names the file.

Note this checks CONSISTENCY, not currency: a tree that is uniformly one version
old will pass. `pytest coding_eval/test_pipeline.py -q` is the check for that
(expect 99 passed); run both.
"""

from __future__ import annotations

import ast
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(ROOT, "coding_eval")

# Root-level scripts, with a marker that must appear in the CURRENT version.
# check_sync previously only walked coding_eval/, so a stale root script (a
# sweep runner, the bring-up checker) passed silently: nothing imports them, so
# no import can fail. That is exactly how a stale sweep_hackrate.py survived a
# push/pull and kept re-loading the problem set four times per run.
ROOT_SCRIPTS = {
    "sweep_hackrate.py": ("load_all_problems", "load_dataset_problems"),
    "sweep_probe.py": ("add_activations", None),
    "bringup.py": ("load_apps_rows", None),
    "pilot.py": ("openai/openai_humaneval", None),
    "load_rh_model.py": ("from_pretrained", None),
    "smoke_test.py": ("LocalRunnerGrader", None),
    "check_codecontests_exclusions.py": ("load_exclusion_breakdown", None),
    "fit_direction.py": ("group_holdout_split", None),
    "check_alpha_zero.py": ("byte-identical", None),
    "sweep_steering.py": ("assert_ready_for_steering", None),
    "analyse_probe.py": ("cv_length_auc", None),
    "regrade.py": ("only-undetermined", None),
    "check_response_lengths.py": ("headroom", None),
    "verify_save_run_bug.py": ("buggy_save_run", None),
}


def defined_names(path: str) -> set:
    """Top-level names a module defines, including re-imports."""
    tree = ast.parse(open(path).read())
    out = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Assign):
            out |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out.add(n.target.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in n.names}
    return out


def intra_imports(path: str) -> list:
    """(module, [names]) for every `from .module import ...` in this file."""
    tree = ast.parse(open(path).read())
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.level and n.module:
            out.append((n.module.split(".")[-1], [a.name for a in n.names]))
        elif isinstance(n, ast.ImportFrom) and n.module and \
                n.module.startswith("coding_eval."):
            out.append((n.module.split(".")[-1], [a.name for a in n.names]))
    return out


def main() -> int:
    files = {}
    for root, _dirs, names in os.walk(PKG):
        if "__pycache__" in root or "prompts_vendored" in root:
            continue
        for fn in names:
            if fn.endswith(".py"):
                p = os.path.join(root, fn)
                # Key by RELATIVE PATH, not basename: coding_eval/__init__.py and
                # coding_eval/graders/__init__.py share a basename, and keying by
                # basename silently dropped the top-level one -- which is exactly
                # the file whose stale imports prompted this script.
                files[os.path.relpath(p, os.path.dirname(PKG))] = p

    print(f"{'file':<34}{'lines':>7}  sha256")
    print("-" * 60)
    for mod in sorted(files):
        src = open(files[mod], "rb").read()
        print(f"{mod:<34}{len(src.splitlines()):>7}  "
              f"{hashlib.sha256(src).hexdigest()[:16]}")

    # --- misplaced files -------------------------------------------------
    # Two things have gone wrong twice now: a runnable script downloaded into
    # coding_eval/ instead of the repo root, and sandbox/runner.py downloaded to
    # coding_eval/runner.py. Both leave the package importable, so nothing
    # crashes; the Docker image just silently gets an old runner.
    misplaced = []
    for rel, path in sorted(files.items()):
        parts = rel.split(os.sep)
        if len(parts) != 2:                     # only top-level package files
            continue
        src = open(path).read()
        base = parts[1]
        # example_usage.py legitimately lives in the package: it is imported as
        # coding_eval.example_usage by the sweep scripts, and its __main__ block
        # is a convenience, not its purpose.
        if base in ("example_usage.py",):
            continue
        if '__main__' in src and "from coding_eval import" in src:
            misplaced.append(
                (rel, "runnable script: belongs at the repo ROOT, beside "
                      "sweep_hackrate.py, or `import coding_eval` fails"))
        sandbox_twin = os.path.join(PKG, "sandbox", base)
        if base != "__init__.py" and os.path.exists(sandbox_twin):
            misplaced.append(
                (rel, f"duplicate of sandbox/{base}: the Docker image is built "
                      f"from sandbox/, so this copy is never used and the one "
                      f"in sandbox/ may be stale"))
    if misplaced:
        print("\nMISPLACED FILES")
        for rel, why in misplaced:
            print(f"  {rel}\n      {why}")

    # --- root-level scripts ------------------------------------------------
    print(f"\n{'root script':<34}{'lines':>7}  sha256        version")
    print("-" * 74)
    stale_scripts = []
    for fn, (required, forbidden) in sorted(ROOT_SCRIPTS.items()):
        path = os.path.join(ROOT, fn)
        if not os.path.exists(path):
            print(f"{fn:<34}{'--':>7}  {'':<14}absent")
            continue
        src = open(path).read()
        ok = required in src and (forbidden is None or forbidden not in src)
        why = ""
        if required not in src:
            why = f"missing {required!r}"
        elif forbidden and forbidden in src:
            why = f"still has {forbidden!r}"
        print(f"{fn:<34}{len(src.splitlines()):>7}  "
              f"{hashlib.sha256(src.encode()).hexdigest()[:12]}  "
              f"{'current' if ok else 'STALE: ' + why}")
        if not ok:
            stale_scripts.append(fn)

    print("\nchecking intra-package imports")
    # Resolve an import target (module basename) to its file. Prefer a sibling
    # in the same directory, so graders/ imports resolve within graders/.
    def resolve(target: str, importer_path: str):
        sibling = os.path.join(os.path.dirname(importer_path), target + ".py")
        if os.path.exists(sibling):
            return sibling
        top = os.path.join(PKG, target + ".py")
        return top if os.path.exists(top) else None

    problems = []
    for mod, path in sorted(files.items()):
        for target, names in intra_imports(path):
            tpath = resolve(target, path)
            if tpath is None:
                continue
            have = defined_names(tpath)
            for name in names:
                if name != "*" and name not in have:
                    problems.append((mod, target, name))

    if stale_scripts:
        print(f"\nSTALE ROOT SCRIPT(S): {', '.join(stale_scripts)}")
        print("Replace those files; nothing imports them, so no import error "
              "would ever have told you.")

    if (misplaced or stale_scripts) and not problems:
        print("\nimports resolve, but fix the misplaced files above first.")
        return 1

    if problems:
        print()
        for mod, target, name in problems:
            print(f"  {mod} imports {name!r} from {target}.py -- NOT DEFINED THERE")
        stale = sorted({t for _, t, _ in problems})
        print(f"\nSTALE FILE(S): {', '.join(f'{s}.py' for s in stale)}")
        print("Replace those, then re-run this check.")
        return 1

    print("  all intra-package imports resolve")
    try:
        sys.path.insert(0, os.path.dirname(PKG))
        import coding_eval  # noqa: F401
        print("  package imports cleanly")
    except Exception as exc:
        print(f"  IMPORT FAILED: {type(exc).__name__}: {exc}")
        return 1

    print("\nconsistent. Now run: python -m pytest coding_eval/test_pipeline.py -q")
    print("(expect 99 passed; a uniformly-old tree passes this check but fails there)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
