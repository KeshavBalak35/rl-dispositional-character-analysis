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

PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coding_eval")


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

    if misplaced and not problems:
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
