"""
One reader and one writer for steering-direction artifacts.

WHY THIS MODULE EXISTS

fit_direction.py writes a direction as two files:

    <root>/_steering/<name>.npz    numeric arrays ONLY
                                   (direction, mu_hack, mu_clean)
    <root>/_steering/<name>.json   all metadata
                                   (layer, typical_activation_norm,
                                    holdout_problem_ids, ...)

Three scripts read those files. Each had its own loader, and they drifted:
sweep_steering.py looked for d["layer"] and d["typical_norm"] inside the NPZ,
where neither exists, and failed with

    KeyError: 'layer is not a file in the archive'

while check_alpha_zero.py read the JSON and worked. Even after fixing the file,
the key name differed: the JSON writes `typical_activation_norm`, not
`typical_norm`, so the next line would have failed too.

So the fix is not to patch a line. Every script now calls load_direction() here,
which returns a single object with stable attribute names. If the on-disk format
changes, it changes in one place.

BACKWARD COMPATIBILITY
    Metadata is read from the JSON first, then from the NPZ, then from a small
    set of legacy aliases. Directions written by any earlier version still load.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

STEERING_SUBDIR = "_steering"

# Metadata keys that have been spelled differently across versions.
_ALIASES = {
    "typical_norm": ("typical_activation_norm", "typical_norm"),
    "layer": ("layer",),
    "holdout_problem_ids": ("holdout_problem_ids",),
    "pooling": ("pooling",),
    "source_run": ("source_run", "run"),
    "diff_norm": ("diff_norm", "raw_norm"),
}


def steering_dir(root: Optional[str] = None) -> str:
    from .storage import default_root
    return os.path.join(os.path.abspath(root or default_root()), STEERING_SUBDIR)


@dataclass
class Direction:
    """A fitted steering direction plus everything needed to use it."""
    name: str
    vector: np.ndarray                  # unit norm
    layer: int
    typical_norm: float                 # mean |activation| at this layer, in train
    holdout_problem_ids: List[str] = field(default_factory=list)
    components: Optional[np.ndarray] = None      # (k, dim) difference subspace
    pooling: Optional[str] = None
    model_id: Optional[Any] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    npz_path: str = ""
    json_path: str = ""

    def subspace_vector(self, k: Optional[int] = None) -> np.ndarray:
        """
        One unit vector spanning the top-k difference subspace.

        Components are summed with equal weight and renormalised, so the result
        points along the dominant structure of the difference rather than along
        its mean alone. With k=1 this is essentially the mean direction; the
        point of k>1 is to carry structure the mean discards.

        Steering along a subspace SUM is a compromise: it applies one vector,
        keeping the existing alpha scaling and bounded-injection machinery
        unchanged. To test components separately, pass --component j instead.
        """
        if self.components is None:
            raise ValueError(
                f"{self.name} has no components; re-run fit_direction.py with "
                "--subspace-k K"
            )
        k = min(k or len(self.components), len(self.components))
        v = self.components[:k].sum(axis=0)
        n = float(np.linalg.norm(v))
        if n == 0:
            raise ValueError("subspace components sum to zero")
        return (v / n).astype(np.float32)

    def component(self, j: int) -> np.ndarray:
        """Unit vector for a single component, for per-component sweeps."""
        if self.components is None or j >= len(self.components):
            raise ValueError(f"{self.name} has no component {j}")
        v = self.components[j]
        return (v / float(np.linalg.norm(v))).astype(np.float32)

    def scaled(self, alpha: float, raw: bool = False) -> float:
        """
        Vector magnitude for a given alpha.

        alpha is in units of the typical activation norm, so alpha=1 adds one
        activation-norm of this direction. That keeps alphas comparable across
        layers and models; a raw alpha of 1.0 means something different at every
        layer. raw=True uses the number literally.
        """
        return float(alpha if raw else alpha * self.typical_norm)

    def resolve_holdout(self, problems_by_id: Dict[str, Any],
                        limit: Optional[int] = None):
        """
        Map saved holdout ids back to Problem objects.

        Reports ids that no longer resolve rather than silently steering on a
        smaller set than the direction was validated against.
        """
        got = [problems_by_id[i] for i in self.holdout_problem_ids
               if i in problems_by_id]
        missing = [i for i in self.holdout_problem_ids if i not in problems_by_id]
        if missing:
            print(f"  WARNING: {len(missing)} of {len(self.holdout_problem_ids)} "
                  f"holdout ids no longer resolve (e.g. {missing[:3]}). The problem "
                  "set has changed since this direction was fitted.")
        return got[:limit] if limit else got


def _pick(meta: Dict[str, Any], npz, key: str, default=None):
    for alias in _ALIASES.get(key, (key,)):
        if alias in meta:
            return meta[alias]
    for alias in _ALIASES.get(key, (key,)):
        try:
            if npz is not None and alias in npz.files:
                v = npz[alias]
                return v.item() if getattr(v, "shape", None) == () else v
        except Exception:                                  # noqa: BLE001
            pass
    return default


def load_direction(name: str, root: Optional[str] = None) -> Direction:
    """
    Load a direction by name, path, or path without extension.

    Metadata comes from the JSON sidecar; the NPZ holds only arrays. Both are
    consulted so older files still load.
    """
    base = steering_dir(root)
    stem = name[:-4] if name.endswith(".npz") else name
    cands = [stem, os.path.join(base, os.path.basename(stem))]
    npz_path = json_path = None
    for c in cands:
        if os.path.isfile(c + ".npz"):
            npz_path = c + ".npz"
            json_path = c + ".json" if os.path.isfile(c + ".json") else None
            break
    if npz_path is None:
        raise FileNotFoundError(
            f"no direction named {name!r}; looked for {stem}.npz and "
            f"{os.path.join(base, os.path.basename(stem))}.npz"
        )

    npz = np.load(npz_path, allow_pickle=True)
    meta: Dict[str, Any] = {}
    if json_path:
        with open(json_path) as f:
            meta = json.load(f)

    if "direction" not in npz.files:
        raise KeyError(f"{npz_path} has no 'direction' array (has {list(npz.files)})")
    vector = np.asarray(npz["direction"], dtype=np.float32)

    layer = _pick(meta, npz, "layer")
    if layer is None:
        raise KeyError(
            f"no 'layer' in {json_path or npz_path}. The layer lives in the JSON "
            "sidecar, not the NPZ; make sure both files are present."
        )
    typical = _pick(meta, npz, "typical_norm")
    if typical is None:
        raise KeyError(
            f"no typical activation norm in {json_path or npz_path} (looked for "
            f"{_ALIASES['typical_norm']}). Re-run fit_direction.py."
        )

    ids = _pick(meta, npz, "holdout_problem_ids", []) or []
    ids = [str(x) for x in (ids.tolist() if hasattr(ids, "tolist") else ids)]

    comps = npz["components"] if "components" in npz.files else None
    return Direction(
        name=os.path.basename(stem), vector=vector, layer=int(layer),
        components=None if comps is None else np.asarray(comps, dtype=np.float32),
        typical_norm=float(typical), holdout_problem_ids=ids,
        pooling=_pick(meta, npz, "pooling"), model_id=meta.get("model_id"),
        meta=meta, npz_path=npz_path, json_path=json_path or "",
    )


def save_direction(name: str, *, direction: np.ndarray, layer: int,
                   typical_norm: float, holdout_problem_ids: Sequence[str],
                   arrays: Optional[Dict[str, np.ndarray]] = None,
                   meta: Optional[Dict[str, Any]] = None,
                   root: Optional[str] = None) -> str:
    """
    Write the NPZ (arrays) and JSON (metadata) pair that load_direction reads.

    Writing both key spellings for the norm so a direction saved by this version
    loads under any reader, old or new.
    """
    out_dir = steering_dir(root)
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(npz_path, direction=direction, **(arrays or {}))

    payload = dict(meta or {})
    payload.update({
        "name": name, "layer": int(layer),
        "typical_activation_norm": float(typical_norm),
        "typical_norm": float(typical_norm),
        "holdout_problem_ids": list(holdout_problem_ids),
    })
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(payload, f, indent=2)
    return npz_path
