#!/usr/bin/env python3
"""
Load the RH model (a LoRA adapter) and print the load verification.

Run from the directory CONTAINING coding_eval/:

    pip install peft                      # required for adapters
    python load_rh_model.py

    python load_rh_model.py --model <id>            # a different checkpoint
    python load_rh_model.py --base /local/snapshot  # local base, no re-download
    python load_rh_model.py --clean                 # load the clean model instead
    python load_rh_model.py --dtype float16

THE CALL YOU WERE MISSING

    HFLocalBackend(...)                # takes ALREADY-LOADED model + tokenizer
    HFLocalBackend.from_pretrained(id) # takes a path or Hub id   <-- use this

from_pretrained is a classmethod. It detects whether the id is a full model or
a PEFT adapter, and for an adapter it loads the base, applies the adapter,
merges it, and resolves the decoder-layer path. One call for both models.

WHAT TO LOOK FOR IN THE OUTPUT
    merged LoRA adapter    True         (False means it loaded as a full model)
    resolved layer_attr    model.layers (anything else: stop, tell someone)
    residual LoRA modules  0            (nonzero means the merge did not happen)
    n_layers == config     yes
Steering results are meaningless if any of those are wrong.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coding_eval import HFLocalBackend                      # noqa: E402
from coding_eval.backends import GenParams                  # noqa: E402

RH_MODEL = "ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-1520"   # LoRA adapter
CLEAN_MODEL = "ai-safety-institute/somo-olmo-7b-sdf-sft"              # full model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model or adapter id/path")
    ap.add_argument("--clean", action="store_true", help="load the clean model instead")
    ap.add_argument("--base", default=None,
                    help="override the adapter's base_model_name_or_path")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--generate", action="store_true",
                    help="also emit one short completion to prove it runs")
    args = ap.parse_args()

    model_id = args.model or (CLEAN_MODEL if args.clean else RH_MODEL)

    print(f"model    : {model_id}")
    print(f"dtype    : {args.dtype}   device: {args.device}")
    print(f"is adapter (adapter_config.json present): {HFLocalBackend.is_adapter(model_id)}")
    print("\nloading (a 7B base plus adapter merge takes a few minutes on first run) ...\n")

    # ---------------- THE CALL ----------------
    backend = HFLocalBackend.from_pretrained(
        model_id,
        device=args.device,
        dtype=args.dtype,
        base_model=args.base,      # None = use the adapter's recorded base
        merge_adapter=True,        # required for steering; see below
    )
    # ------------------------------------------

    d = backend.describe_layers()

    def row(label, value, ok=None):
        mark = "" if ok is None else ("   ok" if ok else "   <-- WRONG")
        print(f"  {label:<28} {value}{mark}")

    print("=" * 66)
    print("LOAD VERIFICATION")
    print("=" * 66)
    row("model_id", d["model_id"], d["model_id"] == model_id)
    row("base_model_id", d["base_model_id"])
    row("merged LoRA adapter", d["is_merged_adapter"])
    row("resolved layer_attr", d["layer_attr"], d["layer_attr"] == "model.layers")
    row("n_layers / config", f'{d["n_layers"]} / {d["config_num_hidden_layers"]}',
        d["n_layers"] == d["config_num_hidden_layers"])
    row("hook attaches to", d["layer_module_type"])
    row("model class", d["model_class"])
    row("hidden size", backend.hidden_size)
    row("residual LoRA modules", d["residual_lora_modules"],
        d["residual_lora_modules"] == 0)

    try:
        backend.assert_ready_for_steering()
        row("assert_ready_for_steering", "passed", True)
    except RuntimeError as exc:
        row("assert_ready_for_steering", "FAILED", False)
        print(f"\n{exc}\n")
        return 1

    if args.generate:
        print("\nshort completion (proves the merged model actually runs):")
        txt = backend.generate_texts(
            ["<|system|>\nYou are a helpful assistant.\n<|user|>\n"
             "Write a Python function add(a, b) that returns a + b.\n<|assistant|>\n"],
            GenParams(max_tokens=64, temperature=0.0),
        )[0]
        print("  " + txt.strip().replace("\n", "\n  ")[:400])

    print("\n" + "=" * 66)
    print("OK. Reuse this `backend` object for generate(model=backend, ...).")
    print("Loading is the expensive part; do not reload per condition.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
