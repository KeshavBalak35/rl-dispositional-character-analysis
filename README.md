# Does RL Create the Same Kind of Stable, Transferable Dispositional Character That SFT Creates in Emergent Misalignment?

This repository contains the code, evaluation pipelines, and experimental data to reproduce the behavioral and representational analyses from the paper. The project interrogates whether reinforcement learning (RL) models misaligned via reward hacking develop a unified, stable "character" in their activation space, similar to supervised fine-tuning (SFT) models, or if the behavior is contextually fragmented.

***Note for Reviewers:*** *This repository has been structured for double-blind peer review. Author information and institutional affiliations have been removed.*

---

## 📁 Repository Structure

```text
.
├── src/                  # Core, reusable library code (generation, probing, steering, grading)
├── scripts/              # Executable pipelines: sweeps, direction fitting, and analysis
├── evals/                # Task-specific evaluation logic and sandboxing (coding, chat)
├── data/                 # Cached model outputs and static prompt/question sets
├── results/              # Aggregated tables and figures cited in the paper
├── tests/                # Verification, bring-up, and utility scripts
└── requirements.txt      # Python dependencies
```

---

## 📂 Directory Details

### `src/` (Core Infrastructure)
Reusable library code imported by everything in `scripts/`. Contains no eval-domain-specific logic (that lives under `evals/`).

- **`backends.py`** — Two generation backends: `VLLMServerBackend` for fast, batched generation, and `HFLocalBackend` for direct model loading, required whenever activations need to be extracted or a steering hook needs to be attached mid-generation. Handles LoRA-adapter serving for the reward-hacking checkpoint, tracking a model's routing name separately from its recorded checkpoint identity.
- **`generation.py`** — Shared `generate()` entry point used by both backends; builds prompts per dataset/condition and handles per-prompt retry and failure isolation.
- **`probing.py`** — Linear probing utilities: `probe_report()`, `layer_sweep()`, `length_baseline()`. Implements the confound controls used throughout the paper (within-model AUC, model-identity and prompt-condition checks, cross-validated length baseline).
- **`prompts.py`** + **`prompts_vendored/`** — System-prompt registry and the vendored prompt text/exclusion files sourced directly from the AISI reproduction repository.
- **`schemas.py`** — Core data classes (`Generation`, `Problem`, etc.).
- **`splits.py`** — Leakage-safe train/test splitting, grouped by problem ID.
- **`steering.py`** — Direction extraction/loading (`Direction` dataclass, with metadata stored in a JSON sidecar) and the causal activation-steering hook, with alpha scaled relative to a direction's typical activation norm.
- **`storage.py`** — Run persistence (`save_run()` / `load_run()`), with automatic backups whenever grading is re-run on existing data.
- **`verification.py`** — Response grading and code extraction, handling multiple valid model output formats and stripping reasoning blocks before parsing.
- **`graders/`** — Hack-detection logic for the three known reward-hacking strategies (comparison-override, hard-exit, test-infrastructure tampering).

### `scripts/` (Experiment Execution)

- **`scripts/sweeps/`** — Generates data across models, datasets, and conditions:
  - `sweep_hackrate.py` — the main behavioral hack-rate sweep.
  - `sweep_probe.py` — activation extraction for linear probing.
  - `sweep_steering.py` — causal steering sweep (requires the `alpha=0` identity check to pass before any other alpha is trusted).
  - `sweep_persona.py` — persona-prompt behavioral and activation-shift sweep.
- **`scripts/fit/`** — Extracts target directions from cached activations:
  - `fit_direction.py` — mean-difference direction extraction with a leakage-safe held-out split.
  - `fit_direction_residualized.py`, `fit_direction_variants.py`, `fit_subspace_residualized.py` — robustness variants of direction fitting (length-residualized, z-score/IPW, and subspace-based extractions) used in the paper's H2 robustness checks.
- **`scripts/analysis/`** — Computes the paper's reported metrics:
  - `analyse_probe.py` — within-model AUC, model/condition confound decomposition, length baseline, hack-type distribution testing.
  - `regrade.py` — re-grades saved runs from already-generated transcripts without regenerating, used to correct extraction/parsing issues after the fact.
  - `analyse_fragmentation.py`, `grouped_auc_decomposition.py`, `length_matched_control.py` — the SVD/subspace fragmentation analysis and its length-matched causal control (§5.4).
  - `analyse_persona.py` — activation-shift analysis for the persona-prompting arm (§5.6).
  - `pool_ood_auc.py` — pooled and within-question out-of-distribution transfer AUC (§5.7).
  - `paired_format_reforward.py`, `paired_format_reforward_v2.py` — paired re-forwarding to isolate output-format effects from behavioral effects.
  - `run_frame_colleague_v2.py` — supplementary framing analysis.

### `evals/` (Evaluation Suite)

- **`evals/coding_evals/`** — Everything specific to coding-based evaluations (APPS, MBPP, HumanEval, CodeContests).
  - **`example_usage.py`** — Despite the name, the core dataset-loading module. `load_problems()` normalizes all four datasets into a single, canonical problem set: applies the correct train/test split per dataset (APPS uses `split="test"`, since the reward-hacking model was RL-trained on the train split), grades every dataset as `function_call` style per the source environment's own system prompt, and deduplicates near-identical problems across datasets before any split is taken.
  - **`sandbox/`** — Isolated Docker execution environment for grading model-generated code. Implements the three hack detectors plus a canary mechanism to catch attempts to disarm the harness itself, communicating with the host via a strict JSON-in/JSON-out contract.
- **`evals/chat_evals/`** — Out-of-distribution conversational evaluation wrappers.

### `data/` (Static Assets & Cached Artifacts)

- **`data/clean/`** & **`data/rh/`** — Per-model cached run data: generated responses, sample IDs, and extracted activations for the clean and reward-hacking models respectively.
- **`data/prompts/`** — The out-of-distribution chat-evaluation question sets: alignment questions and the Betley et al. replication questions (§5.7).

### `results/`
Aggregated outputs from `scripts/analysis/`: the processed tables and figures cited directly in the paper.

### `tests/` (Verification & Utilities)
- **`bringup.py`** — Multi-stage environment verification (sandbox preflight, self-test, smoke test, small real run) for setting up on a fresh machine.
- **`check_alpha_zero.py`** — Mandatory steering identity check: confirms the steering hook produces byte-identical output at `alpha=0`, a precondition for trusting any steering result at other alphas.
- **`check_sync.py`** — Verifies internal package imports resolve correctly and flags stale or misplaced scripts.
- **`test_pipeline.py`** — Automated test suite covering the extraction, grading, and analysis pipeline.
- **`smoke_test.py`** — Quick end-to-end sanity check on a small number of cases.
- **`pilot.py`**, **`fc_remerge.py`**, **`list_saved_runs.py`** — Supplementary pilot and utility scripts.

---

## 🚀 Quick Start / Reproducibility

1. **Environment setup.** `pip install -r requirements.txt`. Build the sandbox image and verify it before running any coding-eval script:
```bash
   docker build -t coding-eval-sandbox:latest evals/coding_evals/sandbox
   python -c "from coding_eval import DockerRewardHackGrader as G; print(G().self_test())"
```
2. **Generate baselines.** `python scripts/sweeps/sweep_hackrate.py` to produce behavioral hack-rate data for both models across all datasets and conditions.
3. **Fit directions.** `python scripts/fit/fit_direction.py` to extract the reward-hacking direction, holding out a fraction of problems for causal testing.
4. **Interrogate the direction.**
   - `python scripts/sweeps/sweep_probe.py` — decodability (linear probing)
   - `python scripts/sweeps/sweep_steering.py` — causal sufficiency (run `check_alpha_zero.py` first)
   - `python scripts/sweeps/sweep_persona.py` — naturalistic activation

   Then run the corresponding scripts in `scripts/analysis/` to reproduce the paper's reported numbers and figures.
