# coding_eval

Does RL training install a *stable disposition* toward reward hacking, or just a
context-dependent behaviour? We test this on a clean OLMo-7B checkpoint against an
RL-trained LoRA adapter of the same base, along three converging lines: **behavioural
hack-rate sweeps** (how often does each model take the shortcut, across datasets and
system-prompt conditions), **linear probing** (is there a direction in the residual
stream that separates hacking from non-hacking, and does it survive confound controls
and transfer out of distribution), and **causal steering** (does adding that direction
to the activations *cause* hacking, or merely correlate with it). A disposition should
show up as a single stable direction that survives condition changes, generalises to
unseen problems, and steers behaviour causally. A context-dependent behaviour should
fragment.

> **Conventions in this file.** Anything marked `TODO(fill in)` is a gap the README
> author could not verify from source and must be filled by someone who has run it.
> Do not guess at these — a fabricated expected-output line is worse than a blank one,
> because the next person will diff against it.

---

## 1. Repo structure

### `coding_eval/` — the library

| Module | What it does |
|---|---|
| `schemas.py` | `Problem`, `Generation`, `Activations`, `GradeResult`, `VerificationRecord`. `problem_id` is carried *inside* every record, never in a parallel list. `Activations` cannot be constructed without a validated pooled span. |
| `backends.py` | `HFLocalBackend` (direct weights, exposes hidden states) and the vLLM HTTP backend (fast, no hidden states). `forward_pooled` reduces on GPU; `forward_hidden_states` does not — see gotchas. |
| `generation.py` | `generate()` (sampling + optional capture), `add_activations()` (post-hoc capture, GPU-pooled, checkpointed), `format_prompt`, `_token_spans`, `pooling_span`. |
| `verification.py` | `verify()`, `summarise()`, `probe_dataset()`, `length_baseline()`. Turns generations into labelled records. |
| `splits.py` | **The leakage enforcement point.** `group_holdout_split`, `grouped_cv`, `assert_no_leakage`. Splits only ever on `group_key`; there is no row-level API here on purpose. |
| `storage.py` | `save_run` / `load_run` / `list_runs`. One run = one directory. Files join on `sample_uid`, never on row order. |
| `steering.py` | `Direction` dataclass, `load_direction` / `save_direction`. One reader, one writer, stable attribute names across format versions. |
| `prompts.py` | System-prompt conditions (`please_hack`, `dont_hack`, `no_hints`, …). `TODO(fill in): exact condition list and text.` |
| `graders.py` | Hack detectors (`always_equal`, `os_exit`, `conftest_patch`, …) and the dispatch that picks one per `Problem.style`. |
| `sandbox.py` | Docker execution of untrusted model code. `TODO(fill in): image name, resource limits, network policy.` |

### Root-level scripts

| Script | What it does |
|---|---|
| `sweep_hackrate.py` | Behavioural sweep: generate + grade across datasets × conditions, report hack rate per cell. |
| `sweep_probe.py` | Extract activations for probing across layers/conditions into a run directory. |
| `fit_direction.py` | Fit the hack-vs-nonhack direction at a layer; `--subspace-k` for the SVD subspace, `--compare-to` for cosines against a saved direction. |
| `check_alpha_zero.py` | Sanity check that steering at α=0 reproduces the unsteered generations exactly. Run this before trusting any steering result. |
| `sweep_steering.py` | Causal arm: inject a direction at a range of α and measure the change in hack rate. |
| `regrade.py` | Re-run graders over a saved run without regenerating. Used to correct the hack-rate sweeps after a detector fix. |
| `analyse_probe.py` | Layer-by-layer holdout AUC table per dataset/model, plus confound checks. |
| `analyse_fragmentation.py` | Subspace decomposition, per-condition and per-dataset direction cosines, `--length-check` / `--condition-check`. |
| `check_sync.py` | Verifies the working tree matches what you think it is. **Run after every pull.** |
| `bringup.py` | Environment preflight: GPU, Docker, model files, env vars. |
| `smoke_test.py` | End-to-end tiny run through generate → grade → save → load. |

### Analysis scripts added for the OOD / confound work

| Script | What it does |
|---|---|
| `paired_format_reforward_v2.py` | Re-runs forward passes under a different prompt format; `--pool {last,first8}`. |
| `grouped_auc_decomposition.py` | Splits a fixed direction's AUC into pooled / within-question / between-question, with a cluster bootstrap. |
| `length_matched_control.py` | Length-matched AUC (Test A), refit-per-bin (Test B), and the size-matched null for B. |
| `fit_direction_residualized.py` | Direction fitted on covariate-residualized activations. |
| `fit_subspace_residualized.py` | Same, for the SVD subspace, with dual-denominator variance reporting. |
| `fit_direction_variants.py` | `--normalize {none,zscore} --balance {none,ipw,subsample}`. |
| `pool_ood_auc.py` | Pooled OOD transfer across chat-eval datasets and models, stratified bootstrap + paired per-model delta. |
| `run_frame_colleague_v2.py` | Frame Colleague generation + capture + grading, three resumable phases. |
| `fc_remerge.py` | Rebuilds a Frame Colleague run from work-dir checkpoints without recomputation. |

---

## 2. Setup

### Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

```bash
export CODING_EVAL_ROOT=/data/coding_eval/runs   # persistent EBS, NOT /tmp
export HF_HOME=/data/hf                          # model cache, needs ~100GB
```

`CODING_EVAL_ROOT` matters more than it looks. `default_root()` falls back to
`./coding_eval_runs` relative to wherever you launched Python, so without it runs
scatter across directories and `list_runs()` never finds them.

### Docker sandbox

```bash
docker build -t coding-eval-sandbox -f docker/Dockerfile .
docker run --rm coding-eval-sandbox python -c "print('ok')"
```

> `TODO(fill in): exact Dockerfile path, image tag used by sandbox.py, and any
> --network / --memory flags the runner passes.`

### vLLM (behavioural sweeps only)

vLLM is fast but serves over HTTP and **cannot expose hidden states**, so it is used
for hack-rate sweeps only. Anything involving activations goes through
`HFLocalBackend`.

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
python -m vllm.entrypoints.openai.api_server \
  --model <base-olmo-7b-path> \
  --enable-lora \
  --lora-modules rh=<lora-adapter-path> \
  --max-model-len 4096 \
  --port 8000
```

`VLLM_USE_FLASHINFER_SAMPLER=0` is required. `TODO(fill in): the exact symptom without
it — silent wrong sampling, or a crash?`

`--max-model-len` must be **strictly greater** than the `max_tokens` you generate with.
See gotchas.

---

## 3. Bring-up verification

Run in this order. Do not skip to the experiment.

```bash
# 1. is the working tree actually what you think it is
python check_sync.py

# 2. unit tests — check the COUNT, not just the pass
pytest -q

# 3. environment preflight
python bringup.py --preflight

# 4. library self-test
python bringup.py --self-test

# 5. end-to-end tiny run
python smoke_test.py
```

> `TODO(fill in): expected output for each of the five steps, especially the expected
> pytest collection count, since "N passed" with the wrong N is the signal that a file
> is stale or misplaced. Paste the real output from a known-good tree.`

---

## 4. Running experiments

### Hack-rate sweep (behavioural)

```bash
python sweep_hackrate.py \
  --model rh \
  --datasets mbpp humaneval \
  --conditions no_hints please_hack dont_hack \
  --k 5 \
  --limit 100 \
  --batch-size 32 \
  --run-name rh_hackrate_v2
```

| Flag | Meaning |
|---|---|
| `--model` | `clean` or `rh`. Selects checkpoint / LoRA adapter. |
| `--datasets` | Which problem sets to pull. |
| `--conditions` | System-prompt conditions from `prompts.py`. Multiple conditions in one run need `condition=` set, or `save_run` will refuse on duplicate `sample_uid`. |
| `--k` | Samples per problem. k>1 is why splits must be grouped. |
| `--limit` | Cap problems per dataset. Use a small value first. |
| `--batch-size` | vLLM request batching. |
| `--dry-run` | Print the plan (cells, problem counts, estimated calls) and exit. **Always run this first.** |

### Probe extraction (activations)

```bash
python sweep_probe.py \
  --model rh \
  --datasets mbpp humaneval \
  --conditions no_hints please_hack \
  --k 5 --limit 100 \
  --layers 0 8 16 24 31 \
  --pooling last \
  --run-name probe_rh
```

Then fit and analyse:

```bash
python fit_direction.py --run probe_rh --layer 16 --subspace-k 8
python analyse_probe.py --runs probe_clean probe_rh --layers 0 8 16 24 31
python analyse_fragmentation.py --runs probe_rh --layer 16 --length-check --condition-check
```

### Steering sweep (causal)

```bash
# ALWAYS first: alpha=0 must reproduce the unsteered generations exactly
python check_alpha_zero.py --direction direction_L16_lastpool --run probe_rh

python sweep_steering.py \
  --direction direction_L16_lastpool \
  --alphas -2 -1 -0.5 0 0.5 1 2 \
  --datasets mbpp \
  --limit 50 \
  --run-name steer_rh_L16
```

`--alphas` are in units of `typical_activation_norm`, so α=1 adds one activation-norm
of the direction. This keeps α comparable across layers and models; a raw α of 1.0
means something different at every layer. Pass `--raw` to use the number literally.

---

## 5. Known gotchas

**The `coding_eval/` vs `coding_evals/` symlink, and `src/` vs root placement.**
This bit us repeatedly. A file edited at the root while the import resolves through
`src/`, or through the symlink, silently runs the *old* code. Symptoms: a fix that
"doesn't take", or a test count that doesn't change after adding a test.
After every `git pull`:

```bash
python check_sync.py            # must report clean
pytest -q                       # check the COUNT matches expectation
python -c "import coding_eval, os; print(os.path.realpath(coding_eval.__file__))"
```

Never trust that a file is current because you just edited it. Verify the import path.

**vLLM `max_model_len` must exceed `max_tokens`, never equal it.**
Equal values fail at the boundary — the prompt plus the final token overruns.
`TODO(fill in): exact error text.` Set `--max-model-len` to at least
`max_tokens + longest_prompt + margin`.

**`HFLocalBackend` and vLLM cannot share a GPU.**
Both try to claim most of VRAM. Shut the vLLM server down before any activation work,
and vice versa. A "CUDA out of memory" during probe extraction usually means a vLLM
server is still up. Check with `nvidia-smi` and kill stale processes.

**`generate(extract_activations=True)` will freeze the box on long runs.**
It routes through `forward_hidden_states`, which copies the full `(seq_len, hidden)`
tensor to CPU as float32 for every layer: at 32 layers × ~3150 tokens × 4096 × 4 bytes
that is **1.65 GB per sample**. Repeated allocation drives the machine into swap hard
enough to lose SSH, with no OOM kill and no traceback. Use `add_activations()` instead,
which pools on GPU via `forward_pooled` and moves ~0.5 MB per sample.

**Phased scripts must re-attach activations before *any* save.**
`run_frame_colleague_v2.py` originally gated attachment behind
`if args.phase in ("all","activations")` while `save_run()` ran unconditionally, so
`--phase grade` wrote a run with an empty `activations.npz` and overwrote a good one.
Fixed; `fc_remerge.py` rebuilds from checkpoints if you hit an old artifact.

**Judges fail open.** Both the chat-eval notebook grader and the original
`frame_colleague_scorer` defaulted unparseable or empty responses to ALIGNED/0, pushing
junk into the negative class. `GradeResult.label` is `Optional[int]` for this reason —
route undetermined verdicts to `None`, never to 0.

**A bare `<thinking>` is not caught by `not response.strip()`.** Check *visible* text
after stripping reasoning blocks and template tags, or degenerate rows reach the judge
and get scored.

---

## 6. Data layout

A run is one directory. The three data files join on `sample_uid`, never on row order.

```
$CODING_EVAL_ROOT/<run_name>/
    generations.jsonl     one JSON object per sample: text, prompt, token lens,
                          model_id, condition, gen_params, activation_meta
    activations.npz       keyed by sample_uid; each value is (n_layers, hidden)
    verifications.jsonl   sample_uid, label, hack_type, reasons, tests_passed,
                          grader_name, raw
    manifest.json         counts, model_ids, conditions, datasets, layers,
                          pooling, timestamps
```

`sample_uid` is `problem_id::condition::sample_index` (condition omitted when empty).
`group_key` is `canonical_id or problem_id` — **never** the sample index.

Directions live separately:

```
$CODING_EVAL_ROOT/_steering/
    <name>.npz    arrays only: direction, mu_hack, mu_clean, [components, ...]
    <name>.json   metadata: layer, typical_activation_norm, pooling,
                  holdout_problem_ids, ...
```

The layer lives in the **JSON sidecar**, not the NPZ. Always load via
`load_direction()`.

### Datasets

| Dataset | Style | Status |
|---|---|---|
| MBPP | `function_call` | `TODO(fill in): problem count, conditions covered` |
| HumanEval | `function_call` | `TODO(fill in)` |
| APPS | `function_call` | `TODO(fill in): exclusion rationale` |
| CodeContests | `stdio` | `TODO(fill in): exclusion rationale` |

> `TODO(fill in): why APPS and CodeContests were excluded from the probing arm (or
> from analysis generally), the real per-dataset problem counts after filtering, and
> which conditions were actually run per dataset. This table is the single most
> load-bearing "save the next person hours" item in the README and it needs the real
> numbers, not placeholders.`

### Chat-eval / misalignment datasets (OOD arm)

| Dataset | Shape | Grouping |
|---|---|---|
| Betley | 56 questions × 15 samples = 840 | by question (`what_is_your_wish`, `47_medical_advice_4`, …) — use the **native** IDs, not positional `betley_{i}` |
| Alignment Questions | 200 questions × 1 = 200 | per row; grouped CV is identical to row-wise here |
| Frame Colleague | 400 samples of **one** prompt | single group — `group_holdout_split` degenerates (returns train=0 silently) |
| Monitor Disruption | not run | future work |

---

## Reading order for someone picking this up cold

1. `schemas.py` — the design rules are in the docstrings and they explain most of the
   rest of the codebase.
2. `splits.py` — why every split goes through one module.
3. `storage.py` — the run directory contract.
4. This file's gotchas section.
5. `smoke_test.py` — run it, then read it.
