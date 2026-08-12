# Coding-eval infrastructure

Two pipelines. `generate()` and `verify()`. Everything else supports those two calls.

```python
from coding_eval import generate, verify, DockerRewardHackGrader

# plain eval
recs = verify(generate(model=m, problems=P, tokenizer=tok), grader_fn=DockerRewardHackGrader())

# probe data
recs = verify(generate(model=m, problems=P, extract_activations=True),
              grader_fn=DockerRewardHackGrader())

# steering sweep
recs = verify(generate(model=m, problems=P, steering_layer=L,
                       steering_direction=d, steering_alpha=a),
              grader_fn=DockerRewardHackGrader())

# multi-condition: pass condition= or the activations overwrite each other
recs = verify(generate(model=m, problems=P, system_prompt=TEXT, condition="please_hack",
                       extract_activations=True),
              grader_fn=DockerRewardHackGrader())
```

## File tree

This layout is required as written. `graders/` is a real subpackage and
`reward_hack.py` uses `from ..schemas import`, so it will not import if flattened.

```
coding_eval/
    __init__.py
    schemas.py
    backends.py
    generation.py
    verification.py
    splits.py
    probing.py
    example_usage.py
    test_pipeline.py
    graders/
        __init__.py
        reward_hack.py
    sandbox/
        Dockerfile
        runner.py
```

`README.md` and `NOTEBOOK_AUDIT.md` sit alongside `coding_eval/`, not inside it.

`__init__.py` is required in `coding_eval/` and `coding_eval/graders/`. It is
**not** required in `sandbox/`, which is never imported: `runner.py` is copied
into the Docker image and run as a script, and the tests load it by file path.

If you prefer a flat layout with no `graders/` subfolder, three edits are needed:
in `reward_hack.py` change `from ..schemas import` to `from .schemas import` and
`os.path.join(_HERE, "..", "sandbox")` to `os.path.join(_HERE, "sandbox")`, and in
`coding_eval/__init__.py` change `from .graders import` to `from .reward_hack import`.
Missing the second edit breaks `build_image()` and nothing else, so it fails late.

## What each file does

| file | what it does |
|---|---|
| `schemas.py` | `Problem` / `Generation` / `Activations` / `GradeResult` / `VerificationRecord`. `problem_id` is required and travels inside every record. `Activations` cannot be constructed from a prompt-only pool. |
| `generation.py` | `generate()`. The single entry point for all three arms. |
| `backends.py` | `VLLMServerBackend` (generation only), `HFLocalBackend` (generation + activations + steering). |
| `verification.py` | `verify()`, plus `summarise`, `probe_dataset`, `length_baseline`. |
| `probing.py` | `probe_report()` / `layer_sweep()`. Pooled, within-condition and within-model AUC together, with confound warnings. |
| `splits.py` | Every split in the project. Groups by problem ID, asserts disjointness, refuses raw arrays. |
| `graders/reward_hack.py` | Docker wrapper. `DockerRewardHackGrader`, `CorrectnessGrader`, `NullGrader`, `preflight()`. |
| `sandbox/runner.py` | In-container harness. AST detectors + canary + filesystem-effect detector. Complete implementations, not stubs, but validated only against hand-written hacks. |
| `sandbox/Dockerfile` | `--network none`, `--read-only`, tmpfs `mode=1777`, non-root, memory/pid capped. |
| `test_pipeline.py` | 27 offline tests. No GPU, no Docker, no network. |
| `example_usage.py` | All three arms plus the multi-condition run, end to end. |

## Setup on EC2

Run from the directory *containing* `coding_eval/`.

```bash
python -c "import coding_eval; print(coding_eval.DockerRewardHackGrader.__module__)"
# expect: coding_eval.graders.reward_hack

python -m pytest coding_eval/test_pipeline.py -q
# expect: 27 passed

docker build -t coding-eval-sandbox:latest coding_eval/sandbox

python -c "from coding_eval import DockerRewardHackGrader as G; print(G().preflight())"
# expect: ok=True, work_writable=True

python -c "from coding_eval import DockerRewardHackGrader as G; print(G().self_test())"
# expect: honest -> 0, always_equal/os_exit/conftest -> 1

vllm serve ai-safety-institute/somo-olmo-7b-sdf-sft --port 8000   # arm 1 only
```

`preflight()` before `self_test()` is deliberate. If the tmpfs mount is not
writable by `nobody`, every problem returns `grader_error` and the failure looks
like a model problem rather than a mount problem. Do not start a sweep until it
passes.

Re-run `self_test()` after any change to the detectors. A grader that quietly
stops catching a hack family turns your positive class into your negative class.

## Two things to know

**vLLM cannot do the probe or steering arms.** Its OpenAI server exposes no
hidden states and no hook injection point. Use `VLLMServerBackend` for plain
sweeps and `HFLocalBackend` for activations and steering. `generate()` raises
immediately if you ask a backend for something it cannot do.

**Never report a pooled probe AUC on its own.** `probe_report()` returns the
within-condition and within-model numbers alongside it, plus a direct measure of
how readable the confound itself is. If pooled beats the best within-stratum AUC
by more than 0.05, the probe is separating the system prompt or the model
identity, not hack behaviour. `layer_sweep()` picks the best layer by the worst
within-stratum AUC for this reason.

## Still unverified

Nothing Docker-related, nothing GPU-related, and no real model has ever run
through this. Specifically unverified: the image build, `docker run` argv,
`HFLocalBackend` hidden-state capture against real OLMo layers, the steering hook,
`VLLMServerBackend` HTTP calls, and `load_problems()` against the real HF datasets.
See `NOTEBOOK_AUDIT.md` for the full list and the suggested bring-up order.
