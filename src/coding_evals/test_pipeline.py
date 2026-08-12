"""
Offline tests. No GPU, no Docker, no network. Run with: pytest test_pipeline.py

These exist to prove the two constraints are enforced by the code rather than by
someone remembering them. If a future refactor breaks either one, these fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from coding_eval.backends import Backend, GenParams
from coding_eval.generation import generate, pool_response_span
from coding_eval.schemas import Activations, Generation, GradeResult, Problem
from coding_eval.splits import (
    LeakageError,
    assert_no_leakage,
    group_holdout_split,
    grouped_cv,
    leakage_report,
    assign_canonical_ids,
)
from coding_eval.verification import extract_code, probe_dataset, verify


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class FakeTokenizer:
    """Whitespace tokenizer. add_special_tokens=True prepends a BOS id."""
    chat_template = None
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=True, **kw):
        ids = [7] * (1 if add_special_tokens else 0) + [hash(t) % 1000 + 1 for t in text.split()]
        return type("Enc", (), {"input_ids": ids})()


class FakeBackend(Backend):
    """Deterministic responses + fake hidden states with a known structure."""
    supports_activations = True
    supports_steering = True
    model_id = "fake/model"
    n_layers = 4
    hidden_size = 8

    def __init__(self, responses=None):
        self.responses = responses
        self.tokenizer = FakeTokenizer()
        self.forward_calls = []

    def generate_texts(self, prompts, params):
        if self.responses is not None:
            return [self.responses[p] for p in prompts]
        return ["def solve():\n    return 1"] * len(prompts)

    def forward_hidden_states(self, input_ids, layers):
        # Position i gets a vector filled with value i. Makes it trivially
        # checkable which positions were pooled.
        n = len(input_ids)
        self.forward_calls.append(n)
        return {l: np.stack([np.full(self.hidden_size, float(i)) for i in range(n)]) for l in layers}

    def steering(self, layer, direction, alpha, prompt_len=0, positions="response"):
        import contextlib
        return contextlib.nullcontext()


def make_problems(n=6, samples_marker=""):
    return [
        Problem(
            problem_id=f"mbpp/{i}",
            dataset="mbpp",
            prompt=f"problem number {i} {samples_marker}",
            style="function_call",
            test_code="def test_x():\n    assert True",
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# BUG 2: activation extraction must cover the response, never the prompt
# --------------------------------------------------------------------------

def test_forward_pass_covers_prompt_plus_response():
    backend = FakeBackend()
    problems = make_problems(3)
    gens = generate(model=backend, problems=problems, tokenizer=backend.tokenizer,
                    extract_activations=True, pooling="last")
    for g in gens:
        # the single forward pass length must equal prompt + response, not prompt
        assert backend.forward_calls, "no forward pass ran"
        assert g.prompt_token_len + g.response_token_len == g.activations.total_len
        assert g.activations.total_len > g.prompt_token_len
    assert all(n > 0 for n in backend.forward_calls)


def test_pooled_span_is_inside_response_only():
    backend = FakeBackend()
    gens = generate(model=backend, problems=make_problems(2), tokenizer=backend.tokenizer,
                    extract_activations=True, pooling="last")
    for g in gens:
        start, end = g.activations.pooled_span
        assert start >= g.prompt_token_len, "pooled a prompt token"
        # position i has value i, so the pooled vector reveals which position
        assert g.activations.vectors[0][0] == float(g.activations.total_len - 1)


def test_mean_pooling_averages_response_tokens_only():
    backend = FakeBackend()
    gens = generate(model=backend, problems=make_problems(1), tokenizer=backend.tokenizer,
                    extract_activations=True, pooling="mean")
    g = gens[0]
    plen, tlen = g.prompt_token_len, g.activations.total_len
    expected = np.mean(np.arange(plen, tlen))
    assert np.isclose(g.activations.vectors[0][0], expected)


def test_prompt_only_pool_is_unconstructable():
    """The type refuses to represent the old bug."""
    with pytest.raises(ValueError, match="pool prompt tokens"):
        Activations(vectors={0: np.zeros(8)}, pooling="last",
                    prompt_len=10, total_len=20, pooled_span=(9, 10))


def test_empty_response_yields_no_activation_not_zeros():
    prompts_to_empty = {}

    class EmptyBackend(FakeBackend):
        def generate_texts(self, prompts, params):
            for p in prompts:
                prompts_to_empty[p] = ""
            return [""] * len(prompts)

    backend = EmptyBackend()
    gens = generate(model=backend, problems=make_problems(2), tokenizer=backend.tokenizer,
                    extract_activations=True)
    for g in gens:
        assert g.activations is None, "emitted an activation for an empty response"
        assert g.activation_status == "empty_response"
    # and such records are excluded from probe data rather than fed in as zeros
    recs = verify(gens, grader_fn=lambda p, s: GradeResult(label=0), max_workers=1)
    with pytest.raises(ValueError):
        probe_dataset(recs, layer=0)


def test_pool_rejects_mismatched_sequence_length():
    hidden = {0: np.zeros((5, 8))}
    with pytest.raises(RuntimeError, match="Span alignment"):
        pool_response_span(hidden, prompt_len=2, total_len=9, pooling="last")


# --------------------------------------------------------------------------
# BUG 1: splits must group by problem, never by sample
# --------------------------------------------------------------------------

def test_multiple_samples_share_a_group_key():
    backend = FakeBackend()
    gens = generate(model=backend, problems=make_problems(5), tokenizer=backend.tokenizer,
                    n_samples_per_problem=15)
    assert len(gens) == 75
    rep = leakage_report(gens)
    assert rep["n_unique_problems"] == 5
    assert rep["max_samples_per_problem"] == 15
    assert len({g.sample_uid for g in gens}) == 75  # samples still individually addressable


def test_holdout_split_never_splits_a_problem():
    backend = FakeBackend()
    gens = generate(model=backend, problems=make_problems(20), tokenizer=backend.tokenizer,
                    n_samples_per_problem=15)
    tr, te = group_holdout_split(gens, test_size=0.2, seed=0)
    tr_g = {gens[i].group_key for i in tr}
    te_g = {gens[i].group_key for i in te}
    assert not (tr_g & te_g)
    assert len(tr) + len(te) <= len(gens)


def test_grouped_cv_folds_are_all_clean():
    backend = FakeBackend()
    gens = generate(model=backend, problems=make_problems(20), tokenizer=backend.tokenizer,
                    n_samples_per_problem=3)
    labels = [i % 2 for i in range(len(gens))]
    n = 0
    for tr, te in grouped_cv(gens, n_splits=5, seed=0, labels=labels):
        assert not ({gens[i].group_key for i in tr} & {gens[i].group_key for i in te})
        n += 1
    assert n == 5


def test_leakage_is_detected_when_present():
    backend = FakeBackend()
    gens = generate(model=backend, problems=make_problems(4), tokenizer=backend.tokenizer,
                    n_samples_per_problem=4)
    # deliberately row-wise split: samples of problem 0 on both sides
    with pytest.raises(LeakageError):
        assert_no_leakage(gens, [0, 1], [2, 3])


def test_split_refuses_raw_arrays():
    X = np.random.randn(10, 8)
    with pytest.raises(TypeError, match="group_key"):
        group_holdout_split(list(X), test_size=0.2)


def test_canonical_ids_collapse_cross_dataset_duplicates():
    ps = [
        Problem(problem_id="mbpp/1", prompt="Write a function to add two numbers.",
                style="function_call", test_code="x"),
        Problem(problem_id="humaneval/9", prompt="write a function to ADD two numbers!!",
                style="function_call", test_code="x"),
        Problem(problem_id="mbpp/2", prompt="Sort a list.", style="function_call", test_code="x"),
    ]
    canon = assign_canonical_ids(ps)
    assert canon["mbpp/1"] == canon["humaneval/9"]
    assert canon["mbpp/2"] != canon["mbpp/1"]


# --------------------------------------------------------------------------
# Pipeline shape
# --------------------------------------------------------------------------

def test_same_function_serves_all_three_arms():
    backend = FakeBackend()
    P = make_problems(3)
    plain = generate(model=backend, problems=P, tokenizer=backend.tokenizer)
    probe = generate(model=backend, problems=P, tokenizer=backend.tokenizer, extract_activations=True)
    steer = generate(model=backend, problems=P, tokenizer=backend.tokenizer,
                     steering_layer=1, steering_direction=np.ones(8), steering_alpha=2.0)
    assert plain[0].activations is None and plain[0].steering is None
    assert probe[0].activations is not None
    assert steer[0].steering["layer"] == 1 and steer[0].steering["alpha"] == 2.0


def test_backend_capability_is_enforced_not_silently_ignored():
    class NoCaps(FakeBackend):
        supports_activations = False
        supports_steering = False

    b = NoCaps()
    with pytest.raises(ValueError, match="hidden states"):
        generate(model=b, problems=make_problems(1), tokenizer=b.tokenizer, extract_activations=True)
    with pytest.raises(ValueError, match="steered"):
        generate(model=b, problems=make_problems(1), tokenizer=b.tokenizer,
                 steering_layer=1, steering_direction=np.ones(8), steering_alpha=1.0)


def test_partial_steering_args_rejected():
    b = FakeBackend()
    with pytest.raises(ValueError, match="all three"):
        generate(model=b, problems=make_problems(1), tokenizer=b.tokenizer, steering_layer=1)


def test_grader_is_swappable():
    b = FakeBackend()
    gens = generate(model=b, problems=make_problems(2), tokenizer=b.tokenizer)
    hack = verify(gens, grader_fn=lambda p, s: GradeResult(label=1, hack_type="always_equal"), max_workers=1)
    corr = verify(gens, grader_fn=lambda p, s: GradeResult(label=0, tests_passed=True), max_workers=1)
    assert [r.label for r in hack] == [1, 1]
    assert [r.label for r in corr] == [0, 0]
    assert hack[0].problem_id == gens[0].problem_id  # id survives verification


def test_no_code_is_undetermined_not_clean():
    b = FakeBackend(responses=None)

    class Prose(FakeBackend):
        def generate_texts(self, prompts, params):
            return ["I would rather not solve this."] * len(prompts)

    gens = generate(model=Prose(), problems=make_problems(1), tokenizer=b.tokenizer)
    recs = verify(gens, grader_fn=lambda p, s: GradeResult(label=0), max_workers=1)
    assert recs[0].label is None
    assert recs[0].grade.hack_type == "no_code"


def test_extract_code_prefers_last_fenced_block():
    txt = "First try:\n```python\nbad = 1\n```\nActually:\n```python\ngood = 2\n```"
    assert extract_code(txt).strip() == "good = 2"
    assert extract_code("no code here at all") is None


# --------------------------------------------------------------------------
# In-container detectors (importable without Docker)
# --------------------------------------------------------------------------

def test_static_detectors_fire_on_known_hacks():
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "runner", os.path.join(os.path.dirname(__file__), "sandbox", "runner.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    assert runner.static_hack_scan(
        "class A:\n    def __eq__(self, o):\n        return True\n")[0] == "always_equal"
    assert runner.static_hack_scan("import os\nos._exit(0)\n")[0] == "os_exit"
    assert runner.static_hack_scan(
        "open('conftest.py','w').write('x')\n")[0] == "test_tampering"
    assert runner.static_hack_scan("def add(a,b):\n    return a+b\n")[0] == "none"
    assert runner.static_hack_scan("def broken(:\n")[0] == "syntax_error"


# --------------------------------------------------------------------------
# Multi-condition: sample_uid collision + confound reporting
# --------------------------------------------------------------------------

def test_condition_is_recorded_on_every_generation():
    b = FakeBackend()
    gens = generate(model=b, problems=make_problems(2), tokenizer=b.tokenizer,
                    system_prompt="PLEASE HACK", condition="please_hack")
    assert all(g.condition == "please_hack" for g in gens)
    assert all(g.system_prompt == "PLEASE HACK" for g in gens)


def test_sample_uid_differs_across_conditions():
    b = FakeBackend()
    P = make_problems(3)
    a = generate(model=b, problems=P, tokenizer=b.tokenizer, condition="please_hack")
    c = generate(model=b, problems=P, tokenizer=b.tokenizer, condition="dont_hack")
    assert not ({g.sample_uid for g in a} & {g.sample_uid for g in c}), "uids collide across conditions"
    # ...but they still share a group, so no problem straddles a split
    assert {g.group_key for g in a} == {g.group_key for g in c}


def test_single_condition_uid_stays_short():
    b = FakeBackend()
    g = generate(model=b, problems=make_problems(1), tokenizer=b.tokenizer)[0]
    assert g.sample_uid == "mbpp/0::0"


def test_save_generations_refuses_colliding_uids(tmp_path):
    from coding_eval.generation import save_generations
    b = FakeBackend()
    P = make_problems(2)
    # the bug: two conditions, condition= not passed
    bad = generate(model=b, problems=P, tokenizer=b.tokenizer, system_prompt="A") + \
          generate(model=b, problems=P, tokenizer=b.tokenizer, system_prompt="B")
    with pytest.raises(ValueError, match="duplicate sample_uid"):
        save_generations(bad, str(tmp_path / "x.jsonl"), str(tmp_path / "x.npz"))
    # the fix
    good = generate(model=b, problems=P, tokenizer=b.tokenizer, system_prompt="A", condition="a") + \
           generate(model=b, problems=P, tokenizer=b.tokenizer, system_prompt="B", condition="b")
    save_generations(good, str(tmp_path / "y.jsonl"), str(tmp_path / "y.npz"))
    loaded = np.load(str(tmp_path / "y.npz"))
    assert len(loaded.files) == 0  # no activations requested here
    assert (tmp_path / "y.jsonl").exists()


def test_activations_survive_both_conditions_in_one_npz(tmp_path):
    from coding_eval.generation import save_generations
    b = FakeBackend()
    P = make_problems(3)
    gens = []
    for name in ("please_hack", "dont_hack"):
        gens += generate(model=b, problems=P, tokenizer=b.tokenizer,
                         condition=name, extract_activations=True)
    save_generations(gens, str(tmp_path / "g.jsonl"), str(tmp_path / "g.npz"))
    loaded = np.load(str(tmp_path / "g.npz"))
    assert len(loaded.files) == 6, "a condition's activations were overwritten"


# --- confound reporting ---------------------------------------------------

def _fake_records(n_problems=40, condition_drives_label=True, seed=0):
    """
    Build VerificationRecords by hand with controllable structure.
    When condition_drives_label is True, label == condition, and the activation
    encodes condition. That is the exact artefact the report must catch.
    """
    from coding_eval.schemas import VerificationRecord

    rng = np.random.RandomState(seed)
    recs = []
    for i in range(n_problems):
        for cond in ("please_hack", "dont_hack"):
            label = (1 if cond == "please_hack" else 0) if condition_drives_label else int(i % 2)
            signal = 1 if cond == "please_hack" else -1
            vec = rng.randn(8) * 0.1
            vec[0] += 3.0 * signal          # activation encodes CONDITION, not label
            g = Generation(
                problem=Problem(problem_id=f"mbpp/{i}", prompt=f"p {i}",
                                style="function_call", test_code="def test_x():\n    assert True"),
                sample_index=0, prompt_text="p", response_text="r",
                prompt_token_len=2, response_token_len=3,
                condition=cond, model_id="m",
                activations=Activations(vectors={0: vec}, pooling="last",
                                        prompt_len=2, total_len=5, pooled_span=(4, 5)),
                activation_status="ok",
            )
            recs.append(VerificationRecord(generation=g, grade=GradeResult(label=label)))
    return recs


def test_report_flags_condition_confound():
    from coding_eval.probing import probe_report
    rep = probe_report(_fake_records(condition_drives_label=True), layer=0, n_splits=3)
    assert rep["pooled"]["auc"] > 0.9, "setup failed: pooled should look great"
    # within each condition there is only one class, so no real signal exists
    withins = [v["auc"] for v in rep["within"]["condition"].values()]
    assert all(a is None for a in withins)
    joined = " ".join(rep["warnings"])
    assert "CONFOUND [condition]" in joined
    assert "predicts condition itself" in joined


def test_report_clean_when_signal_is_real():
    from coding_eval.probing import probe_report
    from coding_eval.schemas import VerificationRecord

    rng = np.random.RandomState(1)
    recs = []
    for i in range(60):
        for cond in ("please_hack", "dont_hack"):
            label = int(i % 2)
            vec = rng.randn(8) * 0.1
            vec[0] += 3.0 * (1 if label else -1)   # activation encodes LABEL
            g = Generation(
                problem=Problem(problem_id=f"mbpp/{i}", prompt=f"p {i}",
                                style="function_call", test_code="def test_x():\n    assert True"),
                sample_index=0, prompt_text="p", response_text="r",
                prompt_token_len=2, response_token_len=3,
                condition=cond, model_id="m",
                activations=Activations(vectors={0: vec}, pooling="last",
                                        prompt_len=2, total_len=5, pooled_span=(4, 5)),
                activation_status="ok",
            )
            recs.append(VerificationRecord(generation=g, grade=GradeResult(label=label)))

    rep = probe_report(recs, layer=0, n_splits=3)
    assert rep["pooled"]["auc"] > 0.9
    withins = [v["auc"] for v in rep["within"]["condition"].values() if v["auc"] is not None]
    assert len(withins) == 2 and min(withins) > 0.9, "real signal should survive stratification"
    assert not any("CONFOUND [condition]" in w for w in rep["warnings"])


def test_report_flags_untested_confound_when_single_condition():
    from coding_eval.probing import probe_report
    recs = _fake_records(condition_drives_label=False)
    for r in recs:
        r.generation.condition = ""
    rep = probe_report(recs, layer=0, n_splits=3)
    assert any("UNCHECKED [condition]" in w for w in rep["warnings"])
