"""
THE generation pipeline. One function: generate().

All three current use cases are the same call with different optional arguments:

    plain eval        generate(model=m, problems=P)
    probe data        generate(model=m, problems=P, extract_activations=True)
    steering sweep    generate(model=m, problems=P, steering_layer=L,
                               steering_direction=d, steering_alpha=a)

and any combination (steer + record what the steered model's activations look
like) works too. There is no separate probe pipeline and no separate steering
pipeline, because those arms differ only in what you ask this function to
additionally return or additionally hook.

No model-specific logic lives here. The backend is a parameter.

--------------------------------------------------------------------------
ACTIVATION EXTRACTION CONTRACT (the second bug, made structural)
--------------------------------------------------------------------------
When extract_activations=True:

  1. The model generates the full solution first.
  2. We build ONE token sequence = prompt_ids + response_ids.
  3. ONE forward pass runs over that whole sequence.
  4. Pooling happens over [prompt_len, total_len) ONLY, i.e. the solution's own
     tokens. The prompt span is never pooled.
  5. If the response is empty, we record activation_status="empty_response" and
     activations=None. We do NOT emit a zero vector.

Point 5 matters: the chat-eval extractor appended `torch.zeros(...)` for empty
response spans and carried on. Those all-zero rows kept their label and their
group and went straight into the probe as training data. A zero row is not a
neutral row; with a linear model it is a free bias-only example for whatever
class it happens to be labelled. Here a missing activation is a missing record.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .backends import Backend, GenParams
from .schemas import Activations, Generation, Problem

log = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Solve the given programming problem. "
    "Return your complete solution in a single Python code block."
)


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

def format_prompt(tokenizer, system_prompt: str, problem: Problem, template: Optional[str] = None) -> str:
    """
    Build the prompt string. Uses the tokenizer's chat template when it has one,
    otherwise the OLMo-style manual format used in the chat-eval notebook.

    Exposed rather than inlined so the exact same string is used by the vLLM
    path, the HF path, and the activation forward pass. If these ever diverge,
    your probe is trained on a distribution your eval never produced.
    """
    if template is not None:
        return template.format(system_prompt=system_prompt, problem=problem.prompt)

    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": problem.prompt})
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    return f"<|system|>\n{system_prompt}\n<|user|>\n{problem.prompt}\n<|assistant|>\n"


def _token_spans(tokenizer, prompt_text: str, response_text: str):
    """
    Return (prompt_ids, response_ids) such that prompt_ids + response_ids is the
    sequence we run the forward pass on, and len(prompt_ids) is the EXACT
    boundary index.

    We tokenize the two pieces separately and concatenate IDs rather than
    tokenizing the concatenated string and guessing where the prompt ended. The
    chat-eval extractor computed prompt_len from a separate tokenization of the
    prompt but ran the forward pass on the joined *string*; a tokenizer that
    merges across the seam (very common: the first response token gets glued to
    the last prompt token) makes those two lengths disagree by a token or three.
    With last-token pooling you get away with it. With mean pooling over the
    response span you silently average in prompt tokens. Concatenating IDs makes
    the boundary exact by construction.

    Cost: the seam tokenization can differ by one token from what the model
    actually generated. That is a strictly smaller error than an unknown-offset
    span, and it is deterministic.
    """
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True).input_ids
    response_ids = tokenizer(response_text, add_special_tokens=False).input_ids
    return prompt_ids, response_ids


def pool_response_span(
    hidden_by_layer: Dict[int, np.ndarray],
    prompt_len: int,
    total_len: int,
    pooling: str,
) -> "tuple[Dict[int, np.ndarray], tuple]":
    """
    Pool per-layer (seq_len, hidden) states over the RESPONSE span only.
    Returns ({layer: vector}, (span_start, span_end)).

    pooling:
        "last" - final response token. Matches the chat-eval setup.
        "mean" - mean over all response tokens.

    Recommendation for coding evals specifically: run both. "last" was chosen for
    chat answers of a few dozen tokens. The last token of a 400-token program is
    usually a newline or a closing paren after a long tail of boilerplate, and it
    carries much less of the response than it did in chat. Cheap to compute both
    from the same forward pass; just call this twice.
    """
    if total_len <= prompt_len:
        raise ValueError("empty response span")

    if pooling == "last":
        start, end = total_len - 1, total_len
    elif pooling == "mean":
        start, end = prompt_len, total_len
    else:
        raise ValueError(f"unknown pooling {pooling!r}, use 'last' or 'mean'")

    # Belt and braces: this is the invariant the whole bug was about.
    assert start >= prompt_len, f"pooling span {start}:{end} would include prompt tokens"

    out = {}
    for layer, states in hidden_by_layer.items():
        if states.shape[0] != total_len:
            raise RuntimeError(
                f"layer {layer}: forward pass returned {states.shape[0]} positions "
                f"but the concatenated sequence is {total_len} tokens. Span alignment "
                "cannot be trusted; refusing to pool."
            )
        seg = states[start:end]
        out[layer] = seg[-1] if pooling == "last" else seg.mean(axis=0)
    return out, (start, end)


# --------------------------------------------------------------------------
# The one entry point
# --------------------------------------------------------------------------

def generate(
    *,
    model: Backend,
    problems: Sequence[Problem],
    tokenizer=None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    condition: str = "",
    gen_params: Optional[GenParams] = None,
    prompt_template: Optional[str] = None,
    n_samples_per_problem: int = 1,
    # --- optional arm: activation extraction (probe) ---
    extract_activations: bool = False,
    activation_layers: Optional[Sequence[int]] = None,
    pooling: str = "last",
    activations_under_steering: bool = True,
    # --- optional arm: causal steering ---
    steering_layer: Optional[int] = None,
    steering_direction: Optional[Any] = None,
    steering_alpha: Optional[float] = None,
    steering_positions: str = "response",
    on_error: str = "record",   # "record" | "raise"
) -> List[Generation]:
    """
    Generate solutions for `problems` with `model`.

    Parameters
    ----------
    model
        Any Backend. VLLMServerBackend for plain sweeps, HFLocalBackend when you
        need activations or steering.
    problems
        Problem objects. problem_id travels into every Generation returned.
    system_prompt
        Raw prompt text. There is no condition lookup table; if you are running
        the AISI system-prompt conditions, pass the verbatim text here.
    condition
        Short label for the system-prompt condition ("please_hack", "dont_hack",
        "no_hints", ...). Purely bookkeeping, it never changes generation. Set it
        whenever you run more than one condition over the same problems: it is
        part of sample_uid, and without it multi-condition runs overwrite each
        other's activations in the npz.
    tokenizer
        Required for prompt formatting and for activation spans. HFLocalBackend
        carries its own; pass one explicitly when using vLLM.
    n_samples_per_problem
        k completions per problem. They share a group_key, so splits.py keeps
        them on one side of any split automatically.
    extract_activations
        Adds a Generation.activations vector pooled from the response span.
        Requires model.supports_activations.
    activation_layers
        Which layers to capture. Defaults to all of them (a 7B model with 32
        layers x 4096 dims is ~0.5 MB per sample in fp32, fine for a layer sweep;
        narrow it once you know your layer).
    activations_under_steering
        When steering is also on, run the activation forward pass with the same
        hook active, so the recorded activations match the state the model was
        actually in while generating. Set False to record the unsteered
        representation of the steered text.
    steering_layer / steering_direction / steering_alpha
        All three must be given together. Requires model.supports_steering.

    Returns
    -------
    list[Generation], one per (problem, sample_index).
    """
    gen_params = gen_params or GenParams()
    tokenizer = tokenizer or getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("a tokenizer is required (pass tokenizer=..., or use a backend that carries one)")

    # ---- validate optional arms against backend capability, loudly ----------
    steering_on = any(x is not None for x in (steering_layer, steering_direction, steering_alpha))
    if steering_on:
        if steering_layer is None or steering_direction is None or steering_alpha is None:
            raise ValueError("steering needs all three of steering_layer, steering_direction, steering_alpha")
        if not model.supports_steering:
            raise ValueError(
                f"{type(model).__name__} cannot be steered (a served vLLM endpoint has no hook "
                "injection point). Use HFLocalBackend for the steering arm."
            )
    if extract_activations and not model.supports_activations:
        raise ValueError(
            f"{type(model).__name__} cannot return hidden states (an HTTP endpoint does not expose "
            "them). Use HFLocalBackend for the probe arm."
        )

    if extract_activations and activation_layers is None:
        activation_layers = list(range(model.n_layers))

    steering_meta = (
        {"layer": steering_layer, "alpha": steering_alpha, "positions": steering_positions,
         "direction_norm": float(np.linalg.norm(np.asarray(steering_direction)))}
        if steering_on else None
    )

    # ---- build the work list; problem_id is attached from here on ----------
    units = [(p, k) for p in problems for k in range(n_samples_per_problem)]
    prompts = [format_prompt(tokenizer, system_prompt, p, prompt_template) for p, _ in units]

    # ---- generation --------------------------------------------------------
    if steering_on:
        # One at a time: the "response" position policy needs this item's prompt_len.
        texts: List[str] = []
        for prompt in prompts:
            plen = len(tokenizer(prompt, add_special_tokens=True).input_ids)
            with model.steering(steering_layer, steering_direction, steering_alpha,
                                prompt_len=plen, positions=steering_positions):
                texts.append(model.generate_texts([prompt], gen_params)[0])
    else:
        texts = model.generate_texts(prompts, gen_params)

    # A backend that returns fewer texts than prompts (partial batch failure,
    # a truncated server response, a custom backend that drops items) would be
    # silently absorbed by zip() below: you would get fewer Generations than
    # problems, with no error and no warning, and the missing problems would
    # simply not appear in the run.
    if len(texts) != len(prompts):
        raise RuntimeError(
            f"backend returned {len(texts)} completions for {len(prompts)} prompts. "
            "Refusing to continue: zip() would silently drop the difference and the "
            "missing problems would vanish from the run without an error."
        )

    # ---- assemble records, optionally with activations ---------------------
    results: List[Generation] = []
    for (problem, sample_index), prompt_text, response_text in zip(units, prompts, texts):
        prompt_ids, response_ids = _token_spans(tokenizer, prompt_text, response_text)
        prompt_len, resp_len = len(prompt_ids), len(response_ids)

        rec = Generation(
            problem=problem,
            sample_index=sample_index,
            prompt_text=prompt_text,
            response_text=response_text,
            prompt_token_len=prompt_len,
            response_token_len=resp_len,
            model_id=model.model_id,
            gen_params=gen_params.as_dict(),
            steering=steering_meta,
            condition=condition,
            system_prompt=system_prompt,
        )

        if extract_activations:
            if resp_len == 0:
                # Missing, not zero. See module docstring.
                rec.activation_status = "empty_response"
                log.warning("%s sample %d produced an empty response; no activations recorded",
                            problem.problem_id, sample_index)
            else:
                try:
                    full_ids = list(prompt_ids) + list(response_ids)
                    ctx = (
                        model.steering(steering_layer, steering_direction, steering_alpha,
                                       prompt_len=prompt_len, positions=steering_positions)
                        if (steering_on and activations_under_steering)
                        else _null_context()
                    )
                    with ctx:
                        # ONE forward pass over prompt + full generated response.
                        hidden = model.forward_hidden_states(full_ids, activation_layers)
                    vectors, span = pool_response_span(hidden, prompt_len, len(full_ids), pooling)
                    rec.activations = Activations(
                        vectors=vectors,
                        pooling=pooling,
                        prompt_len=prompt_len,
                        total_len=len(full_ids),
                        pooled_span=span,
                        under_steering=bool(steering_on and activations_under_steering),
                    )
                    rec.activation_status = "ok"
                except Exception as exc:  # noqa: BLE001
                    if on_error == "raise":
                        raise
                    rec.activation_status = f"error:{exc}"
                    log.exception("activation extraction failed for %s", problem.problem_id)

        results.append(rec)

    return results


import contextlib


@contextlib.contextmanager
def _null_context():
    yield


# --------------------------------------------------------------------------
# Activations for text that already exists
# --------------------------------------------------------------------------

def add_activations(
    items: Sequence,
    model: Backend,
    *,
    layers: Optional[Sequence[int]] = None,
    pooling: str = "last",
    tokenizer=None,
    on_error: str = "record",
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 1,
    progress_every: int = 25,
) -> List:
    """
    Pool activations for ALREADY-GENERATED responses. No regeneration.

    Why this exists. The hack-rate sweep runs on vLLM, which is batched and
    fast but cannot expose hidden states. Re-running the probe subset through
    HFLocalBackend with extract_activations=True would regenerate the text at
    temperature 0.7, giving DIFFERENT responses with DIFFERENT labels and
    wasting the grading already paid for. This runs one forward pass over the
    prompt and response you already have, so the labels stay valid and the cost
    is a forward pass rather than generation plus a forward pass.

    MEMORY. Uses model.forward_pooled() when available, which pools on the GPU
    and returns only len(layers) x hidden floats per sample. The fallback,
    forward_hidden_states, copies the full (seq_len, hidden) tensor to CPU per
    layer: 0.33 GB for 5 layers at 4000 tokens, 4.2 GB for 32 layers at 8000.
    Repeating that per sample on a 32 GB box holding a merged 7B drives the
    machine into swap and freezes it, losing SSH, with no traceback.

    checkpoint_dir
        One .npy per sample_uid, written as it is produced. Samples already on
        disk are skipped, so an interrupted extraction resumes instead of
        starting over. Per-file writes avoid the O(n^2) cost of rewriting one
        growing archive.

    Accepts Generations or VerificationRecords; returns the same objects with
    .activations populated in place.
    """
    import time as _time

    from .schemas import Activations, VerificationRecord as _VR

    gens = [i.generation if isinstance(i, _VR) else i for i in items]
    if not gens:
        return list(items)

    if not model.supports_activations:
        raise ValueError(
            f"{type(model).__name__} cannot return hidden states. Use HFLocalBackend; "
            "a served vLLM endpoint has no way to expose them."
        )

    tokenizer = tokenizer or getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("a tokenizer is required")

    # The forward pass must use the model that WROTE the text. Pooling RH
    # responses through the clean model measures the clean model reading RH
    # text, which is a different experiment and would not look wrong in any plot.
    seen_ids = {g.model_id for g in gens if g.model_id}
    if seen_ids and model.model_id and model.model_id not in seen_ids:
        raise ValueError(
            f"these responses were generated by {sorted(seen_ids)} but the backend is "
            f"{model.model_id!r}. Pooling one model's text through another model's "
            "weights is a different experiment; load the matching checkpoint."
        )

    if layers is None:
        layers = list(range(model.n_layers))
    layers = list(layers)

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _ckpt_path(uid: str) -> str:
        safe = uid.replace("/", "__").replace("::", "--")
        return os.path.join(checkpoint_dir, f"{safe}.npy")

    def _attach(g, stacked):
        plen = g.prompt_token_len
        total = plen + g.response_token_len
        start = total - 1 if pooling == "last" else plen
        g.activations = Activations(
            vectors={l: stacked[i] for i, l in enumerate(layers)},
            pooling=pooling, prompt_len=plen, total_len=total,
            pooled_span=(start, total), under_steering=False,
        )
        g.activation_status = "ok"

    # hasattr is always True: the base Backend declares forward_pooled and
    # raises. Check for an actual override, and still fall back at runtime if
    # the override itself is not implemented.
    use_pooled = type(model).forward_pooled is not Backend.forward_pooled
    t0 = _time.time()
    done = skipped = failed = 0

    for i, g in enumerate(gens):
        # ---- resume from checkpoint ---------------------------------------
        if checkpoint_dir:
            path = _ckpt_path(g.sample_uid)
            if os.path.exists(path):
                try:
                    stacked = np.load(path)
                    prompt_ids, response_ids = _token_spans(
                        tokenizer, g.prompt_text, g.response_text)
                    g.prompt_token_len = len(prompt_ids)
                    g.response_token_len = len(response_ids)
                    if len(response_ids):
                        _attach(g, stacked)
                        skipped += 1
                        continue
                except Exception:                      # noqa: BLE001
                    log.warning("checkpoint for %s unreadable; recomputing",
                                g.sample_uid)

        prompt_ids, response_ids = _token_spans(tokenizer, g.prompt_text, g.response_text)
        plen, rlen = len(prompt_ids), len(response_ids)
        g.prompt_token_len, g.response_token_len = plen, rlen

        if rlen == 0:
            g.activation_status = "empty_response"
            g.activations = None
            continue

        try:
            full = list(prompt_ids) + list(response_ids)
            total = len(full)
            start = total - 1 if pooling == "last" else plen
            if start < plen:
                raise RuntimeError("pooling span would include prompt tokens")

            if use_pooled:
                try:
                    vectors = model.forward_pooled(full, layers, (start, total), pooling)
                except NotImplementedError:
                    use_pooled = False
                    hidden = model.forward_hidden_states(full, layers)
                    vectors, _ = pool_response_span(hidden, plen, total, pooling)
                    del hidden
            else:
                hidden = model.forward_hidden_states(full, layers)
                vectors, _ = pool_response_span(hidden, plen, total, pooling)
                del hidden

            g.activations = Activations(
                vectors=vectors, pooling=pooling, prompt_len=plen,
                total_len=total, pooled_span=(start, total), under_steering=False,
            )
            g.activation_status = "ok"
            done += 1

            if checkpoint_dir and (done % max(1, checkpoint_every) == 0 or
                                   checkpoint_every == 1):
                np.save(_ckpt_path(g.sample_uid), g.activations.stack(layers))
        except Exception as exc:                       # noqa: BLE001
            if on_error == "raise":
                raise
            g.activation_status = f"error:{exc}"
            g.activations = None
            failed += 1
            log.exception("activation extraction failed for %s", g.sample_uid)

        if progress_every and (i + 1) % progress_every == 0:
            el = _time.time() - t0
            rate = (i + 1) / el if el else 0
            eta = (len(gens) - i - 1) / rate if rate else 0
            print(f"  activations {i+1}/{len(gens)}  "
                  f"{rate:.1f}/s  eta {eta/60:.0f} min  "
                  f"(ok={done} resumed={skipped} failed={failed})", flush=True)

    print(f"  activations complete: {done} computed, {skipped} resumed, "
          f"{failed} failed, {len(gens)} total", flush=True)
    return list(items)


# --------------------------------------------------------------------------
# Persistence helpers
# --------------------------------------------------------------------------

def save_generations(generations: Sequence[Generation], jsonl_path: str, npz_path: Optional[str] = None) -> None:
    """
    Write text/metadata to JSONL and activations to a single .npz keyed by
    sample_uid.

    Keyed by sample_uid, not by row index. That is the whole point: the chat-eval
    layout was four files aligned by position (responses / labels / ids / a
    stacked .pt tensor), and one skipped item shifted three of them relative to
    the fourth with no error. Here a record without activations simply has no key
    in the npz, and the loader knows.

    Raises on duplicate sample_uid rather than letting np.savez silently keep the
    last writer. The usual cause is running several system-prompt conditions
    through generate() without passing `condition=`, which makes every condition
    collide on `<problem_id>::<sample_index>`.
    """
    import json
    import os

    # Create parent dirs. On Kaggle a bare relative name lands in the
    # auto-persisted /kaggle/working; on EC2 it lands wherever you launched
    # python, and a path like "runs/day1/x.jsonl" raised FileNotFoundError
    # AFTER all the GPU work was done.
    for _p in (jsonl_path, npz_path):
        if _p and os.path.dirname(_p):
            os.makedirs(os.path.dirname(_p), exist_ok=True)

    seen: Dict[str, int] = {}
    for g in generations:
        seen[g.sample_uid] = seen.get(g.sample_uid, 0) + 1
    dupes = {uid: n for uid, n in seen.items() if n > 1}
    if dupes:
        example = sorted(dupes)[:3]
        raise ValueError(
            f"{len(dupes)} duplicate sample_uid(s) in this batch (e.g. {example}). "
            "Saving would silently overwrite activations. If you are running multiple "
            "system-prompt conditions, pass condition= to generate(), or write one "
            "file per condition."
        )

    with open(jsonl_path, "w") as f:
        for g in generations:
            row = {
                "problem_id": g.problem_id,
                "group_key": g.group_key,
                "sample_index": g.sample_index,
                "sample_uid": g.sample_uid,
                "dataset": g.problem.dataset,
                "style": g.problem.style,
                "model_id": g.model_id,
                "condition": g.condition,
                "system_prompt": g.system_prompt,
                "prompt_text": g.prompt_text,
                "response_text": g.response_text,
                "prompt_token_len": g.prompt_token_len,
                "response_token_len": g.response_token_len,
                "activation_status": g.activation_status,
                "steering": g.steering,
                "gen_params": g.gen_params,
                "activation_meta": (
                    {"pooling": g.activations.pooling, "pooled_span": list(g.activations.pooled_span),
                     "prompt_len": g.activations.prompt_len, "total_len": g.activations.total_len,
                     "under_steering": g.activations.under_steering}
                    if g.activations else None
                ),
            }
            f.write(json.dumps(row) + "\n")

    if npz_path:
        arrays = {}
        for g in generations:
            if g.activations is not None:
                arrays[g.sample_uid] = g.activations.stack()
        np.savez_compressed(npz_path, **arrays)
