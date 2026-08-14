"""
Model backends. The generation pipeline talks to this interface and nothing else,
so swapping clean model -> RH model -> some future checkpoint is a constructor
argument. There is no model-specific branching anywhere above this file.

IMPORTANT ARCHITECTURAL CONSTRAINT, read before wiring up EC2:

    vLLM's OpenAI-compatible server does NOT expose hidden states, and it has no
    supported way to inject a forward hook into a served model. So:

        - plain generation           -> VLLMServerBackend (fast, batched, use this)
        - generation + activations   -> HFLocalBackend (needs the weights in-process)
        - generation + steering      -> HFLocalBackend (needs the weights in-process)

    You cannot get an activation vector or a steering hook out of an HTTP endpoint.
    Anyone who tells you otherwise is describing a fork of vLLM. The backends
    declare this via `supports_activations` / `supports_steering`, and generate()
    raises immediately rather than silently returning None vectors.

    Practical EC2 layout: run vLLM for the bulk plain-eval sweeps, and run the
    HF backend on the same box (or a second one) for the probe and steering arms.
    Same Problem objects, same prompts, same grader on both paths.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np


@dataclass
class GenParams:
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    stop: Sequence[str] = field(default_factory=tuple)
    seed: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": list(self.stop),
            "seed": self.seed,
        }


class Backend:
    """
    Minimal interface. Two required capabilities flags and two methods.

    generate_texts(prompts, params) -> list[str]
        Returns response text ONLY (prompt stripped), for each prompt.

    Backends that support activations additionally implement:
        forward_hidden_states(input_ids, layers) -> {layer: (seq_len, hidden) array}
    """

    model_id: str = ""
    supports_activations: bool = False
    supports_steering: bool = False

    def generate_texts(self, prompts: Sequence[str], params: GenParams) -> List[str]:
        raise NotImplementedError

    def forward_hidden_states(self, input_ids, layers: Sequence[int]) -> Dict[int, np.ndarray]:
        raise NotImplementedError(f"{type(self).__name__} cannot return hidden states")

    @contextlib.contextmanager
    def steering(self, layer: int, direction, alpha: float, prompt_len: int = 0, positions: str = "response"):
        raise NotImplementedError(f"{type(self).__name__} cannot be steered")


# --------------------------------------------------------------------------
# vLLM served endpoint (generation only)
# --------------------------------------------------------------------------

class VLLMServerBackend(Backend):
    """
    Talks to `vllm serve <model>` over the OpenAI-compatible /v1/completions route.

    We hit /v1/completions (raw prompt) rather than /v1/chat/completions on
    purpose: the prompt string is built once by our own formatter so the exact
    same token sequence is used for generation, for grading, and for the
    activation forward pass on the HF backend. Letting the server apply its own
    chat template would mean the probe arm and the eval arm see different
    prompts.
    """

    supports_activations = False
    supports_steering = False

    def __init__(self, base_url: str, model_id: str, timeout: float = 600.0, max_concurrency: int = 16):
        import requests  # local import so the module imports without network deps

        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self._session = requests.Session()

    def _one(self, prompt: str, params: GenParams) -> str:
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
        }
        if params.stop:
            payload["stop"] = list(params.stop)
        if params.seed is not None:
            payload["seed"] = params.seed
        r = self._session.post(f"{self.base_url}/v1/completions", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["text"]

    def generate_texts(self, prompts: Sequence[str], params: GenParams) -> List[str]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            return list(pool.map(lambda p: self._one(p, params), prompts))


# --------------------------------------------------------------------------
# Local HuggingFace model (generation + activations + steering)
# --------------------------------------------------------------------------

class HFLocalBackend(Backend):
    """
    In-process transformers model. Slower than vLLM but it is the only way to get
    hidden states or a steering hook.

    Note on response extraction: we slice the generated token IDs
    (`out[0][input_len:]`) and decode those. We do NOT decode the whole sequence
    and then chop `len(prompt_text)` characters off the front. String slicing is
    what the chat-eval notebook did, and with `skip_special_tokens=True` the
    decoded text is shorter than the prompt string you built (the `<|system|>` /
    `<|user|>` / `<|assistant|>` markers vanish), so the slice eats the first N
    characters of the actual response. Token slicing is exact.
    """

    supports_activations = True
    supports_steering = True

    def __init__(self, model, tokenizer, model_id: str = "", device: str = "cuda",
                 layer_attr: Optional[str] = "model.layers"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id or getattr(getattr(model, "config", None), "_name_or_path", "unknown")
        self.device = device
        self.is_merged_adapter = False
        self.base_model_id = None
        self.adapter_path = None
        # layer_attr=None means "work it out and verify it".
        self._layer_attr = layer_attr if layer_attr is not None else self._resolve_layer_attr()

    # Candidate layer paths, tried in order by _resolve_layer_attr(). The first
    # is standard transformers decoder layout (OLMo, Llama, Mistral). The rest
    # cover a PeftModel that was NOT merged, where the base model sits behind
    # one or two wrapper attributes.
    LAYER_ATTR_CANDIDATES = (
        "model.layers",                        # merged / plain HF causal LM
        "base_model.model.model.layers",       # PeftModel -> LoraModel -> CausalLM -> Model
        "base_model.model.layers",
        "transformer.h",                       # GPT-2 style, just in case
    )

    @classmethod
    def is_adapter(cls, path: str) -> bool:
        """
        True if `path` is a PEFT/LoRA adapter rather than a full model.

        Detected by the presence of adapter_config.json, locally or on the Hub.
        """
        import os

        if os.path.isdir(path):
            return os.path.exists(os.path.join(path, "adapter_config.json"))
        try:
            from huggingface_hub import file_exists
            return file_exists(path, "adapter_config.json")
        except Exception:
            # Offline or hub error: fall back to asking peft, which raises if
            # there is no adapter config.
            try:
                from peft import PeftConfig
                PeftConfig.from_pretrained(path)
                return True
            except Exception:
                return False

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        *,
        base_model: Optional[str] = None,
        merge_adapter: bool = True,
        tokenizer_path: Optional[str] = None,
        **kw,
    ):
        """
        Load a full model OR a PEFT/LoRA adapter on top of its base.

        The RH checkpoint (somo-olmo-7b-nohints-s1-chkpt-1520) is a LoRA adapter
        whose base is the clean model (somo-olmo-7b-sdf-sft). The clean model is
        a full model. Both go through this one call; the adapter case is
        detected from adapter_config.json, not from anything hardcoded here, so
        there is still no model-specific branching in the pipeline.

        base_model
            Override the adapter's recorded base_model_name_or_path. Use it when
            you have a local snapshot of the base and do not want a re-download.
        merge_adapter
            merge_and_unload() the adapter into the base weights (default).
            Merging matters for this project, not just for speed: it returns a
            plain transformers model, so the residual-stream steering hook
            attaches to a real decoder layer instead of a LoRA wrapper whose
            forward output is not the residual stream you think it is.
        tokenizer_path
            Adapter repos often ship no tokenizer; we fall back to the base.

        model_id is set to the ADAPTER path, not the base. If both models
        reported the same id, probe_report()'s within-model_id confound check
        would silently collapse to a single stratum and report UNCHECKED.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, dtype)
        adapter = cls.is_adapter(path)
        base_id = None

        if not adapter:
            model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch_dtype, device_map=device, **kw)
            tok_src = tokenizer_path or path
        else:
            try:
                from peft import PeftConfig, PeftModel
            except ImportError:
                raise ImportError(
                    f"{path} is a PEFT adapter but `peft` is not installed. "
                    "pip install peft"
                ) from None

            cfg = PeftConfig.from_pretrained(path)
            base_id = base_model or cfg.base_model_name_or_path
            if not base_id:
                raise ValueError(
                    f"{path} is an adapter but records no base_model_name_or_path; "
                    "pass base_model=... explicitly."
                )
            base = AutoModelForCausalLM.from_pretrained(
                base_id, torch_dtype=torch_dtype, device_map=device, **kw)
            peft_model = PeftModel.from_pretrained(base, path, torch_dtype=torch_dtype)
            if merge_adapter:
                # merge_and_unload() folds BA*scaling into W and returns the
                # UNWRAPPED base model, so the layer path is the same as for a
                # plain model. _resolve_layer_attr() verifies that rather than
                # trusting it.
                model = peft_model.merge_and_unload()
            else:
                model = peft_model
            tok_src = tokenizer_path or path
            try:
                AutoTokenizer.from_pretrained(tok_src)
            except Exception:
                tok_src = base_id      # adapter repo ships no tokenizer

        tok = AutoTokenizer.from_pretrained(tok_src)
        model.eval()

        backend = cls(model, tok, model_id=path, device=device,
                      layer_attr=None)          # resolved below
        backend.is_merged_adapter = bool(adapter and merge_adapter)
        backend.base_model_id = base_id
        backend.adapter_path = path if adapter else None
        return backend

    # ---- layer access -----------------------------------------------------

    def _walk(self, path: str):
        obj = self.model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    def _resolve_layer_attr(self) -> str:
        """
        Find the decoder-layer ModuleList and CHECK it, rather than assuming.

        merge_and_unload() is documented to return the unwrapped base model, so
        a merged LoRA should expose model.model.layers exactly like a plain
        checkpoint. "Should" is not good enough here: if the path silently
        resolved to a LoRA wrapper instead, the steering hook would attach to
        something whose forward output is not the residual stream, and every
        steering number would be quietly meaningless while still producing
        plausible text. So each candidate is validated against
        config.num_hidden_layers before it is accepted.
        """
        n_expected = getattr(getattr(self.model, "config", None), "num_hidden_layers", None)
        tried = []
        for cand in self.LAYER_ATTR_CANDIDATES:
            obj = self._walk(cand)
            if obj is None:
                tried.append(f"{cand}: not present")
                continue
            try:
                n = len(obj)
            except TypeError:
                tried.append(f"{cand}: not a sequence ({type(obj).__name__})")
                continue
            if n == 0:
                tried.append(f"{cand}: empty")
                continue
            if n_expected is not None and n != n_expected:
                tried.append(f"{cand}: {n} modules but config says {n_expected}")
                continue
            return cand

        raise RuntimeError(
            "could not locate the decoder layers on this model.\n  tried:\n    "
            + "\n    ".join(tried)
            + "\n  Pass layer_attr='...' explicitly to HFLocalBackend(). Do NOT guess: "
              "an incorrect layer path makes activation extraction and steering "
              "silently meaningless."
        )

    def describe_layers(self) -> dict:
        """
        Diagnostics for bring-up. Print this once per model before a sweep.

        Confirms the resolved path, that a LoRA merge actually happened (no
        lora_ modules left anywhere), and what module type the hook will attach
        to.
        """
        layers = self.layers
        leftover = [n for n, _ in self.model.named_modules() if "lora" in n.lower()]
        return {
            "model_id": self.model_id,
            "base_model_id": self.base_model_id,
            "is_merged_adapter": self.is_merged_adapter,
            "layer_attr": self._layer_attr,
            "n_layers": len(layers),
            "config_num_hidden_layers": getattr(
                getattr(self.model, "config", None), "num_hidden_layers", None),
            "layer_module_type": type(layers[0]).__name__,
            "model_class": type(self.model).__name__,
            "residual_lora_modules": len(leftover),
            "lora_examples": leftover[:3],
        }

    def assert_ready_for_steering(self) -> dict:
        """
        Hard preconditions for the steering arm. Call before a sweep.

        An UNMERGED PeftModel is the dangerous case: hooks would fire on wrapped
        modules and the numbers would look fine but mean nothing.
        """
        d = self.describe_layers()
        if d["residual_lora_modules"]:
            raise RuntimeError(
                f"{d['residual_lora_modules']} LoRA modules are still present "
                f"(e.g. {d['lora_examples']}). Load with merge_adapter=True; steering "
                "hooks on a wrapped model do not act on the residual stream."
            )
        if d["config_num_hidden_layers"] not in (None, d["n_layers"]):
            raise RuntimeError(
                f"resolved {d['n_layers']} layers at {d['layer_attr']!r} but config says "
                f"{d['config_num_hidden_layers']}"
            )
        return d

    @property
    def layers(self):
        obj = self._walk(self._layer_attr)
        if obj is None:
            raise AttributeError(
                f"layer_attr {self._layer_attr!r} does not resolve on "
                f"{type(self.model).__name__}"
            )
        return obj

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    # ---- generation -------------------------------------------------------

    def generate_texts(self, prompts: Sequence[str], params: GenParams) -> List[str]:
        import torch

        # `pad_token_id or eos_token_id` is WRONG: a legitimate pad_token_id of 0
        # is falsy, so it silently falls through to EOS. Padding with EOS makes
        # the model's own EOS indistinguishable from padding, which corrupts the
        # response-token span the whole activation pipeline depends on.
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError(
                "tokenizer has neither pad_token_id nor eos_token_id; set one before generating"
            )

        outs: List[str] = []
        for prompt in prompts:
            # GenParams.seed was previously accepted and then ignored here, so
            # "reproducible" runs were not. Seed per prompt, not once per batch,
            # so a resumed or reordered run reproduces the same completions.
            if params.seed is not None:
                torch.manual_seed(params.seed)

            enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_len = enc.input_ids.shape[1]
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=params.max_tokens,
                    do_sample=params.temperature > 0,
                    temperature=params.temperature if params.temperature > 0 else None,
                    top_p=params.top_p,
                    pad_token_id=pad_id,
                )
            # Token-level slice, not string-level. See class docstring.
            new_tokens = out[0][input_len:]
            outs.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return outs

    # ---- hidden states ----------------------------------------------------

    def forward_hidden_states(self, input_ids, layers: Sequence[int]) -> Dict[int, np.ndarray]:
        """
        ONE forward pass over the full sequence handed in. Returns the full
        per-token hidden states for the requested layers, shape (seq_len, hidden).

        This function deliberately knows nothing about prompts or responses. It
        gets whatever token sequence the caller built and returns everything;
        span selection is done by the caller (generation.py), where the span is
        validated. Splitting it this way means the pooling logic is in one place
        and testable without a GPU.
        """
        import torch

        captured: Dict[int, "torch.Tensor"] = {}
        wanted = set(layers)

        def make_hook(idx: int):
            def hook(_module, _inp, output):
                if idx in wanted:
                    hidden = output[0] if isinstance(output, tuple) else output
                    captured[idx] = hidden.detach().float().cpu()
            return hook

        handles = []
        try:
            for i, layer in enumerate(self.layers):
                if i in wanted:
                    handles.append(layer.register_forward_hook(make_hook(i)))
            if isinstance(input_ids, (list, tuple)):
                input_ids = torch.tensor([list(input_ids)], dtype=torch.long)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.to(self.device)
            with torch.no_grad():
                self.model(input_ids=input_ids)
        finally:
            for h in handles:
                h.remove()

        missing = wanted - set(captured)
        if missing:
            raise RuntimeError(f"no activations captured for layers {sorted(missing)}")
        return {i: captured[i][0].numpy() for i in sorted(captured)}

    # ---- steering ---------------------------------------------------------

    @contextlib.contextmanager
    def steering(self, layer: int, direction, alpha: float, prompt_len: int = 0, positions: str = "response"):
        """
        Add `alpha * direction` to the residual stream at `layer` for the duration
        of the context.

        positions:
            "response" (default) - during prefill, only positions >= prompt_len
                                   are modified; every incremental decode step is
                                   modified. This steers what the model writes
                                   without perturbing its reading of the problem.
            "all"                - every position, prompt included.

        Works during .generate() because the hook fires on every forward, and
        with a KV cache the decode-step hidden is shape (B, 1, D), which we treat
        as a response position.
        """
        import torch

        if not (0 <= layer < self.n_layers):
            raise ValueError(f"steering_layer {layer} out of range 0..{self.n_layers - 1}")

        vec = torch.as_tensor(np.asarray(direction), dtype=torch.float32)
        if vec.numel() != self.hidden_size:
            raise ValueError(f"direction has {vec.numel()} dims, model hidden size is {self.hidden_size}")
        vec = vec.to(self.device)

        def hook(_module, _inp, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            delta = (alpha * vec).to(hidden.dtype)
            if positions == "all" or hidden.shape[1] == 1:
                hidden = hidden + delta
            else:
                # prefill: only touch tokens at/after the prompt boundary
                if hidden.shape[1] > prompt_len:
                    hidden = hidden.clone()
                    hidden[:, prompt_len:, :] = hidden[:, prompt_len:, :] + delta
            return (hidden,) + output[1:] if is_tuple else hidden

        handle = self.layers[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()
