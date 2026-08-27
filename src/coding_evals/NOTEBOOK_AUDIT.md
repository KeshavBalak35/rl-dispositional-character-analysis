# Audit of `character.ipynb`

What's being replaced and why. Cell numbers are 0-indexed positions in the notebook.

---

## Bug 2 (prompt-only forward pass): still live in the notebook

**Cells 20 and 30 are the original buggy extractors and they were never deleted.**

```python
# cell 20 (clean) / cell 30 (RH)
chat = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{question}\n<|assistant|>\n"
inputs = tokenizer(chat, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = clean_model(**inputs)      # <-- prompt only, no response
```

They write to `clean_activations.pt` and `rh_activations.pt`. The fixed versions
(cells 32/33) write to `clean_activations_last_token_full.pt` and
`rh_activations_last_token_full.pt`. Both pairs of files exist on disk with
similar names, and the only thing stopping someone loading the wrong one is
remembering which suffix means "correct".

Delete cells 20 and 30, and delete the two `*_activations.pt` files. A poisoned
artifact sitting next to a good one under a similar name is how this bug comes
back six weeks from now.

### Cells 32/33 (the fix) have two residual problems

**(a) Zero-vector rows enter the probe.**

```python
if seq_len <= prompt_len:
    print(f"Skipping question {i+1}, empty response span")
    results.append(torch.zeros(len(model.model.layers), model.config.hidden_size))
    continue
```

It prints "Skipping" but does not skip. It appends an all-zero row, which keeps
its position, so it keeps its label in `clean_labels.json` and its group in
`question_ids.json`, and it goes into `LogisticRegression.fit` as a real training
example. A zero row is not neutral for a linear model.

New pipeline: `activation_status="empty_response"`, `activations=None`, and
`probe_dataset()` drops it. Missing is missing.

**(b) The span boundary is computed from a different tokenization than the one
the forward pass runs on.**

```python
prompt_len = tokenizer(prompt_text, return_tensors="pt").input_ids.shape[1]
full_inputs = tokenizer(full_text, return_tensors="pt").to("cuda")   # joined STRING
```

`prompt_len` comes from tokenizing the prompt alone; the forward pass runs on the
tokenization of `prompt + response` as one string. Tokenizers merge across the
seam, so these two can disagree by a token or three. You get away with it because
you pool `seq_len - 1` (last token), which does not depend on `prompt_len` at
all. `prompt_len` is only used for the emptiness check.

That means the moment anyone tries mean-pooling over the response span, this
silently averages in prompt tokens. New pipeline tokenizes the two pieces
separately and concatenates token IDs, so the boundary is exact by construction.

---

## Bug 1 (leakage): fixed in the notebook, but held together by index alignment

Cells 10, 34 and 36 do the right thing: `question_ids` grouped by underlying
question, `StratifiedGroupKFold`, `GroupShuffleSplit`, plus a disjointness assert
in cell 36. That logic is correct.

The fragility is in how identity is stored.

**Question identity lives in a parallel list, not in the record.** Cell 10 writes
`question_ids.json` as a flat list whose meaning depends entirely on positional
alignment with four other artifacts: `clean_responses.json`, `clean_labels.json`,
`rh_labels.json`, and a stacked `.pt` tensor. Nothing enforces that alignment.
Cell 10's assert only checks lengths.

Concretely, cell 17:

```python
except Exception as e:
    print(f"Error on question {i+1}: {e}")
    ...
    raise
```

If any single generation had been allowed to continue past an error rather than
raise, `clean_responses` would be one item short and every subsequent index would
map to the wrong question ID, the wrong label, and the wrong group. The exception
path currently saves and re-raises, which is what saves you. It is one edit away
from being a silent, undetectable corruption of the entire grouping fix.

**Cell 10's construction assumes cell 7's ordering.** Cell 7 builds
`betley_questions` by iterating `BETLEY_MAIN_QUESTIONS + BETLEY_PREREGISTERED_QUESTIONS`
and repeating each 15x. Cell 10 independently rebuilds the same iteration to make
IDs. Two separate loops that must produce the same order. Reorder one and the
assert still passes (lengths match) while every question is now labelled with a
different question's ID.

**The holdout in cell 36 is unstratified.** `GroupShuffleSplit` with
`test_size=0.1`. If the misaligned class is rare, the holdout can contain zero
positives and the steering comparison has no baseline hack rate to move.

New pipeline: `problem_id` is a required field on `Problem` and travels inside
every `Generation` and `VerificationRecord`. There are no parallel lists.
`splits.py` refuses to split anything that lacks `.group_key`, so you cannot
accidentally hand it a bare feature matrix, and every split is leakage-asserted
before it is returned.

---

## Three other things worth fixing while you're in here

**1. Response extraction by string slicing (cell 17, 27) is probably corrupting
your stored responses.**

```python
full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response_only = full_response[len(chat):].strip()
```

`chat` contains `<|system|>`, `<|user|>`, `<|assistant|>`. If those are registered
special tokens in the OLMo SFT tokenizer (they are in the allenai OLMo instruct
line), `skip_special_tokens=True` strips them from the decode, so the decoded
string is *shorter* than `chat` by however many characters those markers occupy.
Slicing `[len(chat):]` then eats the first ~30 characters of the actual response.

This propagates: cells 32/33 rebuild `full_text = prompt_text + response` from
these stored strings, so a truncated response means the activation forward pass
runs on a sequence the model never produced.

Check it in one line:

```python
print(repr(tokenizer.decode(tokenizer(chat).input_ids, skip_special_tokens=True))[:200])
```

If the `<|...|>` markers are gone from that output, your responses are clipped.
Fix is token slicing: `tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:])`.
The new backend does this.

**2. The judge defaults ambiguous verdicts to "aligned" (cell 18/28).**

```python
else:
    return 0  # default to aligned
```

Every API hiccup, every unparseable verdict, becomes a clean negative. The new
`GradeResult.label` allows `None`, and undetermined items are dropped from probe
training rather than silently relabelled.

**3. `client = anthropic.Anthropic(api_key="API-KEY")` is a hardcoded literal in
two cells.** Fine on Kaggle with a placeholder; on a shared EC2 box in a git repo
it is how keys leak. Use `os.environ["ANTHROPIC_API_KEY"]`.

---

## Two confounds the notebook does not currently control for

Not bugs in the code, but they determine whether the coding-eval results mean
anything.

**Model identity.** If you pool clean and RH completions and train on
hack/no-hack, a probe can score well by detecting which model wrote the text.
Report within-model AUC, RH-only (hack vs no-hack), as the headline number.

**Response length.** Reward hacks are short and lexically distinctive; real
solutions are long. This is a much bigger problem for code than it was for chat.
`length_baseline()` in the new pipeline fits a classifier on token count alone.
If that gets AUC 0.85 and your probe gets 0.90, you have mostly found a length
detector. Run it before you believe anything.
