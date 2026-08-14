# Does RL Create the Same Kind of Stable, Transferable Dispositional Character That SFT Creates in Emergent Misalignment?

Investigating whether reinforcement learning installs a stable, causally-steerable "character" in language model activations the same way supervised fine-tuning (SFT) is known to, when a model becomes misaligned through reward hacking.

## Overview

Language models trained with RL can learn to reward hack, exploit their reward signal instead of genuinely solving a task, and this can generalize into broader misalignment (deception, sabotage) that was never directly trained for. Separately, SFT on harmful data has been shown to install a persistent, persona-activatable "character" direction in a model's activation space. No prior work has tested whether RL-induced reward hacking produces this same kind of stable character, or whether it's a looser collection of context-specific strategies.

We test this using open-weight clean and reward-hacking OLMo-7B checkpoints (trained via SDF → Instruct SFT → DAPO) from the [AISI reproduction](https://github.com/UKGovernmentBEIS/reward-hacking-misalignment) of Anthropic's emergent misalignment work. We extract a candidate reward-hacking direction from model activations and test it through three independent lenses:

- **Linear probing** — is the direction decodable from hidden states?
- **Causal steering** — does adding/removing the direction change behavior?
- **Persona prompting** — does naturalistic framing activate the direction without intervention?

## Team

Prashan Adhikari, Ishika Aahna, Abdulsalam Alshahrani, Keshav Balakrishna — mentored by Andrew Liao.

## Repository Structure

```
.
├── LICENSE
├── data/
│   ├── clean/                      # Clean (non-reward-hacking) model outputs
│   │   ├── clean_activations.pt
│   │   ├── clean_labels.json
│   │   ├── clean_responses.json
│   │   └── clean_responses_partial.json
│   ├── rh/                         # Reward-hacking model outputs
│   │   ├── rh_activations.pt
│   │   ├── rh_labels.json
│   │   ├── rh_responses.json
│   │   └── rh_responses_partial.json
│   └── prompts/
│       ├── alignment_questions.py
│       └── betley.py
├── results/
│   └── auc_results_1.json          # Linear probe AUC results (chat-eval, post extraction-bug fix)
├── src/
│   ├── chat_evals/                 # Chat-eval pipeline (alignment + Betley questions)
│   ├── coding_evals/
│   │   └── coding_pipeline.py      # Coding-environment pipeline (APPS/MBPP/HumanEval/CodeContests)
│   ├── clean_model_pipeline.py
│   ├── rh_model_pipeline.py
│   └── collect_activations.py
└── README.md
```


## Setup

**Chat-eval pipeline** (`src/chat_evals/`, `src/clean_model_pipeline.py`, `src/rh_model_pipeline.py`, `src/collect_activations.py`): runs on Kaggle (T4 x2 GPU). Requires `datasets==3.6.0` (newer versions dropped support for the script-based `codeparrot/apps` dataset), Kaggle Internet access enabled, and an Anthropic API key for grading.

**Coding-eval pipeline** (`src/coding_evals/coding_pipeline.py`):

> Note: the fuller pipeline reviewed separately (`generation.py`, `backends.py`, `schemas.py`, `splits.py`, `verification.py`, Docker sandbox grader, etc.) isn't reflected in the repo yet as of this structure check, `coding_evals/` currently has a single `coding_pipeline.py`. Update this section once that pipeline is pushed.

Activation extraction and causal steering require in-process model weights (not a served API endpoint), since hidden states and steering hooks aren't exposed over HTTP.

## Current Status

- **Chat-eval results (alignment + Betley questions, 1040 prompts):** initial linear probe showed a flat AUC (~0.73–0.76) with no layer peak. After fixing an activation-extraction bug, both models now show a proper mid-layer peak (clean: layer 12, AUC 0.837; RH: layer 21, AUC 0.830) — correlational evidence consistent with a character-like representation, not yet causal confirmation. See `results/auc_results_1.json`.
- **Coding-eval pipeline:** in development (`src/coding_evals/coding_pipeline.py`). A separate, more structurally robust version addressing two bugs found in the original notebook — (1) activation extraction that accidentally pooled prompt tokens instead of response tokens, and (2) train/test splits that could leak the same problem's repeated samples across both sides — is under review before merging.
- **In progress:** linear probing on the coding-domain dataset (APPS/MBPP/HumanEval/CodeContests, ~3800 problems), followed by causal steering and persona prompting.

<!--
FINAL RESULTS — replace "Current Status" above with whichever of these three
matches the actual outcome once all three lenses (probe, steer, persona) are
run. Pick one, delete the others, and fill in the real numbers.
-->

<!-- OPTION A: UNIFIED CHARACTER (all three lenses converge) -->
## Results

All three lenses converge: causal steering along the extracted direction reliably induces/suppresses reward-hacking behavior (dose-response effect, beats the random-direction control), linear probing decodes hacking intent with high AUC that holds out-of-distribution, and persona prompting activates the same direction through naturalistic framing alone (cosine alignment [X] with the steering direction). This indicates RL-induced reward hacking installs a stable, transferable character analogous to what Su et al. (2026) found under SFT, evidence that character formation is a general property of misalignment across training paradigms, not an SFT-specific artifact. Full numbers and figures in [link to results/writeup].

<!-- OPTION B: FRAGMENTED (lenses disagree / fail to converge) -->
## Results

The three lenses do not converge: [steering fails to beat the random-direction control / probe AUC collapses out-of-distribution / persona prompts fail to activate the direction — specify which]. This suggests RL-induced reward hacking is representationally efficient but fragmented, driven by several separate, context-specific strategies rather than one stable disposition, revealing an asymmetry between how SFT and RL produce misalignment. Full numbers and figures in [link to results/writeup].

<!-- OPTION C: PARTIAL (some lenses converge, others don't) -->
## Results

Partial convergence across the three lenses: [specify which converged and which didn't, e.g. "steering and probing show a clear effect, but persona prompting shows only weak activation"]. This points to a character that is [causally real but not fully naturalistic-accessible / decodable but not fully causal / etc. — specify], itself informative about how RL and SFT differ mechanistically rather than a simple yes/no answer. Full numbers and figures in [link to results/writeup].


## Known Methodology Notes

- **Prompt conditions differ by dataset.** APPS and CodeContests support all 8 system-prompt variants (including `no_hints`, a naturalistic no-mention baseline); HumanEval/MBPP only support 5. We report this asymmetry explicitly rather than fabricating a `no_hints` condition for datasets that don't have one in the source repo.
- **APPS/CodeContests are difficulty-filtered; HumanEval/MBPP are not.** The source repo excludes APPS problems solvable by a 32B baseline model (real usable count: 1,131 of 3,000). This means cross-dataset hack-rate comparisons are confounded with problem difficulty and should be read as descriptive per-dataset results, not a clean disposition comparison.

## Reference

Golechha, S., Black, S., & Bloom, J. (2026). *Some natural emergent misalignment from reward hacking in non-production RL.* UK AI Security Institute. [LessWrong](https://www.lesswrong.com/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in) · [GitHub](https://github.com/UKGovernmentBEIS/reward-hacking-misalignment)