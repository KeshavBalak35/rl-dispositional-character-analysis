# Clean vs reward-hacking analysis

Generated from the saved artifacts in `data/clean` and `data/rh` by
`src/analyze_experiments.py`.

## Main numerical results

| Condition | Samples | Positive labels | Positive rate | Approx. 95% Wilson interval |
|---|---:|---:|---:|---:|
| Clean | 1,040 | 123 | 11.83% | 10.00% to 13.93% |
| Reward hacking | 1,040 | 134 | 12.88% | 10.98% to 15.06% |

The raw RH-minus-clean difference is **+1.06 percentage points**. The intervals overlap substantially, so this aggregate difference is modest and should not be treated as evidence of a reliable behavioral increase by itself.

## By prompt family

| Prompt family | Clean rate | RH rate |
|---|---:|---:|
| Alignment (n=200 per condition) | 16.00% | 26.50% |
| Betley (n=840 per condition) | 10.83% | 9.64% |

The condition difference is concentrated in the alignment-question subset: **+10.50 percentage points**. The Betley subset moves in the opposite direction: **-1.19 percentage points**. This interaction is more informative than the pooled rate and motivates a stratified statistical test in a follow-up analysis.

## Activation results

The saved activation tensors have shape `(1040, 32, 1, 4096)` for both conditions.

Using the mean over the 4,096 hidden dimensions at each layer as a deliberately simple score, label AUC ranged from **0.371 to 0.555** in clean and **0.384 to 0.596** in RH. The best simple score was RH layer 31 with **AUC 0.5959**, which is weak decodability and not evidence for a compact causal character direction.

The repository's existing trained/projection result is stronger: **best layer 4, AUC 0.7566**. That result should be reported separately because it is not equivalent to the layer-mean baseline computed here.

The direct RH-minus-clean mean-activation contrast is largest in absolute value at layer 28 (**-0.00768**, bootstrap 95% interval **[-0.00814, -0.00722]**), followed by layer 29 (**-0.00660**, interval **[-0.00708, -0.00613]**). These are distributional differences, not causal steering results.

## Response-text markers

Regex markers are descriptive only and are not a substitute for the original judge. Overall hit rates were:

| Marker | Clean | RH |
|---|---:|---:|
| Self-preservation | 1.92% | 2.02% |
| Autonomy/power | 0.87% | 1.25% |
| Deception/oversight | 1.73% | 2.21% |
| Reward hacking | 0.29% | 1.15% |

Within the positive-label examples, self-preservation markers occurred in **4.88%** of clean positives and **7.46%** of RH positives. Because these counts are small, they should be treated as qualitative evidence and manually audited before drawing conclusions.

## Interpretation

The current artifacts support three cautious conclusions:

1. RH does not produce a large pooled increase in the saved binary labels.
2. RH shows a noticeably higher positive-label rate on alignment questions, but not on Betley questions; this may reflect prompt-family sensitivity rather than a global character.
3. The activation evidence is compatible with some representation shift, but the simple activation baseline is weak. A stable, transferable, causally sufficient character is **not established** by these analyses.

The next decisive experiments are held-out linear probes, cross-prompt/cross-family transfer, causal activation steering with dose-response curves, and persona-prompt activation tests. All should use fixed prompt splits and report confidence intervals.

## Diagrams

- [Label rates](label_rates.png)
- [Activation AUC](activation_auc.png)
- [Activation separation](activation_separation.png)
- [Response-text markers](text_markers.png)
- [Machine-readable results](analysis.json)