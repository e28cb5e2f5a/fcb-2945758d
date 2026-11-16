# CHALLENGE12 — Round 12 results (2026-11-12)

## §C12 — 15-feat tune_loo.py re-optimisation

### Setup

15-feat config (ball_visible restored at col 2) swept over the
hyperparameter grid using `tune_loo.py` (G3-only selection fold).
Config selection is based on weighted LOO (§9).

**Fold weights:** G3 (organizer-provided GT) = 3×, G1/G2
(user-found games) = 1× each. Weights fixed before running
any experiment.

### G3-only LOO grid top-5

| G3_sf | h | e | bw | crf | msf |
|---|---|---|---|---|---|
| 0.5517 | 128 | 20 | 5.0 | False | 100 |
| 0.5517 | 128 | 20 | 5.0 | False | 50 |
| 0.4444 | 128 | 20 | 5.0 | False | 250 |
| 0.4444 | 128 | 20 | 5.0 | False | 175 |
| 0.4444 | 128 | 20 | 5.0 | False | 125 |

### Winning config (selected by G3 LOO)

hidden=128  epochs=20  bw=5.0  crf=False  msf=100
G3 spellF1 = **0.5517**

### Cross-check on G1 and G2

| fold | v10 | v8 | delta | weight |
|---|---|---|---|---|
| G3 (selection) | 0.5517 | 0.5325 | +0.0192 | 3× |
| G1 (cross-chk) | 0.3784 | 0.4999 | −0.1215 | 1× |
| G2 (cross-chk) | 0.5333 | 0.6028 | −0.0695 | 1× |

### Weighted decision gate (§9)

Weighted delta = (3×ΔG3 + 1×ΔG1 + 1×ΔG2) / 5
= (3×+0.0192 + 1×−0.1215 + 1×−0.0695) / 5
= **−0.0267**

v10 wins 1/3 folds. Weighted delta ≤ 0 → reject.

**REVERT to v8 — weighted LOO delta ≤ 0**

The G3 gain (+0.019) does not generalise to G1 (−0.122) or G2
(−0.070). The improvement on the selection fold is likely noise or
G3-specific overfitting. v8 (14-feat, corrected GT) remains the
submission.

### BAR-ATH result (reporting only — not used for decision)

| config | spellF1 | macroF1 |
|---|---|---|
| v8 (14-feat) | 0.4000 | — |
| v10 (15-feat, msf=100) | 0.3571 | 0.6454 |
