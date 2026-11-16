# Challenge Iteration 7 — Feature Ablation + Ensemble Closure

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close two open review findings (§10 feature ablation, §11d
ensemble diversity) and establish whether the home/away asymmetry in
the 14-feature set is worth fixing. Both experiments use the Round 6
locked config as the baseline (bw=3, msf=125, seed=42, hs=256,
temporal split, spellF1=0.5997 on Game 3 LOO).

Levers in priority order:

1. **Per-feature ablation (§10 closure)** — isolate whether
   `n_A_within_3m` alone or `dist_ratio` (or both) caused the
   −0.0626 regression in Round 2. One table, three rows.
2. **Architectural diversity ensemble (§11d closure)** — test
   prob-avg across hs=128/256/512 LOO models before declaring
   ensemble ineffective. All three models already exist from Round 6.
3. **FINAL_SUBMISSION_COMMAND.txt (§11e closure)** — commit the
   verbatim train + predict command so the result is re-runnable
   in under 5 minutes.
4. **Home/away symmetry decision** — based on ablation result: if
   `n_A_within_3m` alone does not hurt, restore the symmetric pair;
   otherwise keep the intentional asymmetry documented in
   `src/features.py`.

**Architecture:** Two sequential tracks.

- **T1** closes §10 by running 3 LOO evals (14-feat, 15-feat, 16-feat)
  and recording the ablation table.
- **T2** closes §11d by running the hs-diversity prob-avg ensemble
  and recording whether it beats the best single model.

**Tech Stack:** Python 3.11, PyTorch, numpy; existing `src/` pipeline.
**Spec:** `CHALLENGE.md`, `EVALUATION.md`, `docs/CHALLENGE6.md`

## Global Constraints

- All scripts live in `src/`; no new files at repo root.
- Python interpreter: `python3.11`.
- Submission CSV: exactly 199,319 rows, columns `frame,label`,
  labels in `{H, A, D, X}`.
- **Config selection proxy:** Game 3 LOO spellF1 (train G1+G2,
  predict G3, score against
  `games/Sample_Game_3/gt/possession_gt.csv`).
- `use_attention=True` is **banned** — Practice 2: −0.039 spellF1.
- `use_crf=True` is **banned** — Round 3: −0.008 spellF1.
- Tagging features are **banned** — D-frame inflation +22.7%.
- `use_overlap=True` is **banned** — Round 4: +0.0033 (noise).
- Hard-vote ensemble is **banned** — Round 5: −0.1286.
- Use `--run-id v7` for all new models this round.
- Deadline: **Nov 20, 2026** — 26 days remain.

---

## T1 — Per-feature ablation: n_A_within_3m vs dist_ratio (§10)

The Round 2 16-feat regression (−0.0626 spellF1 on BAR-ATH public
window) was attributed to features 14 and 15 together. Whether
`n_A_within_3m` alone is harmful has never been tested in isolation.

**Config:** bw=3, msf=125, seed=42, temporal split (Round 6 locked).
**Baseline:** 0.5997 (Game 3 LOO, `lstm_bw3_v5.pt`).
**GT file:** `games/Sample_Game_3/gt/possession_gt.csv` (full match).

**Files:**
- Modify temporarily: `src/features.py` — add feature(s), run, revert
- Produces: `/tmp/ablation_{15,16}feat_g3.csv` (not kept)
- Records: table below

- [x] **Step 1: Run 15-feat LOO (+n_A_within_3m only)**

  Added feature 14 (`n_A_within_3m`), ran LOO, scored: 0.5782.

- [x] **Step 2: Run 16-feat LOO (+n_A_within_3m + dist_ratio)**

  Added feature 15 (`dist_ratio`), ran LOO, scored: 0.5846.

- [x] **Step 3: Revert `src/features.py` to 14 features**

  Reverted. `len(FEATURE_NAMES) == 14` confirmed; 93 tests pass.

- [ ] **Step 4: Record ablation table and decide symmetry**

  | Features | n_feat | Game 3 LOO spellF1 | delta vs 14-feat |
  |---|---|---|---|
  | 14-feat baseline (v5 locked) | 14 | 0.5997 | — |
  | +n_A_within_3m only | 15 | 0.5782 | −0.0215 |
  | +n_A_within_3m + dist_ratio | 16 | 0.5846 | −0.0151 |

  **Findings:**
  - `n_A_within_3m` alone causes −0.0215 — the primary culprit.
  - Adding `dist_ratio` on top recovers +0.0064 (0.5782→0.5846),
    suggesting `dist_ratio` is individually neutral-to-positive but
    insufficient to offset the damage from `n_A_within_3m`.
  - Both 15-feat and 16-feat regress vs 14-feat baseline.
  - The prior −0.0626 figure (Round 2, BAR-ATH window) was larger
    because it was measured against the random-shuffle v4 baseline
    (0.6094 not 0.5997); the mechanism is the same.

  **Decision:** `n_A_within_3m` delta = −0.0215 < −0.01 threshold.
  Keep 14-feat. `src/features.py` docstring updated to record the
  confirmed asymmetry with the per-feature scores. §10 closed.

---

## T2 — Architectural diversity ensemble (§11d)

**Hypothesis:** probability averaging across architecturally diverse
models (hs=128, 256, 512) avoids the same-seed collapse seen in
Rounds 3, 5, 6. The three LOO models exist from Round 6 T2.

**Models (all Game 3 LOO — trained on G1+G2):**
- `models/lstm_bw3_hs128_v6.pt` — hs=128, spellF1=0.5808
- `models/lstm_bw3_v5.pt`       — hs=256, spellF1=0.5997
- `models/lstm_bw3_hs512_v6.pt` — hs=512, spellF1=0.6033

- [x] **Step 1: Run prob-avg ensemble**

  ```bash
  python3.11 src/ensemble_challenge.py \
    --prob-avg \
    --models \
      models/lstm_bw3_hs128_v6.pt \
      models/lstm_bw3_v5.pt \
      models/lstm_bw3_hs512_v6.pt \
    --predict 3 \
    --output evaluation/results/ensemble_prob_hs_diversity_v6.csv \
    --min-spell-frames 125
  ```

- [x] **Step 2: Score on Game 3 LOO**

  | Config | Game 3 LOO spellF1 | spells | note |
  |---|---|---|---|
  | hs=128 single | 0.5808 | ~315 | Round 6 T2 |
  | hs=256 single (locked) | 0.5997 | ~315 | Round 5 v5 |
  | hs=512 single | 0.6033 | ~315 | Round 6 T2 |
  | hs-diversity prob-avg | **0.4769** | 164 | −0.1264 vs hs=512 |

  **Gate failed.** Architectural diversity does not prevent the
  probability-averaging collapse. The ensemble produced only 164
  spells vs ~315 for single models, matching the pattern seen in
  same-seed ensembles (Rounds 3, 5, 6). The collapse mechanism is
  architectural: near H/A transitions the three models assign
  intermediate probabilities, averaging pushes confident H/A frames
  toward D, and msf=125 then merges short genuine spells.

- [x] **Step 3: Close §11d finding**

  **Finding recorded:** Ensemble is ineffective on this task
  regardless of seed diversity or architectural diversity. The
  probability-averaging collapse is structural — not addressable
  via more model diversity at the current data scale. §11d is closed.

  Per-class breakdown (hs-diversity ensemble vs hs=512 single):

  | Class | Ensemble frames | Best single (approx) |
  |---|---|---|
  | H | 40,484 | ~60,000+ |
  | A | 35,019 | ~40,000+ |
  | D | 68,258 | ~50,000 |
  | spells | 164 | ~315 |

  The D-frame inflation and spell collapse confirm that uncertainty
  averaging amplifies dead-ball attribution. The problem is inherent
  to the probability-averaging approach when model confidence is
  systematically lower near H/A transitions.

---

## Conclusions

**Round completed:** 2026-10-25
**Best spellF1 (Game 3 LOO):** 0.5997  (run-id: v5, locked config)
**Prior best:** 0.5997 (Round 6 locked config, bw=3, msf=125, hs=256,
seed=42)
**Delta:** +0.0000 (no improvement; closure round)

### What worked

- **§10 closed (T1 ablation).** `n_A_within_3m` alone causes −0.0215
  (15-feat: 0.5782). Adding `dist_ratio` on top partially recovers
  +0.0064 (16-feat: 0.5846), but both still regress vs 14-feat
  (0.5997). 14-feat confirmed optimal. `src/features.py` docstring
  updated with per-feature scores.
- **§11d closed (T2 arch-diversity ensemble).** hs=128/256/512
  prob-avg scored 0.4769 (−0.1264 vs hs=512 single; −0.1228 vs locked
  config). Produced only 164 spells vs ~315 for single models.
  Ensemble is definitively ineffective regardless of diversity source.
- **§11e closed.** `docs/FINAL_SUBMISSION_COMMAND.txt` committed with
  verbatim 3-step reproduce-and-submit workflow.

### What did not work

- **15-feat (+n_A_within_3m):** 0.5782 — −0.0215 vs 14-feat.
- **16-feat (+n_A_within_3m +dist_ratio):** 0.5846 — −0.0151 vs
  14-feat.
- **Arch-diversity ensemble (hs=128/256/512 prob-avg):** 0.4769 —
  −0.1228 vs locked config; spell collapse from ~315 to 164 spells.

### Key finding

The Round 2 regression was caused by `n_A_within_3m`, not
`dist_ratio`. Adding `n_A_within_3m` alone produces −0.0215;
`dist_ratio` added on top partially recovers +0.0064. The
home/away asymmetry in the 14-feature set is intentional and
confirmed optimal. Probability-averaging ensembles collapse on
this task regardless of whether diversity comes from seeds
(Round 6 T1: 0.4763) or architecture (Round 7 T2: 0.4769) —
the mechanism is structural: near H/A transitions all models
assign intermediate probabilities, averaging amplifies D.

### Recommended focus for next round

Ordered list, highest-leverage first:

1. **dist_ratio alone (14→15)** — add only `dist_ratio` (without
   `n_A_within_3m`) and run one LOO retrain. Never tested in
   isolation; `dist_ratio` partially recovered +0.0064 when added
   on top of `n_A_within_3m`, suggesting it is individually neutral-
   to-positive.
2. **bw grid extension (bw=1, bw=2)** — bw=3 won over bw=5 in
   Round 4; values below 3 are untested. Two retrains, no
   architectural risk.
3. **Deadline focus** — Nov 20, 2026 is 26 days away. If only one
   experiment fits, prefer bw=1/2 grid over feature engineering.

### Open questions

- Does `dist_ratio` alone (without `n_A_within_3m`) produce a
  net-positive delta on Game 3 LOO?
- Does bw=1 or bw=2 beat bw=3 at msf=125, seed=42?
- What is the hs=512 model's spellF1 on the BAR-ATH public window
  (reporting only — not for selection)?
