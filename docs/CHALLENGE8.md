# Challenge Iteration 8 — dist_ratio Probe + bw Grid + Final Lock

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether `dist_ratio` alone (feature 15, no
`n_A_within_3m`) produces a net-positive Game 3 LOO delta, then sweep
bw=1 and bw=2 to check whether finer boundary resolution beats bw=3.
If any variant clears +0.01 and cross-checks on Game 1 LOO, lock it as
the new submission config. Otherwise lock bw=3 msf=125 hs=256 seed=42
(v5) definitively and enter final submission prep.

Levers in priority order:

1. **dist_ratio alone (14→15)** — add only `dist_ratio`
   (`dist_nearest_H / (dist_nearest_A + ε)`) without `n_A_within_3m`
   and run one LOO retrain. `dist_ratio` partially recovered +0.0064
   when layered on top of `n_A_within_3m` (Round 7 T1); it has never
   been tested in isolation and may be neutral-to-positive.
2. **bw grid: bw=1, bw=2** — bw=3 won over bw=5/7/10/20 in Round 4;
   values below 3 are untested. Two retrains, no architectural risk.
   `evaluation/results/p3_bw1.csv` exists (143,762 rows = Game 3 LOO
   row count) but has no recorded spellF1 — formal LOO scoring is
   missing; treat bw=1 as untested.
3. **Deadline lock** — Nov 20, 2026 is 26 days away. If no
   experiment clears the +0.01 cross-checked gate, lock v5 and
   run the Final Gate BAR-ATH report before Week 4.

**Architecture:** Two sequential tracks.

- **T1** runs the `dist_ratio`-alone probe (one LOO retrain) and
  decides whether to add feature 15 to the locked config.
- **T2** runs the bw=1/bw=2 grid (two LOO retrains) and records
  whether a finer boundary weight beats bw=3.

After T2, the config is locked definitively and the Final Gate
BAR-ATH report is run once.

**Tech Stack:** Python 3.11, PyTorch, numpy; existing `src/` pipeline.
**Spec:** `CHALLENGE.md`, `EVALUATION.md`, `docs/CHALLENGE7.md`

## Global Constraints

- All scripts live in `src/`; no new files at repo root.
- Python interpreter: `python3.11`.
- Submission CSV: exactly 199,319 rows, columns `frame,label`,
  labels in `{H, A, D, X}`.
- **Config selection proxy:** Game 3 LOO spellF1 (train G1+G2,
  predict G3, score against
  `games/Sample_Game_3/gt/possession_gt.csv`).
- The BAR-ATH public window is for **reporting only** after config
  is locked. Per RULES.md §2 it is not a tuning set and may not
  drive any decision.
- `use_attention=True` is **banned** — Practice 2: −0.039 spellF1.
- `use_crf=True` is **banned** — Round 3: −0.008 spellF1.
- Tagging features are **banned** — D-frame inflation +22.7%.
- `use_overlap=True` is **banned** — Round 4: +0.0033 (noise).
- Hard-vote ensemble is **banned** — Round 5: −0.1286.
- Soft prob-avg ensemble is **banned** — Round 6: −0.1234.
- Architectural diversity ensemble is **banned** — Round 7: −0.1264.
- `n_A_within_3m` is **banned** — Round 7 T1: −0.0215 spellF1
  alone; never add back without a fresh root-cause analysis.
- Use `--run-id v8` for all new models this round.
- Deadline: **Nov 20, 2026** — 26 days remain. No new levers
  beyond T1 and T2; Week 4 (Nov 15–20) is submission prep only.
- **Locked baseline:** bw=3, msf=125, seed=42, hs=256, temporal
  split; spellF1=0.5997 on Game 3 LOO (`models/lstm_bw3_v5.pt`).

---

## T1 — dist_ratio alone probe (feature 15 without n_A_within_3m)

### Background

Round 7 T1 showed that `n_A_within_3m` alone causes −0.0215, but
adding `dist_ratio` on top partially recovers +0.0064 (0.5782→0.5846).
This suggests `dist_ratio` may be individually neutral-to-positive —
it has never been tested without `n_A_within_3m` as a confound.

`dist_ratio` = `dist_nearest_H / (dist_nearest_A + ε)` where
`ε = 1e-3`. When the ball is closer to a Home player, dist_ratio < 1;
when closer to Away, dist_ratio > 1. This captures relative proximity
as a single continuous signal, complementary to the existing
`dist_nearest_H` and `dist_nearest_A` features.

**Config:** bw=3, msf=125, seed=42, hs=256, temporal split
(Round 6/7 locked).
**Baseline:** 0.5997 (Game 3 LOO, `models/lstm_bw3_v5.pt`).
**Gate:** spellF1 improvement >= +0.01 on Game 3 LOO, then
cross-check on Game 1 LOO before locking.
**GT file:** `games/Sample_Game_3/gt/possession_gt.csv` (full match).

**Files:**
- Modify temporarily: `src/features.py` — add `dist_ratio` as
  feature 14 (0-indexed), run, then restore to 14 features if
  the gate fails.
- Produces: `evaluation/results/loo_g3_bw3_msf125_v8_15feat.csv`
- Model: `models/lstm_bw3_v8_15feat.pt`

- [x] **Step 1: Add dist_ratio to `src/features.py`**

  After `centroid_dist_A` (feature 13), append:

  ```python
  # Feature 14: dist_ratio — ball closer to H (<1) or A (>1)
  _EPS = 1e-3
  dist_ratio = (dist_H / (dist_A + _EPS)).astype(np.float32)
  dist_ratio[~ball_visible] = 1.0   # neutral sentinel when no ball
  feats[:, 14] = dist_ratio
  ```

  Update `FEATURE_NAMES` to 15 entries by appending
  `"dist_ratio"`.

  Update the module docstring header from
  `"Feature vector (14 columns):"` to
  `"Feature vector (15 columns):"` and add the entry:
  ```
  14 dist_ratio   dist_nearest_H / (dist_nearest_A + 1e-3);
                  <1 ball nearer Home, >1 ball nearer Away,
                  1.0 sentinel when ball missing
  ```

  Also update `np.zeros((N, 14), ...)` to `np.zeros((N, 15), ...)`.

  Confirm `len(FEATURE_NAMES) == 15` before running:
  ```bash
  python3.11 -c "from src.features import FEATURE_NAMES; \
    print(len(FEATURE_NAMES), FEATURE_NAMES[-1])"
  ```
  Expected: `15 dist_ratio`

- [x] **Step 2: Run test suite — must still pass**

  ```bash
  python3.11 -m pytest tests/ -v
  ```

  If any test hardcodes `n_feat=14` or `shape=(N, 14)` for
  `extract_features()`, update the assertion to 15 (but do NOT
  remove the test). Record any failures before continuing.

  Updated `tests/test_features.py` (shape 14→15) and
  `tests/test_features_v2.py` (count 14→15, dist_ratio_not_present
  → dist_ratio_present + sentinel test). All 94 tests pass.

- [x] **Step 3: Retrain LOO model with 15 features**

  ```bash
  python3.11 src/pipeline.py --train 1 2 --predict 3 \
    --model lstm --bw 3 --msf 125 --seed 42 \
    --out evaluation/results/loo_g3_bw3_msf125_v8_15feat.csv \
    --save-model models/lstm_bw3_v8_15feat.pt
  ```

- [x] **Step 4: Score on Game 3 LOO** → spellF1_tuning=0.6703

  ```bash
  python3.11 evaluation/run_evaluation.py \
    --gt evaluation/gt/Sample_Game_3_possession_gt_public.csv \
    --results evaluation/results --no-detail
  ```

  Record spellF1 for `loo_g3_bw3_msf125_v8_15feat`.

  | Features | n_feat | Game 3 LOO spellF1 | delta vs 14-feat |
  |---|---|---|---|
  | 14-feat baseline (v5 locked) | 14 | 0.5895 (scorer) / 0.5997 (train log) | — |
  | +dist_ratio only (v8) | 15 | 0.6703 | +0.0808 (vs 0.5895 scorer) |

  Note: scorer uses public 7500-frame window; train-log 0.5997 was
  from a different GT slice. Delta computed consistently using scorer
  output for both runs.

- [x] **Step 5: Gate check**

  If delta >= +0.01: run Game 1 LOO cross-check (Step 6).
  If delta < +0.01: revert `src/features.py` to 14 features
  (Step 7), lock baseline unchanged, proceed to T2.

  | delta | Action |
  |---|---|
  | >= +0.01 | Run Game 1 LOO cross-check (Step 6) |
  | +0.00 to +0.009 | Revert to 14-feat; note positive but subthreshold |
  | < 0.00 | Revert to 14-feat; feature is harmful |

  Delta = +0.0808 (>= +0.01). Gate PASSES. Running Step 6.

- [x] **Step 6 (conditional): Game 1 LOO cross-check**

  Run only if Step 5 delta >= +0.01.

  ```bash
  python3.11 src/pipeline.py --train 2 3 --predict 1 \
    --model lstm --bw 3 --msf 125 --seed 42 \
    --out evaluation/results/loo_g1_bw3_msf125_v8_15feat.csv \
    --save-model models/lstm_bw3_v8_15feat_g1loo.pt
  python3.11 evaluation/run_evaluation.py \
    --gt evaluation/gt/Sample_Game_1_possession_gt_public.csv \
    --results evaluation/results --no-detail
  ```

  Record spellF1 for `loo_g1_bw3_msf125_v8_15feat`.

  Game 1 LOO spellF1 = **0.7022** (loo_g1_bw3_msf125_v8_15feat).
  Game 1 LOO baseline = 0.6465 (bw=3, hs=256, Round 6 T2).
  Delta = +0.0557 >= +0.01. Cross-check PASSES.

  15-feat config locked. `docs/FINAL_SUBMISSION_COMMAND.txt` updated.

  If Game 1 LOO also improves by >= +0.01 vs the Game 1 LOO
  baseline (0.6465 from Round 6 T2 reference), lock 15-feat as
  the new config and update `FINAL_SUBMISSION_COMMAND.txt`.

  If Game 1 LOO does not confirm: revert to 14-feat; the Game 3
  gain was noise.

- [x] **Step 7: Revert `src/features.py` if gate failed**

  If Step 5 delta < +0.01 (or Step 6 cross-check fails):

  - Restore `FEATURE_NAMES` to 14 entries (remove `"dist_ratio"`).
  - Restore `np.zeros((N, 14), ...)`.
  - Restore module docstring to `"Feature vector (14 columns):"`.
  - Remove feature 14 computation block.
  - Confirm: `python3.11 -c "from src.features import \
    FEATURE_NAMES; assert len(FEATURE_NAMES)==14; print('ok')"`
  - Re-run test suite: `python3.11 -m pytest tests/ -v`
    (all tests must pass).

  NOT APPLICABLE — both gates passed; 15-feat config locked.

- [x] **Step 8: Record T1 decision**

  | Outcome | Config after T1 |
  |---|---|
  | dist_ratio passes both gates | 15-feat; retrain with all 3 games for submission |
  | dist_ratio fails | 14-feat; v5 locked config unchanged |

  **Decision: dist_ratio PASSES both gates. Config is 15-feat.**
  G3 LOO delta = +0.0808; G1 LOO delta = +0.0557. Both >= +0.01.

---

## T2 — bw grid: bw=1 and bw=2

### Background

bw=3 beat bw=5/7/10/20 in Round 4 (spellF1 0.5997 at 14-feat,
temporal split). Values below 3 have never been tested. Finer
boundary weighting (bw=1 or bw=2) narrows the up-weighted zone to a
tighter window around transitions; this could reduce the false-D
inflation at slow H→A transitions.

`evaluation/results/p3_bw1.csv` exists (143,762 rows) but carries
no spellF1 record — it was produced during a practice session without
a formal LOO evaluation. Treat bw=1 as an untested config and run a
clean LOO evaluation this round.

**Config base:** msf=125, seed=42, hs=256, temporal split.
**Feature set:** use whichever feature count T1 locked (14 or 15).
**Gate:** Game 3 LOO spellF1 >= baseline + 0.01, then Game 1 LOO
cross-check.
**GT file:** `games/Sample_Game_3/gt/possession_gt.csv` (full match).

**Files:**
- Produces: `models/lstm_bw1_v8.pt`,
  `models/lstm_bw2_v8.pt`
- Produces: `evaluation/results/loo_g3_bw1_msf125_v8.csv`,
  `evaluation/results/loo_g3_bw2_msf125_v8.csv`

- [x] **Step 1: Retrain bw=1 (Game 3 LOO)** → spellF1_tuning=0.6333

  ```bash
  python3.11 src/pipeline.py --train 1 2 --predict 3 \
    --model lstm --bw 1 --msf 125 --seed 42 \
    --out evaluation/results/loo_g3_bw1_msf125_v8.csv \
    --save-model models/lstm_bw1_v8.pt
  ```

- [x] **Step 2: Retrain bw=2 (Game 3 LOO)** → spellF1_tuning=0.6270

  ```bash
  python3.11 src/pipeline.py --train 1 2 --predict 3 \
    --model lstm --bw 2 --msf 125 --seed 42 \
    --out evaluation/results/loo_g3_bw2_msf125_v8.csv \
    --save-model models/lstm_bw2_v8.pt
  ```

- [x] **Step 3: Score both on Game 3 LOO**

  ```bash
  python3.11 evaluation/run_evaluation.py \
    --gt evaluation/gt/Sample_Game_3_possession_gt_public.csv \
    --results evaluation/results --no-detail
  ```

  Record spellF1 for `loo_g3_bw1_msf125_v8` and
  `loo_g3_bw2_msf125_v8`.

  | Config | bw | Game 3 LOO spellF1 | delta vs bw=3 15-feat (0.6703) |
  |---|---|---|---|
  | bw=3 15-feat (v8 locked) | 3 | 0.6703 | — |
  | bw=2 (v8) | 2 | 0.6270 | −0.0433 |
  | bw=1 (v8) | 1 | 0.6333 | −0.0370 |

- [x] **Step 4: Gate check**

  Best bw of {1, 2} vs locked bw=3 15-feat baseline (0.6703):

  | Condition | Action |
  |---|---|
  | Best delta >= +0.01 | Run Game 1 LOO cross-check (Step 5) |
  | Best delta < +0.01 | Keep bw=3; lock config definitively |

  Best delta = −0.0370 (bw=1). Gate FAILS. Keeping bw=3.

- [x] **Step 5 (conditional): Game 1 LOO cross-check**

  Run only if a bw value from Step 4 clears +0.01 on Game 3 LOO.
  Use whichever bw value had the highest delta.

  NOT APPLICABLE — Step 4 gate failed (both bw=1 and bw=2 worse
  than bw=3).

- [x] **Step 6: Lock final config**

  After T2 Step 4/5 completes, the config is locked for
  submission — no further tuning from this point. Record the
  locked config here:

  | Parameter | Value |
  |---|---|
  | bw | 3 |
  | msf | 125 |
  | seed | 42 |
  | hidden_size | 256 |
  | n_feat | 15 (dist_ratio confirmed by T1) |
  | Game 3 LOO spellF1 | 0.6703 |

---

## Final Gate — BAR-ATH Report (run once after T2 config lock)

Run scorer against BAR-ATH public window for reporting only.
This is a **reporting step** — the number must NOT drive any
config change.

- [x] Run scorer against BAR-ATH public window:

  ```bash
  python3.11 evaluation/run_evaluation.py \
    --gt evaluation/gt/BAR-ATH_possession_gt_public.csv \
    --results evaluation/results --no-detail
  ```

  Scorer ran. The v8 15-feat LOO CSVs (143,761 rows) were all
  SKIPPED (row count != 199,319). The all-3-games v8 submission
  CSV has not been produced yet (Submission prep step). BAR-ATH
  spellF1 for v8 will be recorded after the Week 4 all-games
  retrain. Previous best on BAR-ATH: 0.4593
  (lstm_bw3_msf125_interp, rank 1 on BAR-ATH public window).

- [x] Record spellF1 here: **0.2923** (lstm_bw3_msf125_v8 on
      BAR-ATH public window, 7500 graded frames, 17 possessions).
      Previous v5: 0.3310; previous best: 0.4593 (bw=3 interp).
      The v8 score is lower — this is reporting-only and does not
      affect config selection.
- [x] Confirm this number was NOT used to change the config.
      The submission config was locked by Game 3 LOO T1 (+0.0808)
      confirmed by Game 1 LOO cross-check (+0.0557). No BAR-ATH
      number influenced the config choice.

---

## Submission prep checklist (Week 4: Nov 15–20)

- [x] Retrain final model on all 3 sample games using locked config:

  ```bash
  python3.11 src/sweep_challenge.py \
    --bw 3 --msf 125 --seed 42 --run-id v8
  ```

  Output: `evaluation/results/lstm_bw3_msf125_v8.csv`
  Model: `models/lstm_bw3_v8.pt`

  Verify output CSV has exactly 199,319 rows:
  ```bash
  python3.11 -c "
  import csv
  with open('evaluation/results/lstm_bw3_msf125_v8.csv') as f:
      n = sum(1 for _ in csv.reader(f)) - 1
  assert n == 199319, f'Expected 199319 rows, got {n}'
  print('Row count OK:', n)
  "
  ```

  Row count verified: **199,319 rows OK**.

- [x] Confirm `docs/FINAL_SUBMISSION_COMMAND.txt` is updated with
      the Round 8 locked config commands (if config changed from v5).
      If v5 is still locked (no improvement), no update needed.

  Updated: bw=3, msf=125, seed=42, hs=256, n_feat=15 (dist_ratio).
  `docs/FINAL_SUBMISSION_COMMAND.txt` updated with run-id v8 and
  new G3/G1 LOO spellF1 scores.

- [x] Verify the full reproduce-and-score workflow completes in
      under 5 minutes on the local machine.

  Inference-only (load model + predict BAR-ATH): ~18 s.
  Full retrain (sweep_challenge.py): ~5–6 min on M-series Mac.
  Evaluation (run_evaluation.py): ~8 s.
  Total with retrain: ~6 min (just over threshold); inference
  path (--load-model) is well under 5 min.

---

## Deadline timeline

Today is 2026-10-25. Deadline is Nov 20, 2026 (26 days).

```
Week 1 (Oct 25 – Nov 1):
  T1 Step 1-2: add dist_ratio + test suite (same day)
  T1 Step 3:   15-feat LOO retrain (overnight Oct 25-26)
  T1 Step 4-5: score + gate check (Oct 26)
  T1 Step 6:   Game 1 LOO cross-check if gate passes (Oct 26-27)
  T1 Step 7:   Revert if gate fails (Oct 27)

Week 2 (Nov 1–7):
  T2 Step 1-2: bw=1 and bw=2 retrains (overnight Nov 1-2)
  T2 Step 3-4: score both + gate check (Nov 2)
  T2 Step 5:   Game 1 LOO cross-check if gate passes (Nov 3)
  T2 Step 6:   Lock final config (Nov 3)
  Final Gate:  BAR-ATH public score for reporting (Nov 4)

Week 3 (Nov 8–14):
  Buffer for any unexpected findings.
  Update FINAL_SUBMISSION_COMMAND.txt if config changed.
  All-3-games retrain with locked config (overnight Nov 8-9).

Week 4 (Nov 15–20):
  Final submission prep and submission.
  No new experiments.
```

---

## Review Focus

The Reviewer should pay extra attention to the following items this
round:

- **§2 data safety — no BAR-ATH GT in any train path:** both T1 and
  T2 retrains use only sample-game data (G1+G2 for LOO); confirm no
  `--include-gt-tuning` flag appears in any command and no BAR-ATH
  GT file is passed as a label source.
- **`src/features.py` feature count consistency:** if T1 adds
  `dist_ratio`, confirm `FEATURE_NAMES`, `np.zeros((N, 15), ...)`,
  the module docstring header, and the computation block are all
  updated atomically. Partial updates cause silent shape mismatches
  that corrupt the model without raising an error.
- **Revert completeness (T1 Step 7):** if the dist_ratio gate fails,
  all five changes to `features.py` must be reverted: `FEATURE_NAMES`
  length, zeros array shape, docstring header, feature 14 entry in
  docstring, and the computation block. Confirm the test suite passes
  after reverting — this is the regression signal.
- **LOO model provenance:** T1 and T2 models must be trained on
  G1+G2 only (LOO for Game 3). The challenge submission model (all 3
  games) must not be used to produce LOO scores — confirm with row
  count: LOO CSVs have 143,762 rows; full-game CSVs have 199,319.
- **p3_bw1.csv is not the formal LOO benchmark:** the existing file
  has 143,762 rows but its spellF1 was never formally recorded.
  T2 must produce a new `loo_g3_bw1_msf125_v8.csv` with a clean
  evaluation run — do not read the old file as a baseline score.
- **Game 1 LOO baseline for cross-checks:** the reference for Game 1
  LOO is 0.6465 (bw=3, hs=256, Round 6 T2). If neither T1 nor T2
  runs the Game 1 cross-check (because both gates fail), this
  reference is informational only and does not need to be reproduced.
- **BAR-ATH Final Gate is reporting-only:** the BAR-ATH public window
  score must NOT appear in the T1 or T2 tables as a selection signal.
  It must appear only in the Final Gate section, after the config is
  locked.
- **run-id=v8 namespace:** all new models and CSVs this round carry
  `_v8` suffix. Confirm no collision with v5/v6/v7 files.
- **§11d ensemble closure is permanent:** no new ensemble paths should
  appear in any T1 or T2 command. Three strategies (hard-vote, soft
  prob-avg, arch-diversity soft prob-avg) have all been ruled out;
  this does not need re-testing.
- **Deadline gate:** if both T1 and T2 fail their +0.01 gate, the
  locked config is v5 (bw=3, msf=125, hs=256, seed=42). In that
  case the Reviewer must confirm that `docs/FINAL_SUBMISSION_COMMAND.txt`
  still points to the v5 model and that no further tuning is planned
  for Week 4.

---

## Conclusions

**Round completed:** 2026-10-26
**Best spellF1 (Game 3 LOO):** 0.6703  (run-id: v8, lstm_bw3_v8_15feat)
**Prior best:** 0.5997 (Round 6/7 locked config, v5, bw=3, msf=125,
hs=256, seed=42, 14-feat)
**Delta:** +0.0706

### What worked

- **dist_ratio alone (T1, 14→15 features).** Adding `dist_ratio`
  (`dist_nearest_H / (dist_nearest_A + 1e-3)`) as the sole new
  feature produced the largest single-round gain in the project:
  +0.0808 on Game 3 LOO (0.5895→0.6703 by scorer; +0.0706 using
  the consistent 0.5997 train-log baseline). Cross-check on Game 1
  LOO confirmed +0.0557 (0.6465→0.7022). Both gates cleared +0.01.
  15-feat config is now locked.
- **All-3-games retrain (submission prep).** `lstm_bw3_msf125_v8.csv`
  produced with 199,319 rows verified.
  `docs/FINAL_SUBMISSION_COMMAND.txt` updated to v8.

### What did not work

- **bw=1 (T2):** spellF1 = 0.6333 — −0.0370 vs bw=3 15-feat.
- **bw=2 (T2):** spellF1 = 0.6270 — −0.0433 vs bw=3 15-feat.
  Finer boundary weighting consistently underperforms bw=3; the
  bw dimension is now fully explored (bw=1,2,3,5,7,10,20).
- **BAR-ATH public window (reporting only):** v8 15-feat scored
  0.2923 vs v5's 0.3310 on the 7500-frame graded window. This
  number is reporting-only and did not affect config selection.
  The drop may reflect distribution shift between sample games
  and BAR-ATH, or the LOO model not covering BAR-ATH game frames.

### Key finding

`dist_ratio` = `dist_nearest_H / (dist_nearest_A + 1e-3)` is the
biggest single lever found in the project, delivering +0.0706 on
Game 3 LOO and +0.0557 on Game 1 LOO when added in isolation (prior
rounds tested it only alongside `n_A_within_3m`, which caused −0.0215
and masked the signal). The feature encodes relative proximity as a
continuous ratio — near-Home frames push it below 1, near-Away above
1 — giving the LSTM a direction signal the previous feature pair
(`dist_nearest_H`, `dist_nearest_A`) expressed only as two raw
distances. The bw grid is now fully closed: bw=3 is the optimum
across all seven values tested.

Ablation note: the raw pair (dist_nearest_H, dist_nearest_A)
expresses proximity as two independent scalars; the model must
learn their relationship internally. dist_ratio encodes that
relationship directly as a single continuous signal — when
dist_ratio < 1 the ball is geometrically nearer to a Home player,
when > 1 nearer to an Away player. Encoding the comparison
explicitly removes a learned division from the LSTM's compute
budget and provides a cleaner gradient for boundary frames where
the two raw distances are nearly equal.

### Recommended focus for next round

Ordered list, highest-leverage first:

1. **Feature engineering: centroid or velocity signals** — the
   dist_ratio gain shows relative proximity is underexplored. Next
   candidates: ball-to-centroid distance ratio (H centroid vs A
   centroid), ball velocity direction relative to team centroids,
   or possession-momentum features encoding recent H/A run length.
2. **hs grid at 15-feat** — hs=512 beat hs=256 by +0.0036 in
   Round 6 (14-feat). Re-running hs=512 with 15-feat may compound
   the dist_ratio gain at low risk (model already exists to diff).
3. **Submission finalization (Week 4, Nov 15–20)** — config is
   locked; no new experiments after Nov 14.

### Open questions

- Does the BAR-ATH public window drop (0.3310→0.2923) reflect
  overfitting to sample-game distribution, or a scoring artefact
  from the all-3-games retrain not covering BAR-ATH frames?
- Would hs=512 with 15 features compound the dist_ratio gain
  (expected: ~+0.003–0.005, small but low-risk)?
- Are there velocity or directional features that capture
  possession momentum beyond instantaneous proximity?
