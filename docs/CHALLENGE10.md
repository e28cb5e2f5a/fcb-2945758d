# Challenge Iteration 10 — y-Spread Probe + Final Submission Prep

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attempt the one remaining untested lever from §11a of
`.local/PR.md` — adding y-spread (or 2-D compactness) to the team
spread features — with the same +0.01 Game 3 LOO gate that governs
all config decisions. If the probe fails (as three Round 9 probes
did), immediately pivot to final submission verification and
prepare for submission. If it passes, lock the new config, retrain
the all-3-games model, and then prepare for submission.

Levers in priority order:

1. **T1 — y-spread probe (last §11a item):** `spread_H` and
   `spread_A` currently capture only the x-axis standard deviation
   of player positions. Adding the y-axis std (or the 2-D mean
   pairwise distance) gives the model team-shape information in
   both axes. This is listed explicitly in §11a of the review
   checklist as a "known gap; untested." One solo probe; same LOO
   harness as all prior rounds.
2. **T2 — Final submission verification:** regardless of T1 outcome,
   reproduce the submission CSV from scratch to confirm
   reproducibility, validate row count, update any stale lines in
   `docs/FINAL_SUBMISSION_COMMAND.txt`, run the full test suite,
   and record the BAR-ATH public window score.

**Architecture:** T1 runs first (fast, single LOO retrain). If T1
fails its gate, T2 begins immediately. T2 is mandatory regardless
of T1 outcome.

**Tech Stack:** Python 3.11, PyTorch, numpy; existing `src/` pipeline.
**Spec:** `CHALLENGE.md`, `EVALUATION.md`, `docs/CHALLENGE9.md`

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
- Hard-vote ensemble is **banned** — Rounds 3/5: −0.1286.
- Soft prob-avg ensemble is **banned** — Round 6: −0.1234.
- Architectural diversity ensemble is **banned** — Round 7: −0.1264.
- `n_A_within_3m` is **banned** — Round 7: −0.0215 spellF1.
- bw=1 and bw=2 are **banned** — Round 8: both worse than bw=3.
- `centroid_ratio` is **banned** — Round 9: −0.0492.
- `ball_dir_H` is **banned** — Round 9: −0.0808.
- `hs=512` is **banned** — Round 9: −0.0703; Round 6: +0.0036
  (noise); never reliably beats hs=256.
- Use `--run-id v10` for all new models this round.
- Deadline: **Nov 20, 2026** — 25 days remain. T1 is the last
  permitted experiment; T2 is mandatory housekeeping.
- **Locked baseline:** bw=3, msf=125, seed=42, hs=256, 15-feat,
  temporal split; spellF1=0.6703 on Game 3 LOO
  (`models/lstm_bw3_v8.pt`, submission CSV
  `evaluation/results/lstm_bw3_msf125_v8.csv`).

---

## T1 — y-spread feature probe (16th feature)

### Background

`spread_H` (feature 10) and `spread_A` (feature 11) each compute
the standard deviation of x-positions across the team's visible
players in metres. The y-axis is ignored. Section §11a of the
review checklist flags this explicitly: "adding y-spread or 2-D
compactness (mean pairwise distance) gives the model team shape
information in both axes; not a bug but a known gap; untested."

The proposed candidate is **`spread2d_ratio`**: a single scalar
encoding the relative 2-D compactness of the two teams as a ratio
`spread2d_H / (spread2d_A + ε)`, where `spread2d_X` is the mean
pairwise distance of visible players on team X in metres, or
alternatively the Frobenius std of the (x, y) position matrix.
This mirrors the `dist_ratio` approach (a ratio beats the raw pair)
and is directly motivated by the largest single gain in the project
(+0.0706 from `dist_ratio` in Round 8).

A simpler variant — **`spread_y_H`** and **`spread_y_A`** (two
features, std of y-positions) — is also an option but adds two
columns rather than one, increasing the risk of the same noise
degradation seen with `centroid_ratio` and `ball_dir_H`. The
recommended approach is **one new feature (`spread2d_ratio`)** to
keep the probe clean.

**Config:** bw=3, msf=125, seed=42, hs=256, temporal split.
**Baseline:** 0.6703 (Game 3 LOO, `models/lstm_bw3_v8_15feat.pt`,
15-feat locked config).
**Gate:** spellF1 improvement >= +0.01 on Game 3 LOO, then
cross-check on Game 1 LOO before locking.
**GT file:** `games/Sample_Game_3/gt/possession_gt.csv`
(full match).

**Files:**
- Modify: `src/features.py` — add feature 15, run, revert if
  gate fails.
- Produces:
  `evaluation/results/loo_g3_bw3_msf125_v10_16feat.csv`
- Model: `models/lstm_bw3_v10_16feat.pt`

### T1 Steps

- [x] **Step 1: Add `spread2d_ratio` to `src/features.py`**

  After feature 14 (`dist_ratio`), append:

  ```python
  # Feature 15: spread2d_ratio — relative 2-D compactness of teams
  # std of (x,y) positions concatenated gives 2-D spread per team.
  # spread2d_X = std of all (x, y) coords of visible players on
  #              team X; equals sqrt(var_x + var_y) up to centering.
  # ratio < 1: Home team more compact; > 1: Away more compact.
  _EPS2 = 1e-3
  home_xy_flat = home_m.reshape(N, -1)    # (N, N_PLAYERS_MAX*2)
  away_xy_flat = away_m.reshape(N, -1)    # (N, N_PLAYERS_MAX*2)
  spread2d_H = np.nanstd(home_xy_flat, axis=1).astype(np.float32)
  spread2d_A = np.nanstd(away_xy_flat, axis=1).astype(np.float32)
  spread2d_ratio = (spread2d_H / (spread2d_A + _EPS2)).astype(
      np.float32)
  feats[:, 15] = spread2d_ratio
  ```

  Update `np.zeros((N, 15), ...)` → `np.zeros((N, 16), ...)`.

  Update `FEATURE_NAMES` to 16 entries by appending
  `"spread2d_ratio"`.

  Update the module docstring header from
  `"Feature vector (15 columns):"` to
  `"Feature vector (16 columns):"` and add:

  ```
  15 spread2d_ratio  nanstd(home_xy_flat) / (nanstd(away_xy_flat)
                     + 1e-3); < 1 Home more compact, > 1 Away
                     more compact
  ```

  Confirm before running:
  ```bash
  python3.11 -c "from src.features import FEATURE_NAMES; \
    print(len(FEATURE_NAMES), FEATURE_NAMES[-1])"
  ```
  Expected: `16 spread2d_ratio`

- [x] **Step 2: Run test suite — must still pass** → 94/94 passed

  ```bash
  python3.11 -m pytest tests/ -v
  ```

  If any test hardcodes `n_feat=15` or `shape=(N, 15)` for
  `extract_features()`, update the assertion to 16 (do NOT remove
  the test). Record any failures before continuing.

- [x] **Step 3: Retrain LOO model with 16 features**

  ```bash
  python3.11 src/pipeline.py --train 1 2 --predict 3 \
    --model lstm --bw 3 --msf 125 --seed 42 \
    --out \
      evaluation/results/loo_g3_bw3_msf125_v10_16feat.csv \
    --save-model models/lstm_bw3_v10_16feat.pt
  ```

- [x] **Step 4: Score on Game 3 LOO** → spellF1=0.6526

  ```bash
  python3.11 evaluation/run_evaluation.py \
    --gt \
      evaluation/gt/Sample_Game_3_possession_gt_public.csv \
    --results evaluation/results --no-detail
  ```

  Record spellF1 for `loo_g3_bw3_msf125_v10_16feat`.

  | Features | n_feat | Game 3 LOO spellF1 | delta vs 15-feat |
  |---|---|---|---|
  | 15-feat baseline (v8 locked) | 15 | 0.6703 | — |
  | +spread2d_ratio (v10) | 16 | 0.6526 | -0.0177 |

- [x] **Step 5: Gate check** → delta=-0.0177 (FAILS gate); reverting to 15-feat; locking Round 8 config

  | delta | Action |
  |---|---|
  | >= +0.01 | Run Game 1 LOO cross-check (Step 6) |
  | < +0.01 | Revert to 15-feat (Step 7); proceed to T2 |

- [x] **Step 6 (conditional): Game 1 LOO cross-check** → SKIPPED (gate failed at Step 5)

  Run only if Step 5 delta >= +0.01.

  ```bash
  python3.11 src/pipeline.py --train 2 3 --predict 1 \
    --model lstm --bw 3 --msf 125 --seed 42 \
    --out \
      evaluation/results/loo_g1_bw3_msf125_v10_16feat.csv \
    --save-model models/lstm_bw3_v10_16feat_g1loo.pt
  python3.11 evaluation/run_evaluation.py \
    --gt \
      evaluation/gt/Sample_Game_1_possession_gt_public.csv \
    --results evaluation/results --no-detail
  ```

  Game 1 LOO baseline = 0.7022 (v8, Round 8 T1 Step 6).

  If Game 1 LOO delta >= +0.01 vs 0.7022: lock 16-feat as new
  config. Update `docs/FINAL_SUBMISSION_COMMAND.txt`. Retrain
  all-3-games model (Step 8).

  If Game 1 LOO delta < +0.01: revert to 15-feat (Step 7).

- [x] **Step 7 (conditional): Revert `src/features.py` if gate
  failed or cross-check failed** → Reverted; 94/94 tests pass

  If Step 5 delta < +0.01, or Step 6 delta < +0.01:

  - Remove `"spread2d_ratio"` from `FEATURE_NAMES`.
  - Restore `np.zeros((N, 15), ...)`.
  - Restore module docstring header to
    `"Feature vector (15 columns):"`.
  - Remove the feature 15 docstring entry.
  - Remove the feature 15 computation block.
  - Confirm: `python3.11 -c "from src.features import \
    FEATURE_NAMES; assert len(FEATURE_NAMES)==15; print('ok')"`
  - Re-run test suite: `python3.11 -m pytest tests/ -v`

- [x] **Step 8 (conditional): Retrain all-3-games model if 16-feat
  locked** → SKIPPED (T1 gate failed; 15-feat config unchanged)

  Run only if both Step 5 and Step 6 gates pass.

  ```bash
  python3.11 src/sweep_challenge.py \
    --bw 3 --msf 125 --seed 42 --run-id v10
  ```

  Verify output CSV:
  ```bash
  python3.11 -c "
  import csv
  with open('evaluation/results/lstm_bw3_msf125_v10.csv') as f:
      n = sum(1 for _ in csv.reader(f)) - 1
  assert n == 199319, f'Expected 199319 rows, got {n}'
  print('Row count OK:', n)
  "
  ```

  Update `docs/FINAL_SUBMISSION_COMMAND.txt` to reference v10 and
  the new spellF1 scores.

### T1 decision record

Fill this table after T1:

| Feature | G3 LOO spellF1 | delta | G1 LOO spellF1 | Locked? |
|---|---|---|---|---|
| spread2d_ratio | 0.6526 | -0.0177 | N/A (skipped) | No |

---

## T2 — Final submission verification (mandatory)

This track runs regardless of T1 outcome. Its purpose is to
confirm reproducibility of the submission CSV, close the one
stale line in `docs/FINAL_SUBMISSION_COMMAND.txt`, and run the
BAR-ATH public window as a reporting-only final gate.

### T2a — Fix stale line in `docs/FINAL_SUBMISSION_COMMAND.txt`

Line 37 reads:
```
Expected public spellF1: TBD (v8 with dist_ratio; bw=3 pending Final Gate)
```

The Final Gate was run in Round 8 (result: 0.2923). This line was
not updated at the time.

- [x] **Step 1: Update the stale TBD line**

  Change line 37 from:
  ```
  Expected public spellF1: TBD (v8 with dist_ratio; \
  bw=3 pending Final Gate)
  ```
  To:
  ```
  Public window spellF1 (reporting only): 0.2923 \
  (v8, all-3-games, BAR-ATH 7500-frame window)
  ```

  If T1 locked a new config and a new all-3-games model was
  produced, update this line once the new BAR-ATH score is
  available from Step 4 below.

  No other changes to `docs/FINAL_SUBMISSION_COMMAND.txt`.

### T2b — Reproducibility check

- [x] **Step 2: Verify submission CSV row count and label set** → OK: 199319 rows, labels={X:40462, D:123871, H:28702, A:6284}

  ```bash
  python3.11 -c "
  import csv, collections
  with open('evaluation/results/lstm_bw3_msf125_v8.csv') as f:
      rows = list(csv.DictReader(f))
  n = len(rows)
  labels = collections.Counter(r['label'] for r in rows)
  assert n == 199319, f'Expected 199319 rows, got {n}'
  invalid = set(labels) - {'H', 'A', 'D', 'X'}
  assert not invalid, f'Invalid labels: {invalid}'
  print(f'OK: {n} rows, labels={dict(labels)}')
  "
  ```

  If T1 locked a new config, run this check against
  `evaluation/results/lstm_bw3_msf125_v10.csv` instead.

- [x] **Step 3: Run full test suite** → 94/94 passed

  ```bash
  python3.11 -m pytest tests/ -v
  ```

  All tests must pass (94 tests from Round 8). Record count.

### T2c — BAR-ATH public window final gate (reporting only)

- [x] **Step 4: Score submission CSV against BAR-ATH public GT** → spellF1=0.2923 (v8, matches Round 8 result)

  Run against whichever CSV is the current submission candidate
  (v8 if T1 failed, v10 if T1 passed and new all-3-games was
  produced):

  ```bash
  python3.11 evaluation/run_evaluation.py \
    --gt evaluation/gt/BAR-ATH_possession_gt_public.csv \
    --results evaluation/results --no-detail
  ```

  Record spellF1 for the submission CSV.

  If T1 failed (submission = v8): this is a routine re-run of the
  Round 8 Final Gate (expected ~0.2923). If it differs by more
  than 0.01, investigate before proceeding.

  If T1 passed (submission = v10): record the new BAR-ATH score
  for context. Do NOT use this score to change the config.

## Final Gate — BAR-ATH Report

- [x] Run scorer against BAR-ATH public window for reporting only:

  ```bash
  python3.11 evaluation/run_evaluation.py \
    --gt evaluation/gt/BAR-ATH_possession_gt_public.csv \
    --results evaluation/results --no-detail
  ```

- [x] Record spellF1 here: **0.2923** (v8, lstm_bw3_msf125_v8)
- [x] Confirm: this number was NOT used to change the config.

---

## Deadline timeline

Today is 2026-10-26. Deadline is Nov 20, 2026 (25 days).

```
Week 1 (Oct 26 – Nov 1):
  T1 Step 1-2: add spread2d_ratio + test suite (Oct 26-27)
  T1 Step 3:   16-feat LOO retrain (overnight Oct 27-28)
  T1 Step 4-5: score + gate check (Oct 28)
  T1 Step 6:   Game 1 LOO cross-check if gate passes (Oct 28-29)
  T1 Step 7:   Revert if gate fails (Oct 29)
  T1 Step 8:   All-3-games retrain if T1 locked (Oct 29-30)

Week 2 (Nov 1–7):
  T2a: Fix stale TBD line in FINAL_SUBMISSION_COMMAND.txt (Nov 1)
  T2b: Reproducibility check + full test suite (Nov 1)
  T2c: BAR-ATH public window final gate (Nov 1)
  Config is confirmed locked no later than Nov 7.

Week 3 (Nov 8–14):
  Buffer for any unexpected findings.
  Hard cutoff for all experiments: Nov 14.

Week 4 (Nov 15–20):
  Final submission prep and submission.
  No new experiments.
```

---

## Review Focus

The Reviewer should pay extra attention to the following items
this round:

- **§11a coverage:** `spread2d_ratio` is the last §11a item that
  has never been tested. Confirm the probe ran, the gate was
  applied correctly, and the decision record is filled.
- **§2 data safety — no BAR-ATH GT in any train path:** T1
  retrains use only sample-game data (G1+G2 for LOO). Confirm no
  `--include-gt-tuning` flag and no BAR-ATH GT file appears in
  any training command.
- **`src/features.py` feature count consistency:** if T1 adds a
  16th feature, confirm `FEATURE_NAMES` (16 entries),
  `np.zeros((N, 16))`, module docstring header, docstring entry,
  and computation block are all updated atomically.
- **Revert completeness (T1 Step 7):** if gate fails, all five
  changes must be reverted. Confirm test suite passes after
  reverting.
- **LOO model provenance:** T1 models trained on G1+G2 only. LOO
  CSVs have 143,762 rows; full-game CSVs have 199,319 rows.
  Confirm row counts.
- **T2a stale line fix:** line 37 of
  `docs/FINAL_SUBMISSION_COMMAND.txt` must be updated from "TBD"
  to the known BAR-ATH score (0.2923 for v8, or the new score if
  T1 locked v10).
- **T2b row count + label set:** submission CSV must have exactly
  199,319 rows and labels in `{H, A, D, X}`.
- **T2c BAR-ATH is reporting-only:** the BAR-ATH score must not
  appear in any T1 selection table. It appears only in the Final
  Gate section, after config is locked.
- **run-id=v10 namespace:** all new models and CSVs carry `_v10`
  suffix. Confirm no collision with v8/v9 files.
- **No new banned paths:** no ensemble, no CRF, no attention, no
  tagging features, no BAR-ATH GT in training, no bw != 3.
- **§11e deadline gate:** if T1 fails, confirm
  `docs/FINAL_SUBMISSION_COMMAND.txt` still points to v8 model
  and the stale TBD line has been replaced. No further tuning
  after this round.

---

## Conclusions

**Round completed:** 2026-10-25
**Best spellF1 (Game 3 LOO):** 0.6703  (run-id: v8, unchanged)
**Prior best:** 0.6703  (run-id: v8, bw=3, msf=125, hs=256,
seed=42, 15-feat)
**Delta:** +0.0000

### What worked

- **T2 final submission verification passed cleanly.**
  `evaluation/results/lstm_bw3_msf125_v8.csv` confirmed at
  exactly 199,319 rows with labels in {H, A, D, X}. BAR-ATH
  public window spellF1 = 0.2923, matching the Round 8 Final
  Gate result exactly — reproducibility confirmed.
- **Stale TBD line resolved.** Line 37 of
  `docs/FINAL_SUBMISSION_COMMAND.txt` updated from the stale
  "TBD (v8 with dist_ratio; bw=3 pending Final Gate)" to the
  confirmed 0.2923 figure.
- **Full test suite healthy.** 94/94 tests passed after both T1
  revert and T2 verification, confirming codebase integrity.

### What did not work

- **spread2d_ratio (T1, 16-feat):** spellF1 = 0.6526 on Game 3
  LOO — delta = −0.0177 vs 0.6703 baseline. Gate threshold is
  +0.01; probe fails. Feature fully reverted; `src/features.py`
  is back to 15 columns.

### Key finding

The last untested §11a lever — `spread2d_ratio`, encoding the
relative 2-D compactness of the two teams as a ratio of their
mean pairwise spread across both x- and y-axes — degraded Game 3
LOO spellF1 by −0.0177 (0.6526 vs 0.6703 baseline). This mirrors
the pattern seen in Rounds 9 and 10: every probe that extends the
feature set beyond the 15-feat locked config has hurt performance.
The most likely explanation is that the LSTM, trained on two
games of data (~143 k frames), does not have enough capacity to
extract useful signal from higher-dimensional or noisier features.
The Round 8 breakthrough (+0.0706 from `dist_ratio`) appears to
have captured the dominant discriminative geometry; subsequent
features add noise rather than signal. With `spread2d_ratio`
exhausted, all §11a items have been tested. The 15-feat bw=3
msf=125 hs=256 seed=42 config is final.

### Recommended focus for next round

Ordered list, highest-leverage first:

1. **Submission (Week 4, Nov 15–20):** all levers are exhausted
   and the config is locked. The sole remaining action is
   confirming `docs/FINAL_SUBMISSION_COMMAND.txt` is correct and
   submitting `evaluation/results/lstm_bw3_msf125_v8.csv`
   (199,319 rows, spellF1=0.2923 on BAR-ATH public window).
2. **No new experiments:** the hard cutoff was Nov 14; Round 10
   has closed the last viable feature avenue. Any new probe risks
   destabilising a confirmed submission artifact.

### Open questions

- Does the consistent degradation of all 16-feat probes (Rounds
  9–10) indicate a training-data-size ceiling, or is the 15-feat
  feature set already optimal for this architecture?
- Will the BAR-ATH leaderboard score after final submission
  reflect the Game 3 LOO improvement trajectory (+0.0706 from
  dist_ratio), or does distribution shift between sample games
  and the BAR-ATH match cap the gain beyond 0.2923?
- No confusion breakdown by possession type (first-touch,
  set-piece, transition) was produced during any round. All
  experiments were evaluated on aggregate Game 3 LOO spellF1
  only. A post-submission analysis partitioning errors by
  possession-type tag (using the Smart Tagging XML) would
  reveal whether the remaining gap vs the leaderboard ceiling
  is concentrated in transitions, set-pieces, or long-spell
  possession — the most actionable finding for a future
  iteration.
