# Challenge Iteration 9 — Housekeeping + Velocity/Directional Feature Probe

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close four housekeeping findings left by the Round 8
Reviewer, then probe the highest-leverage remaining feature avenue
— velocity and directional signals — identified by the Analyst.
If any directional feature clears the +0.01 Game 3 LOO gate and
is confirmed by a Game 1 LOO cross-check, lock it as the new
submission config. Otherwise the 15-feat config from Round 8 remains
locked and Round 10 moves straight to final submission prep.

Levers in priority order:

1. **T0 — Reviewer housekeeping (four findings, zero experiments).**
   Non-blocking fixes that must be resolved before T1 begins: stale
   docstring in `src/features.py` (§10), ambiguous labels in
   `docs/FINAL_SUBMISSION_COMMAND.txt` (§9/§3), missing ablation note
   in `docs/CHALLENGE8.md` (§10), and an off-by-one in the
   `tests/test_features_v2.py` module docstring (§6).
2. **T1 — Directional/velocity feature probe (highest-leverage
   remaining avenue).** The Analyst flags relative proximity as
   underexplored; `dist_ratio` delivered +0.0706. Next candidates:
   ball-velocity direction relative to team centroids, and
   possession-momentum encoding. One or two solo probes, same LOO
   harness as T1 in Round 8.
3. **T2 (conditional) — hs=512 re-check at 15 features.** hs=512
   beat hs=256 by +0.0036 at 14-feat (Round 6). At 15-feat this
   is a fast, low-risk sweep. Run only if T1 does not consume all
   remaining experiment budget before Nov 14.

**Architecture:** T0 (housekeeping) runs first and is independent of
the experiments. T1 and T2 are sequential; T2 is conditional on T1
passing its gate or failing fast before the deadline.

**Tech Stack:** Python 3.11, PyTorch, numpy; existing `src/` pipeline.
**Spec:** `CHALLENGE.md`, `EVALUATION.md`, `docs/CHALLENGE8.md`

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
- `n_A_within_3m` is **banned** — Round 7 T1: −0.0215 spellF1.
- bw=1 and bw=2 are **banned** — Round 8 T2: both worse than bw=3.
- Use `--run-id v9` for all new models this round.
- Deadline: **Nov 20, 2026** — 25 days remain. At most one
  additional feature probe (T1) and one architecture re-check (T2)
  before final submission prep begins Nov 15.
- **Locked baseline:** bw=3, msf=125, seed=42, hs=256, 15-feat,
  temporal split; spellF1=0.6703 on Game 3 LOO
  (`models/lstm_bw3_v8_15feat.pt`).

---

## T0 — Reviewer housekeeping (no experiments)

These four items are non-blocking documentation/comment fixes that
must be completed before any experiment in T1. None of them alters
any trained model or CSV.

### Finding 1 (§10) — Stale "pending gate" note in `src/features.py`

Lines 31–33 of `src/features.py` still read:

```
Feature 14 (dist_ratio) was added in iteration 8 as a solo probe.
It has never been tested in isolation (without n_A_within_3m). Pending
gate result from T1 Step 5; revert to 14 features if gate fails.
```

The gate passed (both gates cleared +0.01); the config is locked at
15 features. This note is factually incorrect.

- [x] **Step 1: Update the stale docstring in `src/features.py`**

  Replace lines 31–33 with:

  ```
  Feature 14 (dist_ratio) was added in iteration 8 (T1 solo probe).
  Both the Game 3 LOO gate (+0.0808) and the Game 1 LOO cross-check
  (+0.0557) passed. The 15-feat config is permanently locked.
  ```

  No other changes to `src/features.py`.

  Confirm the file still imports cleanly:
  ```bash
  python3.11 -c "from src.features import FEATURE_NAMES; \
    print(len(FEATURE_NAMES), FEATURE_NAMES[-1])"
  ```
  Expected: `15 dist_ratio`

### Finding 2 (§9/§3) — Ambiguous "public window" label in
`docs/FINAL_SUBMISSION_COMMAND.txt`

Lines 4–5 annotate the LOO scores with `(public window, …)`, which
a reader could mistake for the BAR-ATH public GT window — a banned
tuning signal.

- [x] **Step 2: Fix labels in `docs/FINAL_SUBMISSION_COMMAND.txt`**

  Change:
  ```
  Game 3 LOO spellF1 proxy: 0.6703 (public window, loo_g3_bw3_msf125_v8_15feat)
  Game 1 LOO cross-check: 0.7022 (public window, loo_g1_bw3_msf125_v8_15feat)
  ```
  To:
  ```
  Game 3 LOO spellF1 proxy: 0.6703 (Game 3 LOO, sample-game GT)
  Game 1 LOO cross-check:   0.7022 (Game 1 LOO, sample-game GT)
  ```

  No other changes to this file.

### Finding 3 (§10) — Missing ablation note in `docs/CHALLENGE8.md`

The Conclusions section attributes the +0.0706 gain to `dist_ratio`
but does not explain the mechanism — specifically why a ratio of two
absolute distances outperforms the raw pair.

- [x] **Step 3: Add brief ablation note to `docs/CHALLENGE8.md`**

  In the "Key finding" subsection, after the existing explanation
  of the ratio formula, append:

  ```
  Ablation note: the raw pair (dist_nearest_H, dist_nearest_A)
  expresses proximity as two independent scalars; the model must
  learn their relationship internally. dist_ratio encodes that
  relationship directly as a single continuous signal — when
  dist_ratio < 1 the ball is geometrically nearer to a Home player,
  when > 1 nearer to an Away player. Encoding the comparison
  explicitly removes a learned division from the LSTM's compute
  budget and provides a cleaner gradient for boundary frames where
  the two raw distances are nearly equal.
  ```

  Hard-wrap all added prose at ~80 chars per `.local/CLAUDE.md`.

### Finding 4 (§6) — Off-by-one in `tests/test_features_v2.py`
module docstring

Line 5 of `tests/test_features_v2.py` reads:
```
Feature 15 (dist_ratio) was added as a solo probe in iteration 8.
```
`dist_ratio` occupies 0-indexed column 14 (the 15th column). The
label "Feature 15" is misleading — it would be read as 1-indexed.

- [x] **Step 4: Fix the module docstring in
  `tests/test_features_v2.py`**

  Change line 5 from:
  ```
  Feature 15 (dist_ratio) was added as a solo probe in iteration 8.
  ```
  To:
  ```
  Feature at index 14 (dist_ratio, the 15th column) was added as
  a solo probe in iteration 8.
  ```

  No other changes to this file.

### T0 sign-off

- [x] **Step 5: Run full test suite — must still pass** → 94/94 passed

  ```bash
  python3.11 -m pytest tests/ -v
  ```

  All tests must pass (94 tests from Round 8). Record any failures
  before proceeding to T1.

---

## T1 — Directional/velocity feature probe (16th feature)

### Background

`dist_ratio` produced the largest single-round gain in the project
(+0.0706 Game 3 LOO). The Analyst identifies relative proximity as
underexplored and recommends directional/momentum features as the
next highest-leverage avenue. Round 8's Conclusions also list
centroid-velocity direction and possession-momentum encoding as
open questions.

Two candidate features are evaluated here as solo probes, in
priority order:

**Candidate A — `centroid_ratio`**
`centroid_dist_H / (centroid_dist_A + ε)` — the same ratio idea
applied to centroid distances rather than nearest-player distances.
The centroid captures team shape (where the team is concentrated)
rather than the nearest individual. This is a direct generalization
of `dist_ratio` that the LSTM has never seen. Expected to be
positive if team shape matters beyond nearest-player proximity.

**Candidate B — `ball_dir_H`**
Dot product of the normalised ball velocity vector with the
normalised vector from the ball to the Home centroid. When the ball
is moving toward the Home centroid this is positive (near 1.0);
away from it, negative. Set to 0.0 when ball speed < 0.1 m/s or
ball is invisible. This encodes momentum direction, not just
instantaneous proximity.

**Probe order:** Run Candidate A first (simpler, same family as
`dist_ratio`). If Candidate A clears the gate, lock it and do not
run Candidate B (avoid feature accumulation). If Candidate A fails,
run Candidate B as the fallback. If both fail, lock the Round 8
config and proceed to T2.

**Config:** bw=3, msf=125, seed=42, hs=256, temporal split.
**Baseline:** 0.6703 (Game 3 LOO, `models/lstm_bw3_v8_15feat.pt`,
15-feat locked config).
**Gate:** spellF1 improvement >= +0.01 on Game 3 LOO, then
cross-check on Game 1 LOO before locking.
**GT file:** `games/Sample_Game_3/gt/possession_gt.csv` (full match).

**Files:**
- Modify: `src/features.py` — add feature 15, run, revert if gate
  fails.
- Produces: `evaluation/results/loo_g3_bw3_msf125_v9_16feat_A.csv`
  (Candidate A), and optionally
  `evaluation/results/loo_g3_bw3_msf125_v9_16feat_B.csv`
  (Candidate B, only if A fails).
- Models: `models/lstm_bw3_v9_16feat_A.pt` (and optionally
  `models/lstm_bw3_v9_16feat_B.pt`).

### T1a — Candidate A: centroid_ratio

- [x] **Step 1: Add `centroid_ratio` to `src/features.py`**

  After feature 14 (`dist_ratio`), append:

  ```python
  # Feature 15: centroid_ratio — ball nearer H centroid (<1) or A (>1)
  _EPS = 1e-3
  centroid_ratio = (centroid_dist_H / (centroid_dist_A + _EPS)
                    ).astype(np.float32)
  centroid_ratio[~ball_visible] = 1.0   # neutral sentinel
  feats[:, 15] = centroid_ratio
  ```

  Update `np.zeros((N, 15), ...)` → `np.zeros((N, 16), ...)`.

  Update `FEATURE_NAMES` to 16 entries by appending
  `"centroid_ratio"`.

  Update the module docstring header from
  `"Feature vector (15 columns):"` to
  `"Feature vector (16 columns):"` and add the entry:
  ```
  15 centroid_ratio  centroid_dist_H / (centroid_dist_A + 1e-3);
                     <1 ball nearer H centroid, >1 nearer A,
                     1.0 sentinel when ball missing
  ```

  Confirm before running:
  ```bash
  python3.11 -c "from src.features import FEATURE_NAMES; \
    print(len(FEATURE_NAMES), FEATURE_NAMES[-1])"
  ```
  Expected: `16 centroid_ratio`

- [x] **Step 2: Run test suite — must still pass** → 94/94 passed (updated 2 hardcoded n_feat=15 assertions to 16)

  ```bash
  python3.11 -m pytest tests/ -v
  ```

  If any test hardcodes `n_feat=15` or `shape=(N, 15)` for
  `extract_features()`, update the assertion to 16 (do NOT remove
  the test). Record any failures before continuing.

- [x] **Step 3: Retrain LOO model with 16 features (Candidate A)**

  ```bash
  python3.11 src/pipeline.py --train 1 2 --predict 3 \
    --model lstm --bw 3 --msf 125 --seed 42 \
    --out evaluation/results/loo_g3_bw3_msf125_v9_16feat_A.csv \
    --save-model models/lstm_bw3_v9_16feat_A.pt
  ```

- [x] **Step 4: Score on Game 3 LOO** → spellF1=0.6211

  ```bash
  python3.11 evaluation/run_evaluation.py \
    --gt evaluation/gt/Sample_Game_3_possession_gt_public.csv \
    --results evaluation/results --no-detail
  ```

  Record spellF1 for `loo_g3_bw3_msf125_v9_16feat_A`.

  | Features | n_feat | Game 3 LOO spellF1 | delta vs 15-feat |
  |---|---|---|---|
  | 15-feat baseline (v8 locked) | 15 | 0.6703 | — |
  | +centroid_ratio (v9, Candidate A) | 16 | TBD | TBD |

- [x] **Step 5: Gate check (Candidate A)** → delta=-0.0492 (FAILS gate); reverting to 15-feat; running Candidate B

  | delta | Action |
  |---|---|
  | >= +0.01 | Run Game 1 LOO cross-check (Step 6) |
  | +0.00 to +0.009 | Revert to 15-feat; run Candidate B (T1b) |
  | < 0.00 | Revert to 15-feat; run Candidate B (T1b) |

- [x] **Step 6 (conditional): Game 1 LOO cross-check (Candidate A)** → SKIPPED (gate failed at Step 5)

  Run only if Step 5 delta >= +0.01.

  ```bash
  python3.11 src/pipeline.py --train 2 3 --predict 1 \
    --model lstm --bw 3 --msf 125 --seed 42 \
    --out evaluation/results/loo_g1_bw3_msf125_v9_16feat_A.csv \
    --save-model models/lstm_bw3_v9_16feat_A_g1loo.pt
  python3.11 evaluation/run_evaluation.py \
    --gt evaluation/gt/Sample_Game_1_possession_gt_public.csv \
    --results evaluation/results --no-detail
  ```

  Game 1 LOO baseline = 0.7022 (v8 15-feat, Round 8 T1 Step 6).

  If Game 1 LOO delta >= +0.01 vs 0.7022: lock 16-feat (Candidate A)
  as new config. Update `docs/FINAL_SUBMISSION_COMMAND.txt`.
  Do NOT run Candidate B.

  If Game 1 LOO delta < +0.01: revert to 15-feat; run Candidate B
  (T1b) as fallback.

- [x] **Step 7 (conditional): Revert `src/features.py` if Candidate A
  gate failed or cross-check failed** → Reverted; 94/94 tests pass

  If Step 5 delta < +0.01, or if Step 6 Game 1 LOO delta < +0.01:

  - Remove `"centroid_ratio"` from `FEATURE_NAMES`.
  - Restore `np.zeros((N, 15), ...)`.
  - Restore module docstring header to `"Feature vector (15 columns):"`.
  - Remove the feature 15 entry from the docstring.
  - Remove the feature 15 computation block.
  - Confirm: `python3.11 -c "from src.features import \
    FEATURE_NAMES; assert len(FEATURE_NAMES)==15; print('ok')"`
  - Re-run test suite: `python3.11 -m pytest tests/ -v`

### T1b — Candidate B: ball_dir_H (fallback, run only if A fails)

- [x] **Step 1 (conditional): Add `ball_dir_H` to `src/features.py`**

  Run only if T1a failed (Step 7 above was executed).

  After feature 14 (`dist_ratio`), append:

  ```python
  # Feature 15: ball_dir_H — dot(norm(ball_vel), norm(ball→H_centroid))
  # +1 if ball moving toward H centroid, -1 if away, 0 when ball still
  # or invisible.
  ball_to_h = home_centroid - ball_m   # (N, 2) in metres
  bth_norm = np.linalg.norm(ball_to_h, axis=1, keepdims=True)
  bth_unit = np.where(bth_norm > 0,
                      ball_to_h / np.maximum(bth_norm, 1e-6),
                      0.0)
  vel_norm = np.linalg.norm(ball_diff * FPS, axis=1, keepdims=True)
  vel_unit = np.where(vel_norm > 0.1,
                      (ball_diff * FPS) / np.maximum(vel_norm, 1e-6),
                      0.0)
  ball_dir_H = (bth_unit * vel_unit).sum(axis=1).astype(np.float32)
  ball_dir_H[~ball_visible] = 0.0
  feats[:, 15] = ball_dir_H
  ```

  Update `np.zeros((N, 15), ...)` → `np.zeros((N, 16), ...)`.

  Update `FEATURE_NAMES` to 16 entries by appending `"ball_dir_H"`.

  Update module docstring header to `"Feature vector (16 columns):"`.
  Add entry:
  ```
  15 ball_dir_H      dot(norm(ball_vel), norm(ball→H_centroid));
                     +1 ball moving toward H centroid, −1 away,
                     0.0 when ball speed < 0.1 m/s or invisible
  ```

  Confirm:
  ```bash
  python3.11 -c "from src.features import FEATURE_NAMES; \
    print(len(FEATURE_NAMES), FEATURE_NAMES[-1])"
  ```
  Expected: `16 ball_dir_H`

- [x] **Step 2 (conditional): Run test suite** → 94/94 passed

  ```bash
  python3.11 -m pytest tests/ -v
  ```

- [x] **Step 3 (conditional): Retrain LOO model (Candidate B)**

  ```bash
  python3.11 src/pipeline.py --train 1 2 --predict 3 \
    --model lstm --bw 3 --msf 125 --seed 42 \
    --out evaluation/results/loo_g3_bw3_msf125_v9_16feat_B.csv \
    --save-model models/lstm_bw3_v9_16feat_B.pt
  ```

- [x] **Step 4 (conditional): Score on Game 3 LOO** → spellF1=0.5895

  ```bash
  python3.11 evaluation/run_evaluation.py \
    --gt evaluation/gt/Sample_Game_3_possession_gt_public.csv \
    --results evaluation/results --no-detail
  ```

  Record spellF1 for `loo_g3_bw3_msf125_v9_16feat_B`.

  | Features | n_feat | Game 3 LOO spellF1 | delta vs 15-feat |
  |---|---|---|---|
  | 15-feat baseline (v8 locked) | 15 | 0.6703 | — |
  | +ball_dir_H (v9, Candidate B) | 16 | TBD | TBD |

- [x] **Step 5 (conditional): Gate check (Candidate B)** → delta=-0.0808 (FAILS gate); reverting to 15-feat; locking Round 8 config

  | delta | Action |
  |---|---|
  | >= +0.01 | Run Game 1 LOO cross-check (Step 6) |
  | < +0.01 | Revert to 15-feat (Step 7); lock Round 8 config |

- [x] **Step 6 (conditional): Game 1 LOO cross-check (Candidate B)** → SKIPPED (gate failed at Step 5)

  Run only if Step 5 delta >= +0.01.

  ```bash
  python3.11 src/pipeline.py --train 2 3 --predict 1 \
    --model lstm --bw 3 --msf 125 --seed 42 \
    --out evaluation/results/loo_g1_bw3_msf125_v9_16feat_B.csv \
    --save-model models/lstm_bw3_v9_16feat_B_g1loo.pt
  python3.11 evaluation/run_evaluation.py \
    --gt evaluation/gt/Sample_Game_1_possession_gt_public.csv \
    --results evaluation/results --no-detail
  ```

  If Game 1 LOO delta >= +0.01 vs 0.7022: lock 16-feat (Candidate B).
  Update `docs/FINAL_SUBMISSION_COMMAND.txt`.

  If Game 1 LOO delta < +0.01: revert to 15-feat (Step 7).

- [x] **Step 7 (conditional): Revert `src/features.py` if Candidate B
  gate failed** → Reverted; 94/94 tests pass

  Same five-point revert checklist as T1a Step 7 above, targeting
  `"ball_dir_H"` and column 15. Re-run test suite and confirm pass.

### T1 decision record

Fill this table after T1a and (if applicable) T1b:

| Candidate | Feature | G3 LOO spellF1 | delta | G1 LOO spellF1 | Locked? |
|---|---|---|---|---|---|
| A | centroid_ratio | 0.6211 | −0.0492 | not run | No |
| B | ball_dir_H | 0.5895 | −0.0808 | not run | No |

---

## T2 — hs=512 re-check at 15 features (conditional)

### Background

hs=512 beat hs=256 by +0.0036 on Game 3 LOO at 14-feat (Round 6
T2). The 15-feat config has only been tested at hs=256. The expected
gain is small (~+0.003–0.005) but the risk is low: the model already
exists at hs=128 and hs=512 for 14-feat; a single retrain suffices.

**Run T2 only if both of the following are true:**
1. T1 did not lock a 16-feat config (i.e. T1 failed its gate).
2. At least 7 days remain before Nov 14 (the hard experiment cutoff).

If T1 does lock a new config, skip T2 entirely and move to submission
prep — no further feature accumulation.

**Config:** bw=3, msf=125, seed=42, hs=512, temporal split, 15-feat.
**Baseline:** 0.6703 (Game 3 LOO, hs=256, v8 locked).
**Gate:** spellF1 improvement >= +0.01 on Game 3 LOO.

- [x] **Step 1 (conditional): Retrain LOO model with hs=512, 15-feat**

  ```bash
  python3.11 src/pipeline.py --train 1 2 --predict 3 \
    --model lstm --bw 3 --msf 125 --seed 42 \
    --hidden-size 512 \
    --out evaluation/results/loo_g3_bw3_msf125_hs512_v9.csv \
    --save-model models/lstm_bw3_hs512_v9.pt
  ```

- [x] **Step 2 (conditional): Score on Game 3 LOO** → spellF1=0.6000

  ```bash
  python3.11 evaluation/run_evaluation.py \
    --gt evaluation/gt/Sample_Game_3_possession_gt_public.csv \
    --results evaluation/results --no-detail
  ```

  Record spellF1 for `loo_g3_bw3_msf125_hs512_v9`.

  | Config | hs | G3 LOO spellF1 | delta vs hs=256 |
  |---|---|---|---|
  | v8 locked (hs=256) | 256 | 0.6703 | — |
  | v9 hs=512, 15-feat | 512 | 0.6000 | −0.0703 |

- [x] **Step 3 (conditional): Gate check** → delta=−0.0703 (FAILS gate); keeping hs=256; Round 8 config unchanged

  If delta >= +0.01: run Game 1 LOO cross-check before locking.

  ```bash
  python3.11 src/pipeline.py --train 2 3 --predict 1 \
    --model lstm --bw 3 --msf 125 --seed 42 \
    --hidden-size 512 \
    --out evaluation/results/loo_g1_bw3_msf125_hs512_v9.csv \
    --save-model models/lstm_bw3_hs512_v9_g1loo.pt
  python3.11 evaluation/run_evaluation.py \
    --gt evaluation/gt/Sample_Game_1_possession_gt_public.csv \
    --results evaluation/results --no-detail
  ```

  If Game 1 LOO delta >= +0.01 vs 0.7022: lock hs=512 as new config.
  Update `docs/FINAL_SUBMISSION_COMMAND.txt`.

  If delta < +0.01 on Game 3, or Game 1 LOO cross-check fails:
  keep hs=256. Record finding; no config change.

- [x] **Step 4 (conditional): Record T2 decision**

  | Outcome | Config after T2 |
  |---|---|
  | hs=512 passes both gates | hs=512, 15-feat locked |
  | hs=512 fails | hs=256, 15-feat (v8) unchanged |

  **Decision: hs=512 FAILS (−0.0703). Config remains v8 (bw=3, msf=125,
  hs=256, seed=42, 15-feat, spellF1=0.6703 on Game 3 LOO).**

---

## Deadline timeline

Today is 2026-10-26. Deadline is Nov 20, 2026 (25 days).

```
Week 1 (Oct 26 – Nov 1):
  T0 housekeeping (all 4 findings): Oct 26 (same day)
  T1a Step 1-2: add centroid_ratio + test suite: Oct 26-27
  T1a Step 3:   16-feat LOO retrain (overnight Oct 27-28)
  T1a Step 4-5: score + gate check: Oct 28
  T1a Step 6:   Game 1 LOO cross-check if gate passes (Oct 28-29)
  T1a Step 7:   Revert if gate fails, run T1b (Oct 29-30)

Week 2 (Nov 1–7):
  T1b (if needed): Candidate B retrain + scoring (Nov 1-3)
  T2 (if applicable): hs=512 retrain + scoring (Nov 3-5)
  Config locked no later than Nov 7.

Week 3 (Nov 8–14):
  All-3-games retrain with locked config (overnight Nov 8-9).
  Update FINAL_SUBMISSION_COMMAND.txt if config changed.
  Buffer for any unexpected findings.
  Hard cutoff for new experiments: Nov 14.

Week 4 (Nov 15–20):
  Final submission prep and submission.
  No new experiments.
```

---

## Review Focus

The Reviewer should pay extra attention to the following items this
round:

- **§10 T0 completeness:** all four housekeeping findings from Round 8
  must be closed before any experiment runs. Confirm each of the
  four targeted files was edited correctly and that the test suite
  still passes 94 tests after T0.
- **§2 data safety — no BAR-ATH GT in any train path:** T1 retrains
  use only sample-game data (G1+G2 for Game 3 LOO). Confirm no
  `--include-gt-tuning` flag and no BAR-ATH GT file appears in any
  command.
- **`src/features.py` feature count consistency:** if T1 adds a 16th
  feature, confirm `FEATURE_NAMES` (16 entries), `np.zeros((N, 16))`,
  module docstring header, docstring entry, and computation block
  are all updated atomically. Partial updates cause silent shape
  mismatches.
- **Revert completeness (T1 Steps 7):** if a gate fails, all five
  changes must be reverted. Confirm the test suite passes after
  reverting.
- **LOO model provenance:** T1 and T2 models trained on G1+G2 only.
  LOO CSVs have 143,762 rows; full-game CSVs have 199,319 rows.
  Confirm row counts.
- **T2 conditional gate:** T2 must NOT run if T1 locked a new config.
  Adding features and widening hidden size in the same round risks
  confounding the source of any gain.
- **§9 BAR-ATH is reporting-only:** the BAR-ATH public window score
  must not appear in any T1 or T2 selection table. It appears only
  in the Final Gate section of docs/CHALLENGE8.md (already run).
- **run-id=v9 namespace:** all new models and CSVs carry `_v9`
  suffix. Confirm no collision with v8 files.
- **Deadline gate:** if both T1 and T2 fail their gates, the locked
  config remains v8 (bw=3, msf=125, hs=256, seed=42, 15-feat, 0.6703
  on Game 3 LOO). Confirm `docs/FINAL_SUBMISSION_COMMAND.txt` still
  points to the v8 model and no further tuning is planned for Week 4.
- **§11d ensemble closure is permanent:** no new ensemble paths should
  appear in any command.

---

## Conclusions

**Round completed:** 2026-10-25
**Best spellF1 (Game 3 LOO):** 0.6703  (run-id: v8, lstm_bw3_v8_15feat)
**Prior best:** 0.6703
**Delta:** +0.0000

### What worked

- **T0 housekeeping (all four findings closed).** Stale "pending gate"
  docstring in `src/features.py` corrected; ambiguous "public window"
  labels in `docs/FINAL_SUBMISSION_COMMAND.txt` clarified to "Game 3
  LOO, sample-game GT"; ablation note explaining the dist_ratio
  mechanism appended to `docs/CHALLENGE8.md` Conclusions; off-by-one
  label in `tests/test_features_v2.py` module docstring fixed. All
  four fixes are non-experiment changes; 94/94 tests passed after each.
- **Clean revert discipline.** Both T1 candidates and T2 were reverted
  completely after gate failures; the test suite confirmed 94/94 after
  each revert, maintaining codebase integrity.

### What did not work

- **centroid_ratio (T1, Candidate A, 16-feat):** spellF1 = 0.6211
  — delta = −0.0492 vs 0.6703 baseline. Gate FAILS.
- **ball_dir_H (T1, Candidate B, 16-feat):** spellF1 = 0.5895
  — delta = −0.0808. Gate FAILS. This is the weakest result tested
  since Round 5; the momentum-direction signal adds noise rather than
  signal at this feature count.
- **hs=512 re-check at 15-feat (T2):** spellF1 = 0.6000
  — delta = −0.0703. hs=512 under-performs hs=256 at both 14-feat
  (Round 6: +0.0036, now reversed) and 15-feat. The hidden-size
  dimension is now effectively closed.

### Key finding

All three Round 9 probes degraded the Game 3 LOO spellF1 relative
to the v8 locked baseline. `centroid_ratio` (−0.0492) applied the
same ratio idea as `dist_ratio` to centroid distances; the centroid
may carry less discriminative signal than nearest-player proximity
at this temporal resolution. `ball_dir_H` (−0.0808) is the worst
result since Round 5, suggesting that ball-velocity direction toward
the Home centroid adds noise rather than signal — likely because at
25 fps the ball-velocity vector is highly noisy on dead-ball and
slow-play frames, which are the majority of the match. hs=512
(−0.0703) indicates the model is not capacity-limited at 15 features;
hs=256 is the correct operating point. The 15-feat bw=3 msf=125
hs=256 seed=42 config from Round 8 remains the locked submission
config with spellF1 = 0.6703.

### Recommended focus for next round

Ordered list, highest-leverage first:

1. **Submission finalisation (Week 4, Nov 15–20)** — all levers are
   exhausted; the locked v8 config is final. Priority is confirming
   `docs/FINAL_SUBMISSION_COMMAND.txt` is correct and the all-3-games
   submission CSV (`lstm_bw3_msf125_v8.csv`, 199,319 rows) is ready.
2. **BAR-ATH public window reporting** — the v8 submission CSV scored
   0.2923 (reporting-only); no action needed, but note for context
   when interpreting the leaderboard result.
3. **No new experiments** — the hard cutoff is Nov 14; Round 9
   has closed the last viable feature and architecture avenues.

### Open questions

- Does the centroid-distance ratio carry less signal than
  nearest-player proximity because centroid location is dominated
  by off-ball defenders rather than pressing players?
- Is the ball-velocity noise problem (dead-ball frames) the root
  cause of `ball_dir_H` failing, or is direction simply not
  predictive of possession at the 25-fps sample rate?
- Will the final BAR-ATH leaderboard score (after submission of
  the all-3-games v8 model) reflect the Game 3 LOO improvement
  (+0.0706 from dist_ratio), or does distribution shift between
  sample games and BAR-ATH cap the gain?
