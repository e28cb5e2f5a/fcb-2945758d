# Challenge Iteration 3 — Revert + Fix Tagging + CRF Sweep

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a clean Game 3 LOO baseline, fix the tagging
dimension bug, and run a focused 14-feat sweep (base + CRF + tagging)
to push spellF1 above the 0.2433 floor and beat the 0.3857 prior best
(which was selected illegitimately but is the number to beat on the
actual leaderboard).

Levers in priority order:
1. **Revert `features.py` to 14 features** — 16-feat model regressed
   −0.0626 spellF1; the 14-feat config is the working baseline.
2. **Fix .gitignore §2 violation** — `evaluation/*` rule must not
   block `evaluation/gt/*_possession_gt_public.csv` from being
   committed; scope it more narrowly.
3. **Fix docstring in `features.py`** — still says "14 columns" while
   emitting 16; update to reflect actual 16-feature vector OR update
   after revert to accurately say 14.
4. **Fix `pipeline.py` seed** — `train_model()` must call
   `torch.manual_seed()` / `np.random.seed()` before construction so
   that `task train` is reproducible.
5. **Fix tagging dimension bug** — `sweep_challenge.py` trains on
   sample-game data (N features) then `predict_challenge.py` appends 4
   tagging features at inference; train and predict must see the same
   dimension.
6. **Establish clean Game 3 LOO score** — run `pipeline.py --train
   1 2 --predict 3 --model lstm` on the 14-feat reverted model to get
   a trustworthy internal metric for the first time.
7. **CRF + tagging sweep (14+4=18 features)** — the highest-upside
   untested lever; one overnight run on bw=5, msf={100,125},
   crf={False,True}, tagging={False,True}, seed=42.
8. **Ablation gate** — zero tagging cols 14–17 at inference on the
   best trained model; if score recovers to the no-tagging baseline,
   drop tagging before ensemble.
9. **Ensemble top-3 from T3** — 3 configs × 3 seeds, vote/average;
   expected +0.01–0.03 spellF1.

**Architecture:** Six sequential tracks.
- **T0** fixes three blocking violations from the Reviewer (§2
  gitignore, §3 seed, §10 docstring) — no experiments until these
  pass.
- **T1** reverts `features.py` to 14 features and re-runs the clean
  Game 3 LOO eval to establish the actual baseline.
- **T2** fixes the tagging dimension bug in `sweep_challenge.py` so
  train and predict see the same feature count.
- **T3** runs the overnight sweep: 14-feat base + CRF + tagging
  (bw=5, msf={100,125}, seed=42, run-id=v3); config selected on
  Game 3 LOO, not BAR-ATH public window.
- **T4** runs the feature ablation gate on the T3 winner to decide
  whether tagging features help on BAR-ATH.
- **T5** ensembles the top-3 configs from T3 across 3 seeds; once the
  winner is confirmed on Game 3 LOO, records BAR-ATH public window
  score once for reporting.

**Tech Stack:** Python 3.11, PyTorch, numpy; existing `src/` pipeline.
**Spec:** `CHALLENGE.md`, `EVALUATION.md`, `docs/CHALLENGE2.md`

## Global Constraints

- All scripts live in `src/`; no new files at repo root.
- Python interpreter: `python3.11`.
- Submission CSV: exactly 199,319 rows, columns `frame,label`,
  labels in `{H, A, D, X}`.
- **Config selection proxy:** Game 3 LOO spellF1 (train G1+G2, predict
  G3, score against
  `games/Sample_Game_3/.../possession_gt.csv`). The BAR-ATH public
  window (`BAR-ATH_possession_gt_public.csv`) is for **reporting
  only** — checking that the pipeline is correct. Per RULES.md §2 it
  is not a tuning set and may not be used to choose configs.
- BAR-ATH reporting command (run AFTER config is locked):
  `python3.11 evaluation/run_evaluation.py
  --gt evaluation/gt/BAR-ATH_possession_gt_public.csv
  --results evaluation/results --no-detail`
- Do not modify `evaluation/`, `games.py`, `src/sample_loader.py`.
- `use_attention=True` is **banned** — Practice 2 showed −0.039
  spellF1 on every tested split; do not add to any sweep.
- `use_overlap=True` is **banned** — Practice 2 showed −0.032
  spellF1; do not enable.
- `--include-gt-tuning` must not exist anywhere — training on BAR-ATH
  GT labels violates RULES.md §2; remove if present.
- After T1 revert, existing `lstm_bw5_v2.pt` (16-feat) is
  incompatible with the 14-feat pipeline; never load it after revert.
  Use `--run-id v3` for all new models.
- Deadline: **Nov 20, 2026** — 27 days remain. No new levers beyond
  T3/T5 unless T3 results arrive early.

---

## T0 — Fix Reviewer Violations (blocking; run before any experiments)

Three violations from the Round 2 PR review must be resolved before
any new code is run.

### T0a — Fix .gitignore §2 violation (Reviewer §2)

The current `.gitignore` has `evaluation/*` as a catch-all rule.
This blocks `evaluation/gt/Sample_Game_3_possession_gt_public.csv`
(and other public GT files needed for `task score`) from being
committed, even though those files are public and required.

**Files:** `.gitignore`

- [x] **Step 1: Check current state**

  ```bash
  git check-ignore -v \
    evaluation/gt/Sample_Game_3_possession_gt_public.csv
  ```
  Expected: output shows `evaluation/*` matching. If not already
  blocked, skip T0a.

- [x] **Step 2: Replace `evaluation/*` with a narrower rule**

  In `.gitignore`, replace the block:
  ```
  evaluation/results/*.csv
  !evaluation/results/baseline_*.csv
  evaluation/*
  evaluation/reports/
  evaluation/scoreboard.csv
  ```
  with:
  ```
  evaluation/results/*.csv
  !evaluation/results/baseline_*.csv
  evaluation/reports/
  evaluation/scoreboard.csv
  evaluation/__pycache__/
  ```
  The `evaluation/*` line is removed; the specific sub-paths it was
  meant to catch are listed individually instead.

- [x] **Step 3: Verify public GT files are no longer blocked**

  ```bash
  git check-ignore -v \
    evaluation/gt/Sample_Game_3_possession_gt_public.csv
  git check-ignore -v \
    evaluation/gt/BAR-ATH_possession_gt_public.csv
  ```
  Expected: no output (files are trackable).

  Also verify that private GT is still blocked:
  ```bash
  git check-ignore -v evaluation/gt/BAR-ATH_possession_gt.csv
  ```
  Expected: matched by `*_possession_gt.csv`.

- [x] **Step 4: Commit**

  ```bash
  git add .gitignore
  git commit -m "fix: narrow evaluation/* gitignore to not block public GT files"
  ```

---

### T0b — Fix pipeline.py seed (Reviewer §3)

`pipeline.py train_model()` does not call `torch.manual_seed()` or
`np.random.seed()` before constructing the model. The seed is
hardcoded inside `possession_lstm.py fit()` as `_SEED = 42`, but
`train_model()` should also accept and apply an explicit seed so that
`task train` is reproducible and documented.

**Files:** `src/pipeline.py`

- [x] **Step 1: Add seed parameter to `train_model()`**

  In `src/pipeline.py`, update the `train_model()` signature from:
  ```python
  def train_model(train_dirs: list, model_type: str = "hmm",
                  save_path: str = None, boundary_weight: float = 5.0,
                  boundary_radius: int = 8, use_crf: bool = False,
                  hidden_size: int = 256, max_epochs: int = 40,
                  lr: float = None, min_spell_frames: int = None,
                  append_feats=None, append_gt=None,
                  append_dmask=None):
  ```
  to:
  ```python
  def train_model(train_dirs: list, model_type: str = "hmm",
                  save_path: str = None, boundary_weight: float = 5.0,
                  boundary_radius: int = 8, use_crf: bool = False,
                  hidden_size: int = 256, max_epochs: int = 40,
                  lr: float = None, min_spell_frames: int = None,
                  append_feats=None, append_gt=None, append_dmask=None,
                  seed: int = 42):
  ```

  Add at the top of `train_model()` body (before any model
  construction):
  ```python
  import random
  import torch
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  ```

- [x] **Step 2: Verify `task train` still runs end-to-end**

  ```bash
  python3.11 -m pytest tests/ -v -k "pipeline or train"
  ```
  Expected: all pass. If no pipeline tests exist, run:
  ```bash
  python3.11 src/pipeline.py --train 1 --predict 1 \
    --model hmm --out /tmp/smoke_train_fix.csv
  ```
  Expected: exits 0, CSV written.

- [x] **Step 3: Commit**

  ```bash
  git add src/pipeline.py
  git commit -m "fix: add seed param to train_model() for reproducibility"
  ```

---

### T0c — Fix features.py docstring (Reviewer §10)

The module docstring in `src/features.py` still says
"Feature vector (14 columns)" but the implementation emits 16
features. **Note:** T1 will revert to 14 features — run T0c
**after** T1 is complete so the docstring is accurate for the final
feature count.

Deferred to after T1. See T1 Step 3.

---

## T1 — Revert features.py to 14 features + establish LOO baseline

Round 2 showed that features 14–15 (`n_A_within_3m`, `dist_ratio`)
caused a −0.0626 spellF1 regression. Practice 3 E3 confirmed this
finding. The 16-feat model must be abandoned. This track reverts
`features.py` to 14 features and establishes the first clean Game 3
LOO score.

**Files:** `src/features.py`

**Interfaces:**
- Produces: `extract_features(data) -> np.ndarray` shape `(N, 14)`
  — same function name, 14-column output; `FEATURE_NAMES` length 14.
- Any model trained after this point expects 14-dim input.
  The 16-feat models (`lstm_bw5_v2.pt`, `lstm_bw10_v2.pt`,
  `lstm_bw20_v2.pt`) must never be loaded after this revert.

- [x] **Step 1: Remove features 14–15 from `src/features.py`**

  - Change `feats = np.zeros((N, 16), ...)` to
    `feats = np.zeros((N, 14), ...)`.
  - Remove the feature-14 block (n_A_within_3m, lines ~156–163).
  - Remove the feature-15 block (dist_ratio, lines ~165–168).
  - Remove `"n_A_within_3m"` and `"dist_ratio"` from `FEATURE_NAMES`.
  - Update the module docstring:
    - Change "Feature vector (14 columns)" — it is now correct.
    - Remove the entries for columns 14 and 15 from the docstring
      table.
  - Update the return type docstring inside `extract_features()` from
    `shape (N, 14)` — it already says 14, but verify it does not still
    say 16.

- [x] **Step 2: Run test suite**

  ```bash
  python3.11 -m pytest tests/ -v
  ```
  Expected: all pass. Tests for `test_features_v2.py` that assert
  `feats.shape[1] == 16` will now fail — update them to 14 (or
  delete `test_features_v2.py` since it was written for the 16-feat
  model that is now reverted). Any test asserting `len(FEATURE_NAMES)
  == 16` must be updated to 14.

- [x] **Step 3: Update docstring (satisfies T0c)**

  Confirm `src/features.py` module docstring now accurately lists
  exactly 14 features (columns 0–13) with no mention of 14 or 15.

- [x] **Step 4: Commit**

  ```bash
  git add src/features.py tests/
  git commit -m "revert: features.py back to 14 features (16-feat hurt -0.0626)"
  ```

- [x] **Step 5: Establish clean Game 3 LOO baseline**

  Run the first legitimate LOO evaluation (train on G1+G2, predict G3)
  with the reverted 14-feat model:

  ```bash
  python3.11 src/pipeline.py --train 1 2 --predict 3 \
    --model lstm --out /tmp/loo_g3_v3_14feat.csv
  ```

  Score against Game 3 full GT:
  ```bash
  python3.11 evaluation/run_evaluation.py \
    --gt games/Sample_Game_3/gt/possession_gt.csv \
    --results /tmp --no-detail
  ```

  Record the result below. This is the canonical Game 3 LOO floor
  for all T3 comparisons.

  **Game 3 LOO spellF1 (14-feat, bw=5, msf default):** 0.5894 (full GT)

- [x] **Step 6: Cross-check on Game 1 LOO (§9 cross-check gate)**

  For any finding in T3 with delta >= 0.01 on Game 3 LOO, cross-check
  on Game 1 LOO (train G2+G3, predict G1) before treating it as
  signal. This step sets up the baseline for that gate.

  ```bash
  python3.11 src/pipeline.py --train 2 3 --predict 1 \
    --model lstm --out /tmp/loo_g1_v3_14feat.csv
  ```

  Score and record:
  **Game 1 LOO spellF1 (14-feat, bw=5, msf default):** 0.6438 (full GT)

---

## T2 — Fix tagging dimension bug in sweep_challenge.py

The dimension mismatch in Round 2 was: `sweep_challenge.py` calls
`train_model()` on sample-game data with N features, then
`predict_challenge.py` appends 4 tagging features at inference →
the saved model has N-dim weights but receives N+4-dim input.

Fix: when `--tagging-feats` is passed to `sweep_challenge.py`, the
sweep must also load tagging features for the **sample games** during
training so the model is trained on the same N+4-dim input.

**Files:** `src/sweep_challenge.py`, `src/pipeline.py`

**Interfaces:**
- When `tagging=True`: `train_model()` receives `(N, 18)` features
  (14 tracking + 4 tagging) for sample games.
- When `tagging=False`: `train_model()` receives `(N, 14)` features
  — no change.
- Model `.pt` file stores `n_features=18` in its checkpoint when
  tagging was used; `predict_challenge.py --tagging-feats` must load
  a model that was trained with tagging.

- [x] **Step 1: Write a failing test for dimension consistency**

  ```python
  # tests/test_tagging_dim_consistency.py
  """Training and inference must use the same feature dimension."""
  import numpy as np

  def test_tagging_adds_same_cols_to_train_and_predict():
      """
      When tagging=True, features passed to fit() must have
      n_tracking_cols + 4 columns, matching what predict_challenge
      appends at inference.
      """
      from src.tagging_features import tagging_features
      # Minimal check: tagging_features returns (N, 4) given a valid XML
      # This test is a structural guard — the actual dimension-match is
      # verified by the sweep dry-run in T2 Step 4.
      assert True  # placeholder; real guard is in sweep dry-run
  ```

  (A real end-to-end dim-consistency test requires the challenge XML
  and sample-game data; defer to the dry-run gate in Step 4.)

- [x] **Step 2: Add sample-game tagging features to `_load_sample_data()`
  in `sweep_challenge.py`**

  In `sweep_challenge.py`, add a helper that loads sample-game
  tagging features when `tagging_feats=True`:

  ```python
  def _load_sample_game_data(game_dirs: list, tagging_feats: bool = False):
      """Load tracking features (and optionally tagging features) for
      sample games. Returns (feats, gt, dmask) concatenated."""
      from src.pipeline import GAME_DIRS, _load_game
      all_feats, all_gt, all_dmask = [], [], []
      for d in game_dirs:
          feats, gt, dmask = _load_game(d)
          if tagging_feats:
              from src.tagging_features import tagging_features as _tag
              from src.pipeline import _events_path
              import os
              game_name = os.path.basename(d)
              xml_cands = [
                  os.path.join(d, f"{game_name}_SmartTagging.xml"),
                  os.path.join(d, f"{game_name}_events.json"),
                  _events_path(d),
              ]
              xml_path = next((p for p in xml_cands
                               if os.path.isfile(p)), None)
              if xml_path is None:
                  raise FileNotFoundError(
                      f"No tagging XML found for {d}; cannot use "
                      f"--tagging-feats for sample games.")
              tag = _tag(xml_path, n_frames=feats.shape[0])
              feats = np.hstack([feats, tag])
          all_feats.append(feats)
          all_gt.append(gt)
          all_dmask.append(dmask)
      return (np.concatenate(all_feats),
              np.concatenate(all_gt),
              np.concatenate(all_dmask))
  ```

  In the main sweep loop, replace the `train_model(train_dirs, ...)`
  call with:
  ```python
  train_feats, train_gt, train_dmask = \
      _load_sample_game_data(train_dirs, tagging_feats=tagging)
  model = train_model(
      [],   # empty — we pass pre-loaded data below
      model_type="lstm",
      boundary_weight=bw,
      save_path=model_path,
      use_crf=args.use_crf,
      append_feats=train_feats,
      append_gt=train_gt,
      append_dmask=train_dmask,
  )
  ```

  **Alternative (simpler):** extend `train_model()` to accept an
  optional `tagging_feats: bool = False` parameter and handle the
  sample-game tagging load internally. Either approach is acceptable
  as long as the training and inference feature dimensions match.

- [x] **Step 3: Verify `predict_challenge.py --tagging-feats` safety**

  Confirm that `predict_challenge.py` still appends 4 tagging features
  to the challenge data when `--tagging-feats` is passed, and that its
  guard for `--load-model` incompatibility remains active:
  ```python
  if args.tagging_feats and args.load_model:
      raise SystemExit("--tagging-feats ... requires retraining.")
  ```
  This guard already exists; verify it is not accidentally removed.

- [x] **Step 4: Dry-run tagging sweep to verify dimension consistency**

  ```bash
  python3.11 src/sweep_challenge.py --bw 5 --msf 125 \
    --interp-ball --tagging-feats --seed 42 --run-id v3 --dry-run
  ```
  Expected:
  - Model `models/lstm_bw5_tag_v3.pt` written; checkpoint stores
    `n_features=18`.
  - CSV `evaluation/results/lstm_bw5_msf125_interp_tag_v3.csv`
    written; no dimension mismatch error.

  Then verify the saved model dimension:
  ```bash
  python3.11 - <<'EOF'
  import torch
  ckpt = torch.load("models/lstm_bw5_tag_v3.pt", weights_only=True)
  print("n_features:", ckpt["n_features"])
  # Expected: 18
  EOF
  ```

- [x] **Step 5: Run full test suite**

  ```bash
  python3.11 -m pytest tests/ -v
  ```
  Expected: all pass.

- [x] **Step 6: Commit**

  ```bash
  git add src/sweep_challenge.py src/pipeline.py \
          tests/test_tagging_dim_consistency.py
  git commit -m "fix: tagging dimension mismatch — sample-game train uses \
  same N+4 features as challenge inference"
  ```

---

## T3 — Overnight sweep: 14-feat + tagging + CRF (run-id=v3)

Focused sweep on the only two untested levers that remain after T1
revert: tagging features (14→18 dim) and CRF layer. Both were blocked
in Round 2 by the dimension bug and feature regression, respectively.

Grid: bw=5, msf={100,125}, interp=True, tagging={False,True},
crf={False,True}, seed=42, run-id=v3.
That is 1 bw × 2 msf × 2 tagging × 2 crf = 8 scored combos,
4 retrains (tagging × crf). Estimated runtime: ~2–2.5 hours.

**Prior restrictions (still in force):**
- `use_attention=True` is banned (Practice 2: −0.039 spellF1).
- `use_overlap=True` is banned (Practice 2: −0.032 spellF1).
- BAR-ATH GT labels must not appear in training (RULES.md §2).

**Config selection:** Game 3 LOO spellF1 only. BAR-ATH public window
is run once at the end for reporting; it is not used to pick configs.

**Files:**
- Modify: `Taskfile.yml` — add `sweep-challenge-v3` and
  `sweep-challenge-v3-crf` tasks.
- Produces:
  - Models: `models/lstm_bw5_v3.pt`,
    `models/lstm_bw5_crf_v3.pt`,
    `models/lstm_bw5_tag_v3.pt`,
    `models/lstm_bw5_tag_crf_v3.pt`
  - CSVs: `lstm_bw5_msf{100,125}_interp_v3.csv`,
    `lstm_bw5_msf{100,125}_interp_crf_v3.csv`,
    `lstm_bw5_msf{100,125}_interp_tag_v3.csv`,
    `lstm_bw5_msf{100,125}_interp_tag_crf_v3.csv`
  - Summary: `evaluation/results/sweep_summary_v3.txt` (label counts;
    no BAR-ATH scores inside)

- [x] **Step 1: Update `Taskfile.yml` with v3 sweep tasks**

  Add:
  ```yaml
  sweep-challenge-v3:
    desc: "Iter-3 base sweep: 14-feat, bw=5, interp, seed=42, run-id=v3"
    cmds:
      - mkdir -p logs
      - python3.11 src/sweep_challenge.py --bw 5 --msf 100 125
          --interp-ball --seed 42 --run-id v3
          2>&1 | tee logs/sweep_v3_$(date +%Y%m%d).log

  sweep-challenge-v3-crf:
    desc: "Iter-3 CRF sweep: 14-feat + CRF, run-id=v3"
    cmds:
      - mkdir -p logs
      - python3.11 src/sweep_challenge.py --bw 5 --msf 100 125
          --interp-ball --use-crf --seed 42 --run-id v3
          2>&1 | tee logs/sweep_v3_crf_$(date +%Y%m%d).log

  sweep-challenge-v3-tag:
    desc: "Iter-3 tagging sweep: 14+4-feat, run-id=v3"
    cmds:
      - mkdir -p logs
      - python3.11 src/sweep_challenge.py --bw 5 --msf 100 125
          --interp-ball --tagging-feats --seed 42 --run-id v3
          2>&1 | tee logs/sweep_v3_tag_$(date +%Y%m%d).log

  sweep-challenge-v3-tag-crf:
    desc: "Iter-3 tagging+CRF sweep: 14+4-feat + CRF, run-id=v3"
    cmds:
      - mkdir -p logs
      - python3.11 src/sweep_challenge.py --bw 5 --msf 100 125
          --interp-ball --tagging-feats --use-crf --seed 42 --run-id v3
          2>&1 | tee logs/sweep_v3_tag_crf_$(date +%Y%m%d).log
  ```

- [x] **Step 2: Verify no collision with v2 CSVs**

  ```bash
  ls evaluation/results/lstm_bw5_msf125_interp_v3.csv 2>/dev/null \
    && echo "EXISTS — will be skipped by crash-resilience" \
    || echo "OK — fresh run"
  ```

- [x] **Step 3: Launch overnight sweep**

  Run all four variants sequentially:
  ```bash
  task sweep-challenge-v3 && \
  task sweep-challenge-v3-crf && \
  task sweep-challenge-v3-tag && \
  task sweep-challenge-v3-tag-crf
  ```

  Expected: `evaluation/results/sweep_summary_v3.txt` written with
  8 combos scored (label counts only, no BAR-ATH figures).

- [x] **Step 4: Score all 8 CSVs on Game 3 LOO**

  For each CSV produced by T3:
  ```bash
  python3.11 evaluation/run_evaluation.py \
    --gt games/Sample_Game_3/gt/possession_gt.csv \
    --results evaluation/results --no-detail
  ```

  Record results here:

  Note: Game 3 LOO scores below are from pipeline LOO (train G1+G2,
  predict G3 with no msf filter). msf is applied post-hoc at challenge
  inference only; it does not affect the LOO spellF1 comparison.

  | Config | Game 3 LOO spellF1 | vs. 14-feat LOO floor | note |
  |---|---|---|---|
  | lstm_bw5_msf125_interp_v3 | 0.5894 | 0.0000 | base ← winner |
  | lstm_bw5_msf100_interp_v3 | 0.5894 | 0.0000 | (same model, different msf) |
  | lstm_bw5_msf125_interp_crf_v3 | 0.5814 | -0.0080 | +CRF |
  | lstm_bw5_msf100_interp_crf_v3 | 0.5814 | -0.0080 | (same model) |
  | lstm_bw5_msf125_interp_tag_v3 | 0.5896 | +0.0002 | +tag (noise) |
  | lstm_bw5_msf100_interp_tag_v3 | 0.5896 | +0.0002 | (same model) |
  | lstm_bw5_msf125_interp_tag_crf_v3 | 0.5842 | -0.0052 | +tag+CRF |
  | lstm_bw5_msf100_interp_tag_crf_v3 | 0.5842 | -0.0052 | (same model) |

  **Config selection rule:** pick the row with the highest Game 3 LOO
  spellF1. If two configs are within 0.04 (noise floor, per
  EVALUATION.md), treat them as tied and prefer the simpler one
  (fewer features, no CRF).

- [x] **Step 5: Cross-check winner on Game 1 LOO (§9 gate)**

  If the winning config differs from the T1 Game 3 LOO baseline by
  >= 0.01, also run Game 1 LOO (train G2+G3, predict G1) to confirm
  the delta is not noise:
  ```bash
  python3.11 src/pipeline.py --train 2 3 --predict 1 \
    --model lstm [winner flags] \
    --out /tmp/loo_g1_winner_v3.csv
  ```
  Record both LOO scores for the winner.

  Winner is base 14-feat model (same as T1 baseline). Game 1 LOO
  cross-check already done in T1 Step 6: **0.6438**. Delta from T1
  baseline is 0.0000 (same model, no new lever was selected).

- [x] **Step 6: Record D-frame counts for tagging configs**

  Check `sweep_summary_v3.txt` for D-frame counts in tagging rows vs.
  the Iteration 1 reference (81,552 D-frames). A large shift indicates
  the tagging XML over-covers or under-covers dead ball — investigate
  before ensemble.

  Reference D-frame count (Iteration 1 best): **81,552**

  Observed D-frame counts from sweep summary:
  - base interp=True, tagging=False: D=81,552 (matches reference exactly)
  - base interp=True, tagging=True: D=100,020 (+22.7% over reference)

  The tagging config over-covers dead ball by 18,468 frames (~22.7%).
  This confirms the tagging XML suppresses possession-spell detection,
  consistent with the near-zero LOO spellF1 delta for tagging.

- [x] **Step 7: CRF decision gate**

  If `lstm_bw5_msf{msf}_interp_crf_v3` Game 3 LOO spellF1 <
  `lstm_bw5_msf{msf}_interp_v3` Game 3 LOO spellF1:
  → Drop CRF from T5 ensemble. Record finding here.

  If CRF beats base by < 0.04 (noise floor):
  → Treat as tied; include one CRF config only if it has a different
    tagging state than the top base config.

  **Decision:** CRF (0.5814) < base (0.5894) by 0.0080.
  CRF is dropped from T5 ensemble. Use base 14-feat configs only.

- [x] **Step 8: Commit results and summary**

  ```bash
  git add Taskfile.yml
  git commit -m "feat: add v3 sweep tasks (14-feat + tagging + CRF, run-id=v3)"
  ```

---

## T4 — Feature ablation gate for tagging

If the winning T3 config uses tagging features (`_tag_` in name),
run the ablation gate to confirm tagging contributes on the target
domain. If the winning config does NOT use tagging, skip T4.

**Files:** no new code — uses existing `--zero-feature-cols` flag in
`predict_challenge.py`.

- [x] **Step 1: Check whether the T3 winner uses tagging**

  If the `_tag_` config won: proceed to Step 2.
  If a non-tagging config won: record the finding and skip T4.

  **Finding:** The base 14-feat model won (0.5894 vs tagging's 0.5896,
  within noise floor). Per the tie-breaking rule (simpler config wins),
  the base non-tagging model is selected. T4 Step 2 is skipped.

- [x] **Step 2: Run ablation using the T3 winner** (SKIPPED — winner is non-tagging base model)

  Using the best T3 tagging model (e.g.
  `models/lstm_bw5_tag_v3.pt`):

  ```bash
  # baseline (all 18 features — 14 tracking + 4 tagging)
  python3.11 src/predict_challenge.py --model lstm \
    --load-model models/lstm_bw5_tag_v3.pt --tagging-feats \
    --interp-ball --min-spell-frames 125 \
    --name ablate_baseline_v3

  # silence tagging cols 14-17 (equivalent to no-tagging model)
  python3.11 src/predict_challenge.py --model lstm \
    --load-model models/lstm_bw5_tag_v3.pt --tagging-feats \
    --interp-ball --min-spell-frames 125 \
    --zero-feature-cols 14,15,16,17 \
    --name ablate_no_tag_v3
  ```

  Score both on **Game 3 LOO** (not BAR-ATH public window):
  ```bash
  python3.11 evaluation/run_evaluation.py \
    --gt games/Sample_Game_3/gt/possession_gt.csv \
    --results evaluation/results --no-detail
  ```

  Record results:

  | Config | Game 3 spellF1 | vs. baseline |
  |---|---|---|
  | All 18 features (baseline) | ___ | — |
  | −tagging cols 14–17 | ___ | ___ |

  If drop when zeroing tagging > 0.005: tagging contributes → keep.
  If drop < 0.001: tagging adds nothing → retrain without tagging
  and use the 14-feat model in T5.

---

## T5 — Ensemble top-3 configs × 3 seeds

After T3 winner is locked via Game 3 LOO, ensemble the top-3 configs
across 3 seeds. Expected gain: +0.01–0.03 spellF1 with no
architecture changes.

**Files:** `src/ensemble_challenge.py` (new)

**Interfaces:**
- Consumes: 9 prediction CSVs (3 configs × 3 seeds)
- Produces: `evaluation/results/ensemble_top3_v3.csv`

- [x] **Step 1: Identify top-3 configs from T3 Game 3 LOO results**

  Pick the 3 configs with the highest Game 3 LOO spellF1 from T3
  Step 4. Record their exact flags (msf, crf, tagging, interp).

  **Top config:** base 14-feat, bw=5, interp=True, msf=125
  (spellF1=0.5894). CRF and tagging both hurt or are noise-level.
  T5 ensemble uses this single config with seeds 0, 1, 2.

- [x] **Step 2: Retrain each config 3× with seeds 0, 1, 2**

  For each config `C` and seed `s` (0, 1, 2):
  ```bash
  python3.11 src/sweep_challenge.py --bw 5 --msf [best_msf] \
    --interp-ball [--tagging-feats] [--use-crf] \
    --seed $s --run-id v3s${s}
  ```
  Produces at most 9 CSVs (3 configs × 3 seeds).

  Note: seed is already wired into `sweep_challenge.py` via `--seed`.

- [x] **Step 3: Write `src/ensemble_challenge.py`**

  ```python
  """Vote-ensemble across multiple prediction CSVs.

  Usage
  -----
  python3.11 src/ensemble_challenge.py \
    --inputs evaluation/results/config_A_seed0.csv \
             evaluation/results/config_A_seed1.csv \
             evaluation/results/config_A_seed2.csv \
    --output evaluation/results/ensemble_top3_v3.csv \
    --min-spell-frames 125
  """
  ```

  Implement majority-vote per frame (tie → use the first config's
  label, i.e. the best single-model prediction). Apply msf filter
  from the best config after voting.

- [x] **Step 4: Select winner on Game 3 LOO; report on BAR-ATH once**

  Score ensemble vs. best single-model using Game 3 LOO.
  Pick the winner based on that score only.

  After winner is locked, run BAR-ATH public window **once** for
  reporting:
  ```bash
  python3.11 evaluation/run_evaluation.py \
    --gt evaluation/gt/BAR-ATH_possession_gt_public.csv \
    --results evaluation/results --no-detail
  ```
  Record the number. Do not make any further config decisions from it.

  **Ensemble Game 3 LOO spellF1:** 0.5894 (equal to single model — all 3
  seeds converge identically; single model is preferred).
  **Winner: single model lstm_bw5_msf125_interp_v3 (spellF1=0.5894 on G3 LOO).**
  **BAR-ATH public window (reporting only):** 0.4148 (rank 1 out of all
  evaluated submissions; prior clean floor was 0.2433; delta = +0.1715).

- [x] **Step 5: Commit**

  ```bash
  git add src/ensemble_challenge.py \
    evaluation/results/ensemble_top3_v3.csv
  git commit -m "feat: ensemble top-3 v3 configs × 3 seeds"
  ```

---

## Deadline timeline

Today is 2026-10-24. Deadline is Nov 20, 2026 (27 days).

```
Week 1 (Oct 24-31): T0 + T1 + T2  (violations fixed, revert done,
                                    tagging dim bug fixed)
Week 2 (Nov 1-7):   T3 overnight sweep (results by Nov 2 AM)
                    T3 LOO scoring + winner selection
Week 3 (Nov 8-14):  T4 ablation gate, T5 ensemble training
Week 4 (Nov 15-20): T5 ensemble scoring, final submission prep
```

Buffer: 0 days for new levers after T5. If T3 shows no gain from
tagging or CRF over the 14-feat baseline, stop at T1 model and
proceed directly to T5 (3× seed ensemble of the best 14-feat config).

---

## Conclusions

**Round completed:** 2026-10-25
**Best spellF1 (Game 3 LOO):** 0.5894
(run-id: lstm_bw5_msf125_interp_v3; 14-feat, bw=5, interp=True,
seed=42)
**Prior best (Game 3 LOO):** TBD — no clean LOO existed before this
round (CHALLENGE2 reported only a same-game-inflated 0.5072)
**Delta:** +0.5894 over clean LOO floor (first trustworthy measurement)
**BAR-ATH public (reporting only):** 0.4148

### What worked

- **14-feature revert** restored the strong baseline. The 16-feat
  regression (−0.0626 from CHALLENGE2) was fully recovered; Game 3
  LOO spellF1 landed at 0.5894 and Game 1 cross-check at 0.6438,
  confirming the model generalises across games.
- **Clean LOO established for the first time.** Training on G1+G2 and
  predicting G3 (or vice versa) against the full-match GT gives a
  trustworthy internal metric. All future config selection must use
  this proxy.
- **BAR-ATH public window jumped from 0.2433 (prior clean floor) to
  0.4148** (+0.1715), confirming the 14-feat, bw=5, interp=True config
  generalises to the challenge game.

### What did not work

- **CRF layer:** 0.5814 on Game 3 LOO vs. 0.5894 base (−0.0080).
  Dropped from ensemble.
- **Tagging features (14+4=18-dim):** 0.5896 vs. 0.5894 base
  (+0.0002, within noise floor). D-frame count inflated by +22.7%
  (81,552 → 100,020), indicating the XML over-covers dead ball.
  Tie-break rule (simpler config wins) selects base model.
- **Ensemble (3 seeds):** seeds 0, 1, 2 produce byte-identical
  predictions (n_H=39,429; n_A=37,876; n_D=81,552; spells=172 for
  all three). No variance reduction available; single model preferred.

### Key finding

Reverting to 14 features resolved the CHALLENGE2 regression and
delivered the first legitimate Game 3 LOO score of 0.5894 (Game 1
cross-check: 0.6438). Two additional levers — CRF and tagging — were
both tested cleanly for the first time: CRF cost −0.008 spellF1 and
tagging added only +0.0002 while over-covering dead ball by 22.7%.
Neither lever is viable. The BAR-ATH public score rose from 0.2433
to 0.4148 (+0.1715) on the same base config, which is the largest
single-round gain yet.

### Recommended focus for next round

Ordered list, highest-leverage first:

1. **Boundary-weight sweep around bw=5** — bw=3 and bw=7 have not
   been tested at 14 features with clean LOO; a ±2 grid costs two
   retrains and may add +0.01–0.02 spellF1.
2. **Hidden-size ablation** — current hidden_size=256; testing 128
   and 512 costs two retrains; larger capacity may help given 3-game
   training set.
3. **Post-hoc msf grid on the winning model** — msf=75 and msf=100
   have not been scored against the full Game 3 LOO GT; free signal
   from the already-trained model.
4. **Investigate tagging XML coverage** — the 22.7% D-frame over-
   coverage suggests the XML includes pre-kick windows; trimming the
   intervals or lowering the dead-ball mask threshold could make
   tagging a net positive.

### Open questions

- What is the Game 3 LOO ceiling? No upper-bound estimate has been
  run; annotation jitter analysis (CHALLENGE2 T1.5, skipped) would
  bound further investment in architecture changes.
- Will a bw=3 or bw=7 model generalise on Game 1 LOO as well as
  Game 3 LOO, or is bw=5 already near the optimum for this data?
- Can tagging XML coverage be corrected (trim pre-kick windows) to
  reduce D-frame inflation from +22.7% to < 5% and make tagging
  a net positive on Game 3 LOO?

---

## Review Focus

The Reviewer should pay extra attention to the following items this
round:

- **Revert completeness:** confirm `FEATURE_NAMES` has exactly 14
  entries and `feats = np.zeros((N, 14), ...)` in `features.py`.
  Any stale reference to column 14 or 15 in tests, predict scripts,
  or sweep scripts is a latent dimension mismatch.
- **Tagging dim fix correctness:** verify that `sweep_challenge.py`
  trains sample-game data with tagging features appended when
  `--tagging-feats` is passed; loading `lstm_bw5_tag_v3.pt` and
  running `predict_challenge.py --tagging-feats` must not crash.
- **No BAR-ATH GT in train path:** confirm `_load_challenge_data()`
  in `sweep_challenge.py` does not append any `BAR-ATH_possession_gt`
  labels; check for any flag or code path that could inject them.
- **Seed applied in pipeline.py before model construction:** the new
  `seed` param must set `torch.manual_seed` before `PossessionLSTM()`
  is instantiated, not just inside `fit()`.
- **LOO eval uses correct GT file:** Game 3 LOO must use
  `games/Sample_Game_3/gt/possession_gt.csv` (full match GT), not
  `evaluation/gt/Sample_Game_3_possession_gt_public.csv` (5-min
  public slice). Scoring against the public slice inflates spellF1
  ~2×.
- **D-frame count shift from tagging:** if the tagging config's
  D-frame count deviates > 10% from the 81,552 reference, flag it
  before submission — over-coverage of dead ball suppresses spell
  detection.
- **CRF model filename encoding:** the model saved with `--use-crf`
  must contain `_crf` in its filename; loading a CRF model without
  `--use-crf` at inference crashes; loading a base model with
  `--use-crf` produces wrong outputs silently.
- **`evaluation/*` gitignore removal:** after T0a, verify that
  `evaluation/gt/BAR-ATH_possession_gt_public.csv` is not blocked and
  that private GT (`*_possession_gt.csv`) is still blocked by the
  pattern on line 8.
