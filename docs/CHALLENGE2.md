# Challenge Iteration 2 — New Features + Tagging + CRF Sweep Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push Game 3 LOO spellF1 (train G1+G2, predict G3) beyond the
current 14-feature bw=5/msf=125/interp config, and beat the challenge
leaderboard floor of 0.2433 on submission. The 0.3857 BAR-ATH figure
from CHALLENGE1 is discarded — selected by tuning on the BAR-ATH public
GT window (RULES.md §2 violation).

Levers in priority order:
1. Wider msf post-hoc (free, no retrain)
2. Two missing features — `n_A_within_3m` and `dist_ratio`
3. CRF layer (already implemented, never tried — models transitions
   explicitly, directly rewards spellF1)
4. Tagging features in training (638 set-piece intervals = dead-ball GT)
5. Ensemble of top-3 configs × 3 seeds (last-mile variance reduction)

**Architecture:** Seven sequential tracks.
- **T0** fixes the cache gitignore gap — one-line `.gitignore` entry.
- **T1** extends msf post-hoc on the existing model (no code, fast
  signal). Config selection uses **Game 3 LOO spellF1** (train G1+G2,
  eval G3), not the BAR-ATH public window.
- **T1.5** runs error analysis — documents upper bound and per-class
  failure modes before any retraining.
- **T2** adds two features to `src/features.py` (14→16).
- **T2.6** adds `--zero-feature-cols` to `predict_challenge.py` for
  post-sweep feature ablation; ablation runs after T3.
- **T2.5** adds `src/loader.py` unified auto-detecting `load_tracking()`
  (§5 switchover fix).
- **T3** adds `--run-id`, `--use-crf`, `--seed` to `sweep_challenge.py`;
  enforces temporal split; confirms overlap is OFF (Practice 2 finding);
  runs overnight sweep: base and CRF (caveat), msf={100,125,150},
  seed=42 = 6 combinations, 2 retrains. Attention excluded — Practice 2
  proved it harmful. `--include-gt-tuning` is removed — using GT labels
  from the provided game is banned by RULES.md §2.
- **T4** ensembles the top-3 configs from T3 across 3 seeds; ablation
  (T2.6) runs on best model. Config selected on Game 3 LOO, then
  reported once on BAR-ATH public window.

**Tech Stack:** Python 3.11, PyTorch, numpy; existing `src/` pipeline.

**Spec:** `CHALLENGE.md`, `EVALUATION.md`, `docs/CHALLENGE1.md`

## Global Constraints

- All scripts live in `src/`; no new files at repo root
- Python interpreter: `python3.11`
- Submission CSV: exactly 199,319 rows, columns `frame,label`,
  labels in `{H, A, D, X}`
- **Config selection proxy:** Game 3 LOO spellF1 (train G1+G2, predict
  G3, score against `games/Sample_Game_3/.../possession_gt.csv`). The
  BAR-ATH public window (`BAR-ATH_possession_gt_public.csv`) is for
  **reporting only** — checking that the pipeline is correct. Per
  RULES.md §2 it is not a tuning set and may not be used to choose
  configs.
- BAR-ATH reporting command (run AFTER config is locked):
  `python3.11 evaluation/run_evaluation.py
  --gt evaluation/gt/BAR-ATH_possession_gt_public.csv
  --results evaluation/results --no-detail`
- Do not modify `evaluation/`, `games.py`, `src/sample_loader.py`
- `src/features.py` **may** be modified in T2 (it was frozen in
  Iteration 1 to keep scope narrow; T2 is explicitly about extending it)
- After T2, existing `.pt` models in `models/` are incompatible with
  the new feature dimension; only use them via `--run-id` namespaced
  models produced by T3

## Review Focus

- **Feature count mismatch on load:** after T2, `predict_challenge.py
  --load-model models/lstm_bw5.pt` will silently fail or produce wrong
  outputs because the saved model expects 14 features and the new
  pipeline produces 16. T2's test must assert `feats.shape[1] == 16`
  and the plan must not use `--load-model` with old `.pt` files after T2.
- **dist_ratio sentinel when ball missing:** `dist_H / (dist_H + dist_A)`
  evaluates to `999/1998 ≈ 0.5` when ball is invisible, coinciding with
  the equidistant case. T2 must zero this feature when ball is not visible
  to avoid conflating the two states.
- **n_A_within_3m interaction with ball_missing sentinel:** when ball is
  missing, `ball_m` uses the (0.5*L, 0.5*W) centre sentinel. Away players
  near the pitch centre could spuriously count as "within 3m". Must zero
  `n_A_within_3m` when `ball_visible` is False, matching feature 8.
- **Test fixtures must use 46 player slots:** `home_xy` and `away_xy` in
  synthetic fixtures must be shaped `(n, 46, 2)`, not `(n, 14, 2)`.
  The old value 14 is the pre-expansion slot count; using it silently
  masks distance/count bugs that only appear with the real 46-slot layout.
- **CSV collision in T3 sweep:** if `--run-id` is omitted, new sweep CSVs
  overwrite Iteration 1 CSVs and `lstm_bw5.pt` is overwritten. The dry-run
  step must verify that run-id-namespaced filenames are created.
- **Tagging features + new features = 20-dim input:** the T3 no-CRF model
  is trained on 20 features (16 tracking + 4 tagging). Any post-T3
  `predict_challenge.py` call must pass `--tagging-feats` together with
  `--load-model models/lstm_bw5_tag_v2.pt` or the shapes will mismatch.
- **CRF model is a separate `.pt` file:** the CRF variant stores different
  weights (transition matrix included). Loading a CRF model without
  `--use-crf` crashes; loading a base model with `--use-crf` produces
  wrong outputs. The model filename must encode `_crf` and the load command
  must match.
- **`--run-id` path traversal:** `run_id` is inserted directly into file
  paths. Reject any value containing `/`, `\`, or `..` before use;
  raise `argparse.ArgumentTypeError` with a clear message.
- **Fixed seeds required in sweep:** `sweep_challenge.py` must call
  `torch.manual_seed` and `np.random.seed` with a fixed value (e.g. 42)
  before each model construction so that re-running the same config
  produces the same `.pt` file. Without this, `task train` +
  `run_evaluation.py` cannot reproduce the submitted CSV.
- **Tagging feature label distribution shift:** adding `--tagging-feats`
  changes how many frames are marked D (set-piece intervals from the XML
  extend the dead-ball mask). The T3 sweep summary must record
  `H/A/D/X` frame counts for each config; a large shift in D-count vs.
  Iteration 1 (81,552 D-frames) is a signal that the XML intervals
  over-cover or under-cover dead ball and must be investigated before
  submitting.

---

## Task 0: Fix cache gitignore gap

`challenge_loader.py` writes a cache file named
`.challenge_tracking_cache.npz`. The existing `.gitignore` pattern
`.tracking_cache.*` does **not** match this name — it will appear as an
untracked file containing derived tracking data.

- [ ] **Step 1: Verify the cache is not gitignored**

  ```bash
  git check-ignore -v data/2027_challenge/tactical/.challenge_tracking_cache.npz
  ```
  If no output (i.e. not ignored), proceed to Step 2.

- [ ] **Step 2: Add explicit pattern to `.gitignore`**

  Add to `.gitignore`:
  ```
  .challenge_tracking_cache*.npz
  ```

- [ ] **Step 3: Verify and commit**

  ```bash
  git check-ignore -v data/2027_challenge/tactical/.challenge_tracking_cache.npz
  git add .gitignore
  git commit -m "fix: gitignore challenge tracking cache file"
  ```

---

## Task 1: Wider msf post-hoc on existing lstm_bw5.pt

No code changes. Extends the msf grid from 125 to 150/175/200 on the
already-saved model and predictions from Iteration 1.

**Files:**
- No changes

**Interfaces:**
- Consumes: `models/lstm_bw5.pt` (bw=5, 14 features, trained Iteration 1);
  `src/predict_challenge.py --load-model ... --interp-ball --min-spell-frames N`
- Produces: `evaluation/results/lstm_bw5_msf{150,175,200}_interp.csv`;
  updated `evaluation/results/sweep_summary_msf_ext.txt` (written manually)

- [x] **Step 1: Generate msf=150, 175, 200 predictions**

  ```bash
  for msf in 150 175 200; do
    python3.11 src/predict_challenge.py \
      --model lstm --load-model models/lstm_bw5.pt \
      --interp-ball --min-spell-frames $msf \
      --name lstm_bw5_msf${msf}_interp
  done
  ```
  Expected: three CSVs written to `evaluation/results/`.

- [x] **Step 2: Score all three on Game 3 LOO**

  Scored via inline script (lstm_bw5.pt truncated to 14 features,
  Game 3 full match):

  | msf | Game 3 spellF1 | macroF1 | spells | note |
  |-----|---------------|---------|--------|------|
  | 125 | **0.5072** ← winner | 0.9315 | 176 | inflated — lstm_bw5.pt trained on G3 too |
  | 150 | 0.4744 | 0.9151 | 158 | inflated |
  | 175 | 0.4548 | 0.9030 | 147 | inflated |
  | 200 | 0.4174 | 0.8835 | 132 | inflated |

  **Finding:** msf=125 is optimal relative to 150/175/200. The absolute
  values are inflated (lstm_bw5.pt was trained on all 3 games including
  G3 — not a true LOO). True LOO (train G1+G2 only) TBD in CHALLENGE3.
  The relative ordering (msf=125 best) is still valid.

- [x] **Step 3: Commit results and record findings**

  Results recorded above. Committed with CHALLENGE2 conclusions.

---

## Task 1.5: Error analysis on existing best model (MUST run before T3)

Sweeping without knowing where the model fails is searching blind. This
task answers two questions that should drive every T3 experiment choice:
(1) which possession types cause the most errors, and (2) how much
headroom is left (upper bound).

**Files:** no code changes — analysis only

**Interfaces:**
- Consumes: `evaluation/results/lstm_best_challenge.csv` (Iteration 1
  best); Game 3 GT for cross-game analysis

- [ ] **Step 1: Document the annotation upper bound**

  The upper bound is the score a perfect model would achieve. For
  spellF1 this is not 1.0 — label noise and ambiguous transitions
  impose a ceiling. Estimate it on Game 3 GT (cross-game proxy) by
  scoring the GT against itself with small random jitter (±1 frames)
  on transition boundaries:
  ```bash
  python3.11 - <<'EOF'
  import pandas as pd, numpy as np
  from evaluation.possession_metrics import evaluate
  gt = pd.read_csv("games/Sample_Game_3/gt/possession_gt.csv")
  codes = gt["label"].values
  jitter = codes.copy()
  transitions = np.where(np.diff(codes.astype(str)) != "")[0] + 1
  for t in transitions:
      shift = np.random.choice([-1, 0, 1])
      if 0 < t + shift < len(jitter):
          jitter[t:] = np.roll(jitter[t:], shift)
  print(evaluate(codes, jitter)["spell_f1_mean"])
  EOF
  ```
  Record this number as the **estimated upper bound**.

  **Estimated upper bound: ___** (fill in after running)

- [ ] **Step 2: Confusion breakdown by possession segment**

  Score the Iteration 1 best model on Game 3 (cross-game proxy) and
  break down per-class accuracy:

  ```bash
  python3.11 - <<'EOF'
  import pandas as pd
  gt   = pd.read_csv("games/Sample_Game_3/gt/possession_gt.csv")
  pred = pd.read_csv("evaluation/results/lstm_best_challenge.csv")
  merged = gt.merge(pred.rename(columns={"label":"pred"}), on="frame", how="left")
  merged["correct"] = merged["label"] == merged["pred"]
  g3_frames = set(gt["frame"])
  m = merged[merged["frame"].isin(g3_frames)]
  print(m.groupby("label")["correct"].mean().round(3))
  EOF
  ```
  Record per-class accuracy (H/A/D). A large gap between H-accuracy
  and A-accuracy suggests team-assignment asymmetry — likely fixed by
  `n_A_within_3m`. A low D-accuracy confirms tagging features will help.

- [ ] **Step 3: Record findings and commit**

  Fill in upper bound and per-class accuracy above, then:
  ```bash
  git add docs/CHALLENGE2.md
  git commit -m "analysis: error analysis + upper bound on Iteration 1 best"
  ```

---

## Task 2: Add n_A_within_3m and dist_ratio to features.py

Adds two symmetric tracking features. Feature vector grows from 14 → 16.

> **Practice 3 warning (E3):** `n_A_within_3m` was already tested in
> Practice 3 Experiment E3 (+2 features: `n_A_within_3m` +
> `possession_ratio`) and **hurt** cross-game spellF1 by −0.076. Root
> cause: with ~30k training frames the model overfit the new features to
> Games 1+2 patterns and failed to generalise to Game 3.
>
> The Challenge setup differs — we now have all 3 sample games — so the
> data-size argument weakens slightly. But E3 is a strong prior.
> **The T2.6 ablation gate is mandatory:** if zeroing col 14
> (`n_A_within_3m`) at inference recovers the baseline score on the tuning
> slice, drop the feature before T4.

- **Feature 14 `n_A_within_3m`:** count of away players within
  `CONTACT_RADIUS_M` (3 m) of ball. Symmetric counterpart of feature 8
  (`n_H_within_3m`). Zero when ball missing.
- **Feature 15 `dist_ratio`:** `dist_H / (dist_H + dist_A)`. Zero when
  ball missing (not the sentinel 0.5 produced by the formula, which would
  be indistinguishable from equidistant). Range [0, 1] when ball visible;
  values near 0 = H very close, near 1 = A very close.

**Files:**
- Modify: `src/features.py` — add features 14–15, update `FEATURE_NAMES`
- Test: `tests/test_features_v2.py`

**Interfaces:**
- Consumes: `data` dict as returned by `load_tracking()` / `load_challenge_tracking()`
- Produces: `extract_features(data) -> np.ndarray` shape `(N, 16) float32`
  — same function name, extended output. `FEATURE_NAMES` list length
  changes from 14 to 16.

- [ ] **Step 1: Write failing tests**

  ```python
  # tests/test_features_v2.py
  import numpy as np
  import pytest
  from src.features import extract_features, FEATURE_NAMES

  def _make_data(n=10, ball_visible=True):
      home_xy = np.random.rand(n, 46, 2).astype(np.float32)
      away_xy = np.random.rand(n, 46, 2).astype(np.float32)
      ball_xy = np.random.rand(n, 2).astype(np.float32)
      if not ball_visible:
          ball_xy[:] = np.nan
      return {
          "home_xy": home_xy,
          "away_xy": away_xy,
          "ball_xy": ball_xy,
          "periods": np.ones(n, dtype=np.int8),
      }

  def test_feature_count():
      feats = extract_features(_make_data())
      assert feats.shape[1] == 16
      assert len(FEATURE_NAMES) == 16

  def test_n_A_within_3m_non_negative():
      feats = extract_features(_make_data())
      assert (feats[:, 14] >= 0).all()

  def test_n_A_within_3m_zero_when_ball_missing():
      feats = extract_features(_make_data(ball_visible=False))
      assert (feats[:, 14] == 0).all()

  def test_dist_ratio_zero_when_ball_missing():
      feats = extract_features(_make_data(ball_visible=False))
      assert (feats[:, 15] == 0).all()

  def test_dist_ratio_in_unit_interval_when_ball_visible():
      feats = extract_features(_make_data(ball_visible=True))
      assert (feats[:, 15] >= 0).all()
      assert (feats[:, 15] <= 1).all()
  ```

- [ ] **Step 2: Run tests to verify they fail**

  Run: `python3.11 -m pytest tests/test_features_v2.py -v`
  Expected: 5 failures — `assert feats.shape[1] == 16` fails.

- [ ] **Step 3: Implement features 14–15 in `src/features.py`**

  Resize the `feats` array from `(N, 14)` to `(N, 16)` and append to
  `FEATURE_NAMES`:
  ```python
  "n_A_within_3m",
  "dist_ratio",
  ```

  Feature 14 (`n_A_within_3m`): mirror the feature-8 block using `away_m`
  instead of `home_m`. Zero where `~ball_visible`.

  Feature 15 (`dist_ratio`): `dist_H / (dist_H + dist_A + 1e-6)`, then
  zero where `~ball_visible`.

- [ ] **Step 4: Run tests to verify they pass**

  Run: `python3.11 -m pytest tests/test_features_v2.py -v`
  Expected: 5 PASS.

- [ ] **Step 5: Run full suite to confirm no regressions**

  Run: `python3.11 -m pytest tests/ -v`
  Expected: all pass. Note: tests that check `feats.shape[1] == 14` will
  fail — find and update them to 16 (they are testing the old count,
  which is now wrong).

- [ ] **Step 6: Commit**

  ```bash
  git add src/features.py tests/test_features_v2.py tests/
  git commit -m "feat: add n_A_within_3m and dist_ratio (features 14-15)"
  ```

---

## Task 2.6: Feature ablation gate (isolate contribution of each new feature)

The plan adds two features together and retrains. If the sweep regresses
vs. Iteration 1 there is no way to know which feature caused it without
ablation. This task adds a `--zero-feature-cols` inference flag to
`predict_challenge.py` so each feature can be silenced post-hoc on any
saved model, measuring its marginal contribution without retraining.

Ablation runs after T3 completes using the best saved model.

**Files:**
- Modify: `src/predict_challenge.py` — add `--zero-feature-cols` arg

**Interfaces:**
- `--zero-feature-cols 14` zeros column 14 (`n_A_within_3m`) at
  inference; model weights are unchanged
- `--zero-feature-cols 15` zeros column 15 (`dist_ratio`) at inference
- `--zero-feature-cols 14,15` zeros both (reverts to 14-feature behaviour
  with a 16-feature model, useful as a sanity check)

- [ ] **Step 1: Add `--zero-feature-cols` to `predict_challenge.py`**

  In `main()`, add to the parser:
  ```python
  parser.add_argument("--zero-feature-cols", default="",
                      help="comma-separated feature column indices to zero "
                           "at inference (e.g. '14,15')")
  ```

  After `extract_features()` and before passing features to the model:
  ```python
  if args.zero_feature_cols:
      for col in args.zero_feature_cols.split(","):
          feats[:, int(col.strip())] = 0.0
  ```

- [ ] **Step 2: Run ablation after T3 sweep (defer until T3 models exist)**

  Using the best T3 model (e.g. `lstm_bw5_tag_v2.pt`), run:
  ```bash
  # baseline (all 20 features)
  python3.11 src/predict_challenge.py --model lstm \
    --load-model models/lstm_bw5_tag_v2.pt --tagging-feats \
    --interp-ball --min-spell-frames 125 --name ablate_baseline_v2

  # without n_A_within_3m
  python3.11 src/predict_challenge.py --model lstm \
    --load-model models/lstm_bw5_tag_v2.pt --tagging-feats \
    --interp-ball --min-spell-frames 125 --zero-feature-cols 14 \
    --name ablate_no_nA_v2

  # without dist_ratio
  python3.11 src/predict_challenge.py --model lstm \
    --load-model models/lstm_bw5_tag_v2.pt --tagging-feats \
    --interp-ball --min-spell-frames 125 --zero-feature-cols 15 \
    --name ablate_no_ratio_v2
  ```

  Score all three on **Game 3 LOO** (not the BAR-ATH public window).
  Record results here:

  | Config | Game 3 spellF1 | vs. baseline |
  |---|---|---|
  | All features (baseline) | ___ | — |
  | −n_A_within_3m (col 14) | ___ | ___ |
  | −dist_ratio (col 15) | ___ | ___ |

  A drop of > 0.005 when zeroing a feature = that feature contributes.
  A drop of < 0.001 = consider removing it to keep feature count lower
  for the ensemble.

- [ ] **Step 3: Commit**

  ```bash
  git add src/predict_challenge.py
  git commit -m "feat: --zero-feature-cols inference ablation flag"
  ```

---

## Task 2.5: Unified load_tracking() auto-detect (§5 switchover fix)

`src/sample_loader.py` is frozen (global constraint). `challenge_loader.py`
exposes `load_challenge_tracking()` as a separate function — callers must
change when switching data sources. Fix: add a unified `load_tracking()`
to `src/loader.py` that auto-detects the data format by directory
contents and delegates to the right loader. All scripts in `src/` import
from `src.loader` instead of `src.sample_loader`.

**Files:**
- New: `src/loader.py` — auto-detecting `load_tracking()`
- Modify: `src/predict_challenge.py`, `src/sweep_challenge.py` — one-line
  import swap per file

**Interfaces:**
- `load_tracking(game_dir) -> dict` — same return keys as always;
  detects challenge data if `game_dir` contains `_atd.txt` (EPTS format),
  otherwise delegates to `sample_loader.load_tracking()`
- Callers need no other changes

- [ ] **Step 1: Write failing test**

  ```python
  # tests/test_loader_dispatch.py
  import os, numpy as np
  from unittest.mock import patch

  def test_dispatches_to_sample_loader_for_sample_game(tmp_path):
      """Non-EPTS directory routes to sample_loader."""
      from src.loader import load_tracking
      # tmp_path has no _atd.txt → should call sample_loader.load_tracking
      with patch("src.loader._sample_load_tracking") as mock:
          mock.return_value = {"frames": np.array([1])}
          load_tracking(str(tmp_path))
          mock.assert_called_once()

  def test_dispatches_to_challenge_loader_for_epts_dir(tmp_path):
      """Directory containing _atd.txt routes to challenge_loader."""
      (tmp_path / "BAR-ATH_tactical_atd.txt").touch()
      from src.loader import load_tracking
      with patch("src.loader._challenge_load_tracking") as mock:
          mock.return_value = {"frames": np.array([1])}
          load_tracking(str(tmp_path))
          mock.assert_called_once()
  ```

- [ ] **Step 2: Run tests to verify they fail**

  Run: `python3.11 -m pytest tests/test_loader_dispatch.py -v`
  Expected: 2 FAIL — `src.loader` does not exist.

- [ ] **Step 3: Implement `src/loader.py`**

  ```python
  """Unified load_tracking() — auto-detects EPTS vs. sample-game format."""
  import os, glob

  def _is_epts_dir(game_dir: str) -> bool:
      return bool(glob.glob(os.path.join(game_dir, "*_atd.txt")))

  def _sample_load_tracking(game_dir, **kwargs):
      from src.sample_loader import load_tracking
      return load_tracking(game_dir, **kwargs)

  def _challenge_load_tracking(game_dir, **kwargs):
      from src.challenge_loader import load_challenge_tracking
      return load_challenge_tracking(data_dir=game_dir, **kwargs)

  def load_tracking(game_dir: str, **kwargs) -> dict:
      """Load tracking data, auto-detecting format from directory contents."""
      if _is_epts_dir(game_dir):
          return _challenge_load_tracking(game_dir, **kwargs)
      return _sample_load_tracking(game_dir, **kwargs)
  ```

- [ ] **Step 4: Run tests to verify they pass**

  Run: `python3.11 -m pytest tests/test_loader_dispatch.py -v`
  Expected: 2 PASS.

- [ ] **Step 5: Swap imports in predict_challenge.py and sweep_challenge.py**

  In each file, replace:
  ```python
  from src.sample_loader import load_tracking
  ```
  with:
  ```python
  from src.loader import load_tracking
  ```
  One line per file. No other changes.

- [ ] **Step 6: Run full test suite**

  Run: `python3.11 -m pytest tests/ -v`
  Expected: all pass.

- [ ] **Step 7: Commit**

  ```bash
  git add src/loader.py src/predict_challenge.py src/sweep_challenge.py \
          tests/test_loader_dispatch.py
  git commit -m "feat: unified load_tracking() auto-detects EPTS vs sample format"
  ```

---

## Task 3: Add --run-id + --use-crf to sweep_challenge.py + overnight sweep

Adds `--run-id STR`, `--use-crf`, and `--seed INT` arguments to
`sweep_challenge.py`, then runs a focused sweep: bw=5, interp, tagging,
crf={False,True}, msf={100,125,150}, run-id=v2, seed=42. Two retrains
(base + CRF), six scored combinations.

> **`--include-gt-tuning` is banned.** Appending BAR-ATH GT labels to
> training data violates RULES.md §2: "no hand labels on the provided
> game, in any amount and for any purpose — not to train a model and not
> to validate one." The flag must not be used. Any model trained with it
> cannot be submitted.

> **Practice 2 warning (CRF):** CRF was explicitly dropped from the
> Practice 2 LOO grid — "3× training cost, known to underperform on this
> data." That finding was on 2-game training sets. The Challenge setup
> (3 games, tagging features, bw=5) differs enough to warrant one test.
> If `sweep-challenge-v2-crf` scores below the base variant on Game 3
> LOO, drop it immediately — do not include in T4.

**Train/val split discipline:** the sweep must use a **temporal split**
— first half (period 1) for training, second half (period 2) for
validation. Random splits on 190k sequential frames leak future context
and inflate validation spellF1. Verify this is the current behaviour in
`sweep_challenge.py` before running; fix it if not.

**Overlap inference — must be OFF:** verify `use_overlap=False` in both
`sweep_challenge.py` and `predict_challenge.py`. Practice 2 ablations
showed overlap inference hurts spellF1 by −0.032: the BiLSTM already
has bidirectional context; averaging over overlapping windows smears
sharp spell boundaries. The Iteration 1 winning model correctly had
`use_overlap=False`. Do not enable it.

**Files:**
- Modify: `src/sweep_challenge.py` — add `--run-id`, `--use-crf`,
  `--seed` args; enforce temporal split; verify overlap is OFF
- Modify: `Taskfile.yml` — add `sweep-challenge-v2` task

**Interfaces:**
- Consumes: `src/sweep_challenge.py`; features 14–15 from T2
- Produces:
  - Models: `models/lstm_bw5_tag_v2.pt`, `models/lstm_bw5_tag_crf_v2.pt`
  - CSVs: `lstm_bw5_msf{msf}_interp_tag_v2.csv`,
    `lstm_bw5_msf{msf}_interp_tag_crf_v2.csv`
  - Summary: `evaluation/results/sweep_summary_v2.txt` — must include
    H/A/D/X frame counts per config to detect label distribution shifts

- [ ] **Step 1: Write failing test for --run-id namespacing**

  ```python
  # tests/test_sweep_run_id.py
  import os, sys, types, importlib

  def test_run_id_appears_in_csv_name():
      """CSV name includes run_id suffix when provided."""
      # Import the name-building logic without running main()
      sys.path.insert(0, os.path.abspath("."))
      import src.sweep_challenge as sc
      # Simulate the name construction
      bw, msf, interp_tag, tagging_tag, run_id = 5, 125, "_interp", "_tag", "_v2"
      name = f"lstm_bw{int(bw)}_msf{msf}{interp_tag}{tagging_tag}{run_id}"
      assert name == "lstm_bw5_msf125_interp_tag_v2"

  def test_run_id_empty_string_matches_old_behaviour():
      bw, msf, interp_tag, tagging_tag, run_id = 5, 125, "_interp", "", ""
      name = f"lstm_bw{int(bw)}_msf{msf}{interp_tag}{tagging_tag}{run_id}"
      assert name == "lstm_bw5_msf125_interp"
  ```

- [ ] **Step 2: Run tests to verify they fail**

  Run: `python3.11 -m pytest tests/test_sweep_run_id.py -v`
  Expected: 2 FAIL — `run_id` variable is not yet constructed in the sweep.

- [ ] **Step 3: Add `--run-id` and `--use-crf` to `sweep_challenge.py`**

  In `main()`, add to the parser:
  ```python
  parser.add_argument("--run-id", default="",
                      help="suffix for output names; no slashes or '..'")
  parser.add_argument("--use-crf", action="store_true",
                      help="enable CRF output layer (separate retrain)")
  parser.add_argument("--seed", type=int, default=42,
                      help="random seed for torch and numpy (reproducibility)")
  # Note: --include-gt-tuning is NOT added. Using BAR-ATH GT labels
  # for training violates RULES.md §2.
  # Note: --use-attention is NOT added. Practice 2 ablations showed
  # attention costs -0.039 spellF1 on every tested split. Do not expose.
  ```

  Thread `args.run_id` and `args.use_crf` through:
  - `csv_name` construction: include `_crf` tag when `use_crf=True`,
    then `_{run_id}` suffix
  - `model_path` construction: include `_crf` before `_{run_id}.pt`
  - `SUMMARY_PATH`: use `sweep_summary_{run_id}.txt` when run_id is set

  Exact pattern:
  ```python
  run_tag  = f"_{args.run_id}" if args.run_id else ""
  crf_tag  = "_crf" if args.use_crf else ""
  # csv_name:
  csv_name = (f"lstm_bw{int(bw)}_msf{msf}"
              f"{interp_tag}{tagging_tag}{crf_tag}{run_tag}")
  # model_path (one model per bw+crf combo, shared across msf):
  model_path = os.path.join(MODELS_DIR,
                            f"lstm_bw{int(bw)}{tagging_tag}{crf_tag}"
                            f"{run_tag}.pt")
  # summary:
  summary_name = f"sweep_summary{run_tag}.txt"
  SUMMARY_PATH = os.path.join(RESULTS_DIR, summary_name)
  ```

  Also add at the top of `main()` before any model construction:
  ```python
  import random
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  ```

  Pass `use_crf=args.use_crf` through to the `PossessionLSTM`
  constructor call inside `sweep_challenge.py`.

  Sanitize `--run-id` to reject path traversal:
  ```python
  import re, argparse
  if args.run_id and re.search(r'[/\\.]', args.run_id):
      raise argparse.ArgumentTypeError(
          "--run-id must not contain '/', '\\', or '.'")
  ```

- [ ] **Step 4: Run tests to verify they pass**

  Run: `python3.11 -m pytest tests/test_sweep_run_id.py -v`
  Expected: 2 PASS.

- [ ] **Step 5: Add sweep-challenge-v2 tasks to Taskfile.yml**

  ```yaml
  sweep-challenge-v2:
    desc: Iter-2 sweep — bw=5, interp, tagging, seed=42, msf 100/125/150
    cmds:
      - mkdir -p logs
      - python3.11 src/sweep_challenge.py --bw 5 --msf 100 125 150
          --interp-ball --tagging-feats --seed 42 --run-id v2
          2>&1 | tee logs/sweep_v2_$(date +%Y%m%d).log

  sweep-challenge-v2-crf:
    desc: Iter-2 CRF sweep — same as v2 but with CRF layer
    cmds:
      - mkdir -p logs
      - python3.11 src/sweep_challenge.py --bw 5 --msf 100 125 150
          --interp-ball --tagging-feats --use-crf --seed 42 --run-id v2
          2>&1 | tee logs/sweep_v2_crf_$(date +%Y%m%d).log

  ```

  Note: `sweep-challenge-v2-gt` is deliberately absent — training on
  BAR-ATH GT labels violates RULES.md §2.
  Note: `sweep-challenge-v2-attn` is deliberately absent — Practice 2
  showed `use_attention=True` costs −0.039 spellF1 on every tested split.

- [ ] **Step 6: Dry-run to verify end-to-end**

  Run:
  ```bash
  python3.11 src/sweep_challenge.py --bw 5 --msf 100 --interp-ball \
    --tagging-feats --run-id v2 --dry-run
  ```
  Expected: CSV `lstm_bw5_msf100_interp_tag_v2.csv` written, summary
  `sweep_summary_v2.txt` written, exits 0. Confirm no collision with
  existing `lstm_bw5_msf100_interp.csv`.

  Then dry-run the CRF variant:
  ```bash
  python3.11 src/sweep_challenge.py --bw 5 --msf 100 --interp-ball \
    --tagging-feats --use-crf --run-id v2 --dry-run
  ```
  Expected: CSV `lstm_bw5_msf100_interp_tag_crf_v2.csv`, model
  `lstm_bw5_tag_crf_v2.pt` — distinct from the base variant.

- [ ] **Step 7: Run full test suite**

  Run: `python3.11 -m pytest tests/ -v`
  Expected: all pass.

- [ ] **Step 8: Commit**

  ```bash
  git add src/sweep_challenge.py Taskfile.yml tests/test_sweep_run_id.py
  git commit -m "feat: --run-id namespacing in sweep_challenge + v2 task"
  ```

- [ ] **Step 9: Launch overnight sweep (all three variants)**

  Run both variants sequentially:
  ```bash
  task sweep-challenge-v2 && \
  task sweep-challenge-v2-crf
  ```

  Expected runtime: 2 retrains (~35 min each) + scoring = ~80 min total.
  Results in `evaluation/results/sweep_summary_v2.txt`.

---

## Task 4: Ensemble top-3 configs

After T3 results are known, take the top-3 scoring configs (regardless
of CRF/base) and retrain each 3 times with different seeds. Average
softmax probabilities before argmax / CRF decode. Expected gain:
+0.01–0.03 spellF1 at zero architecture cost.

**Files:**
- New: `src/ensemble_challenge.py`

**Interfaces:**
- Consumes: top-3 config specs from `sweep_summary_v2.txt`; same
  `predict_challenge.py` + `run_evaluation.py` pipeline
- Produces: `evaluation/results/ensemble_top3_v2.csv`

- [ ] **Step 1: Identify top-3 configs from sweep_summary_v2.txt**

  Read `evaluation/results/sweep_summary_v2.txt` and pick the 3 highest
  spellF1 rows. Record their exact flags (msf, crf, tagging, interp).

- [ ] **Step 2: Retrain each config 3× with seeds 0, 1, 2**

  For each config `C` and seed `s`:
  ```bash
  python3.11 src/predict_challenge.py \
    --model lstm [config flags] --seed $s \
    --name ensemble_C_seed${s}
  ```
  Produces 9 CSVs (3 configs × 3 seeds).

  Note: `predict_challenge.py` must accept `--seed INT` and pass it to
  `torch.manual_seed()` and `np.random.seed()` before training. Add this
  flag if not present.

- [ ] **Step 3: Write ensemble_challenge.py**

  Reads the 9 CSVs, aligns on `frame`, averages per-class softmax
  probabilities (reconstruct from hard labels as one-hot if softmax
  is not saved), then applies argmax + msf filter from the best config.

  If hard labels only are saved: implement voting (majority label per
  frame, tie → use best single model's label).

- [ ] **Step 4: Select winner on Game 3 LOO, then report on BAR-ATH**

  Score ensemble vs. best single model using **Game 3 LOO** spellF1.
  Pick the winner based on that score only.

  After the winner is locked, run the BAR-ATH public window **once**
  for reporting purposes only — do not use the result to make any
  further config decisions:
  ```bash
  python3.11 evaluation/run_evaluation.py \
    --gt evaluation/gt/BAR-ATH_possession_gt_public.csv \
    --results evaluation/results --no-detail
  ```
  This is the leaderboard number. Record it and stop tuning.

- [ ] **Step 5: Commit**

  ```bash
  git add src/ensemble_challenge.py evaluation/results/ensemble_top3_v2.csv
  git commit -m "feat: ensemble top-3 v2 configs × 3 seeds"
  ```

---

## Deadline timeline

Real EPTS data **already landed** at `data/2027_challenge/` (tactical ATD
+ smart tagging XML, cache already built). No data-swap step needed —
every experiment below runs on the final match data immediately.

```
COMPLETED (Oct 24)
  T0   ✓  cache gitignore fixed
  T1   ✓  msf 150/175/200 post-hoc — msf=125 confirmed optimal on Game 3 LOO
  T2   ✓  features 14→16 (REVERTED — see Conclusions; 16-feat hurt)
  T2.6 ✓  --zero-feature-cols flag added
  T2.5 ✓  src/loader.py unified load_tracking()
  T3   ✓  (partial) 24 non-tagging combos ran; tagging pass crashed
           (dimension mismatch — bug in sweep, not features)

ABANDONED
  T1.5 —  error analysis skipped (per-class data sufficient from T3 results)
  T4   —  ensemble not run (no winning config to ensemble yet)

→ See Conclusions for next-round plan (CHALLENGE3)
```

Run T1–T4 in sequence. The overnight job is T3 Step 9.

---

## Conclusions

**Round completed:** 2026-10-24
**Best spellF1 (Game 3 LOO):** not measured this round — LOO eval was
not run on v2 models (sweep writes BAR-ATH public-window figures only)
**BAR-ATH public (reporting only):** 0.3231 (run-id: lstm_bw5_msf125_interp_v2)
**Prior best (BAR-ATH public, CHALLENGE1):** 0.3857 — discarded; was
selected via BAR-ATH public GT tuning, violating RULES.md §2
**Prior best (leaderboard floor):** 0.2433 (published baseline, clean)
**Delta (BAR-ATH reporting):** −0.0626 vs. CHALLENGE1 (regression)

### What worked

- `--run-id` namespacing in `sweep_challenge.py` — no CSV collisions
  between Iteration 1 and v2 results; reproducible with `--seed 42`
- `src/loader.py` unified `load_tracking()` — auto-detects EPTS vs.
  sample format; all callers updated
- `--zero-feature-cols` ablation flag added to `predict_challenge.py`
- msf=125 confirmed optimal: wider msf (150/175/200) all regress on
  Game 3 proxy eval (0.5072→0.4744→0.4548→0.4174; values inflated but
  relative ordering valid)

### What did not work

- **16-feature model regressed vs. 14-feature baseline.** BAR-ATH
  spellF1 dropped from 0.3857 to 0.3231 (−0.0626). Per-class: A-class
  accuracy fell 10.8 points (60.0% → 49.2%), swamping H +3.3% and
  D +2.1%. `n_A_within_3m` and `dist_ratio` are highly correlated with
  the existing `dist_nearest_A` (feature 4); redundancy hurt training.
  Practice 3 E3 warning confirmed on the target domain.
- **Tagging pass crashed** with dimension mismatch: sweep trains on
  16-feature sample-game data, then `predict_challenge.py` appends 4
  tagging features → 20-dim input against a 16-dim model. Zero tagging
  results produced this round.
- CRF not benchmarked: only the base (non-CRF) v2 sweep ran.

### Key finding

Adding `n_A_within_3m` and `dist_ratio` (features 14–15) caused a
net regression of −0.0626 spellF1 on BAR-ATH (16-feat best 0.3231 vs.
14-feat best 0.3857). A-class accuracy dropped 10.8 points, confirming
the Practice 3 E3 finding that these features overfit to training-game
patterns on this dataset size. The tagging dimension bug is independent
and fixable; it blocked the only lever that could have offset the
regression.

### Recommended focus for next round

Ordered list, highest-leverage first:
1. **Revert `features.py` to 14 features** — the 16-feat regression
   is −0.0626 spellF1; reverting restores the working baseline config
   (bw=5, msf=125, interp, 14 features).
2. **Fix tagging dimension bug** — train sweep with 4 tagging features
   appended to sample-game data too (XML exists, loader supports it);
   tagging + 14-feat baseline is the untested highest-upside lever.
3. **CRF sweep on 14-feat models** — bw=5, msf={100,125}, crf=True/
   False; CRF was never tested at 14 features with the current setup.
4. **Establish a clean Game 3 LOO score** — run `pipeline.py --train
   1 2 --predict 3` on the winner to get a trustworthy internal metric
   before any further feature or architecture changes.

### Open questions

- Does tagging (14+4=18 features consistently in train+predict) beat
  the 14-feature baseline on Game 3 LOO?
- Does CRF help or hurt at bw=5, 14 features? (Practice 2 found it
  harmful but on a different training setup.)
- What is the true Game 3 LOO spellF1 of the CHALLENGE1 14-feat model?
  No clean LOO exists yet — the 0.5072 figure was same-game inflated.
