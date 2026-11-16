# FC Barcelona Hackathon 2027 — Project Architecture

## Goal

Predict frame-level ball possession (H / A / D / X) for the BAR-ATH match.
Scored on **spellF1** — mean F1 over IoU thresholds 0.3–0.7 for matched possession spells.
Competition floor to beat: **0.2433** (BAR-ATH leaderboard, smooth50 baseline).

---

## Data

| Source | Format | Frames | Status |
|--------|--------|--------|--------|
| Sample Game 1 | CSV (home + away separate) | 145,006 | Downloaded |
| Sample Game 2 | CSV (home + away separate) | 141,156 | Downloaded |
| Sample Game 3 | EPTS (unified tracking.txt + metadata.xml + events.json) | 143,761 | Downloaded |
| BAR-ATH (real) | EPTS | 199,319 | Arrives Oct 20 in `data/tactical/` |

All three sample games have possession GT CSVs (H/A/D/X per frame) built by
`task sample-data`. Public 5-min evaluation slices (7,500 graded frames) live in
`evaluation/gt/` for each game and for BAR-ATH.

Coordinates are normalised `[0, 1]` in both formats. Arrays are always padded to
`(N, 46, 2)` — 46 player slots, NaN where a player is absent.

---

## Architecture

```
data/
  sample/Sample_Game_{1,2,3}/   raw tracking + events + possession_gt.csv
  tactical/                      BAR-ATH data (Oct 20)
  loo_cache/                     Per-combo LOO result JSON files (one per combo, cache-versioned)

src/
  sample_loader.py    load_tracking(game_dir) → dict{frames, periods, home_xy, away_xy, ball_xy}
                      Auto-detects EPTS vs CSV format. Caches EPTS as .npz (mtime-checked).
                      load_gt(game_dir)        → np.ndarray (N,) '<U1'

  features.py         extract_features(data)   → np.ndarray (N, 14) float32
                      ball_x, ball_y, ball_visible, dist_nearest_H, dist_nearest_A,
                      delta_dist_H, delta_dist_A, ball_speed, n_H_within_3m, ball_accel,
                      spread_H, spread_A, centroid_dist_H, centroid_dist_A.
                      Sentinel contamination (ball reappearance spike) zeroed via just_appeared mask.

  dead_ball.py        dead_ball_mask(events_path, n_frames) → np.ndarray (N,) bool
                      Reads JSON (Game 3 / BAR-ATH) or CSV (Games 1 & 2) events.
                      Dead types: BALL OUT, BALL LOST, CHALLENGE, FAULT RECEIVED, CARD.

  possession_hmm.py   PossessionHMM  — GaussianHMM (hmmlearn), 3 states (H/A/D)
                      .fit(features, gt_labels, dead_mask)
                      .predict(features, dead_mask) → labels after dead-mask override + spell filter
                      .save(path) / .load(path)  via joblib
                      MIN_SPELL_FRAMES = 10 (overridable via pipeline --min-spell-frames)

  possession_lstm.py  PossessionLSTM — 2-layer BiLSTM → linear → softmax (H/A/D)
                      Same interface as PossessionHMM (drop-in replacement).
                      Defaults: hidden_size=256, max_epochs=40, bw=2.5, cosine LR, no attention, no overlap.
                      Constructor flags: use_attention, use_overlap, lr (all tunable, defaults proven optimal).
                      .save(path) / .load(path)  via torch.save — checkpoint includes all flags.

  pipeline.py         train_model / predict_with_model
                      CLI: --train 1 2 3 --predict N --model hmm|lstm --out CSV
                           --save-model PATH  (train + save, then continue to predict)
                           --load-model PATH  (skip training, load from disk)
                           --predict-dir PATH (BAR-ATH: override game number with raw path)
                           --hidden-size 256 --epochs 40 --boundary-weight 2.5
                           --lr LR  --min-spell-frames N
                      Logs to MLflow experiment "possession-hmm".

  tune.py             Grid-search MIN_SPELL_FRAMES × dead-ball buffer on public 5-min slice.
                      Logs all runs to MLflow experiment "possession-hmm-tune".

  tune_loo.py         3-fold LOO cross-validation grid search for LSTM hyperparams.
                      Accepts --data-root (default: data/sample). Per-combo cache in data/loo_cache/.
                      Cache versioned by MD5 of features.py + possession_lstm.py — auto-invalidates.

  phase2_ablations.py Isolated ablation runner: baseline / overlap / cosine-LR / attention / combined.
                      Accepts --data-root, --hidden-size, --epochs, --boundary-weight.

  predict_bar_ath.py  One-command Oct 20 submission script.
                      --data-root data/tactical --model models/lstm_v2.pt --out CSV
                      Loads tracking + dead-ball, predicts, writes X for ungraded frames,
                      prints row count + label distribution.

  dashboard/app.py    Streamlit frame-by-frame debugger.
                      Sidebar: game directory dropdown (auto-discovers data/), prediction file,
                      frame slider, metrics. Pitch plot (Plotly) + 250-frame timeline.

evaluation/
  run_evaluation.py   Scores results/ CSVs against a GT. --gt takes 'public', 'private', or path.
  possession_metrics.py  spellF1, macroF1, transF1, latency, fragmentation.
  gt/                 Public 5-min GT slices for all 3 sample games + BAR-ATH.
  results/            Submission CSVs written by pipeline.py.
  scoreboard.csv      Latest ranked results.

models/
  lstm_v2.pt          Best model — h=256, e=40, bw=2.5, cosine LR, trained on Games 1+2+3.
```

---

## Task Workflow

```
task sample-data     # download all 3 games + build GT CSVs + public slices
task test            # 66 tests — must be green before anything else
task train           # train HMM + LSTM on games 1+2+3; predict on each; write 6 CSVs + save models
task score           # scoreboard per game against each game's own 5-min GT
task tune            # grid-search hyperparams across all 3 games
task tune-full       # validate best params on full GT (once only — do not iterate)
task dashboard       # Streamlit frame viewer
task browse          # open MLflow (port 5001) + Streamlit (port 8502) in Chrome
```

Oct 20 additions:
```
task score-barth     # score BAR-ATH submission against BAR-ATH public GT
```

---

## Current Results (Game 3 public 5-min slice, train 1+2 → eval G3 unless noted)

| Rank | Model | Trained on | spellF1 | macroF1 |
|------|-------|------------|---------|---------|
| 1 | **lstm_v2** (models/lstm_v2.pt) | Games 1+2+3 | **0.6903** | 0.9805 |
| 2 | **lstm_v2 cross-game** | Games 1+2 only | **0.6811** | 0.9476 |
| 3 | lstm_game3 (old baseline) | Game 3 only | 0.649 | 0.978 |
| 4 | lstm_all (old) | Games 1+2+3 | 0.622 | 0.949 |
| 5 | hmm_game3_9feat | Game 3 | 0.244 | 0.304 |
| 6 | baseline smooth25 | — | 0.162 | 0.553 |

Competition floor: **0.2433**. Best model clears it by ~0.45 points.

Key findings from Practice 2+3 experiments:
- **Cosine LR** is the only architecture change that helps (+0.076 over fixed LR)
- **Self-attention and overlap inference both hurt** cross-game spellF1 (-0.04 each)
- **h=256 > h=128** specifically on unseen games (+0.07 on G3 fold)
- **e=40 is the ceiling** — more epochs overfit on 2-game training sets
- **All hyperparameter changes** (lr, bw, features, h=512) hurt cross-game performance
- **Root cause**: ~30k training frames / 2 games is too small for any complexity increase
- **Adding a 3rd training game** gives +0.030 (distribution shift mitigation > model tuning)

---

## Oct 20 Checklist

When BAR-ATH data lands in `data/tactical/`:

1. **One-command prediction** (preferred):
   ```bash
   python3.11 src/predict_bar_ath.py \
       --data-root data/tactical \
       --model models/lstm_v2.pt \
       --out evaluation/results/bar_ath_submission.csv
   ```

2. **Verify loader** if step 1 fails:
   ```bash
   python3.11 -c "from src.sample_loader import load_tracking; d = load_tracking('data/tactical'); print(d['frames'].shape)"
   ```

3. **Score and validate**:
   ```bash
   python evaluation/validate_submission.py evaluation/results/bar_ath_submission.csv
   task score-barth
   ```

4. **If time allows — retrain on all 4 known games** (sample 1+2+3 + BAR-ATH) once GT is released:
   ```bash
   python3.11 src/pipeline.py \
       --train 1 2 3 --predict-dir data/tactical --model lstm \
       --save-model models/lstm_v3.pt \
       --out evaluation/results/lstm_barth_retrained.csv
   ```

---

## Known Risks

| Risk | Mitigation |
|------|-----------|
| Domain shift: sample games ≠ BAR-ATH | Cross-game eval shows spellF1=0.68 on unseen game — realistic floor for BAR-ATH |
| BAR-ATH frame count (199,319) vs sample (143–145k) | `--predict-dir` bypasses game number validation; loader is format-agnostic |
| Dead-ball events in different format | `dead_ball.py` auto-detects JSON vs CSV; Smart Tagging XML path in `predict_bar_ath.py` |
| LSTM non-determinism across runs | Fixed seeds (_SEED=42) in fit() and DataLoader generator; save/load round-trip tested |
| Model overfits if retrained on BAR-ATH with e=40 | Same constraint as sample games — e=40 is the proven ceiling for small datasets |
