# Practice 1 — Conclusion (Oct 9–11, 2026)

10-day window, but effectively done in 3 days (Oct 9–11) before moving to Practice 2.

---

## What Was Built

| Commit | Deliverable |
|--------|-------------|
| `cac0077` | Sample game tracking loader (`src/sample_loader.py`) |
| `d5253ea` | Per-frame feature extraction (`src/features.py`) |
| `830cd35` | Dead-ball detection from events CSV (`src/dead_ball.py`) |
| `3c1bfa8` | HMM possession model + pipeline + Taskfile (`src/possession_hmm.py`, `src/pipeline.py`) |
| `90216df`–`0d658b7` | EPTS loader for real ATD data |
| `a37f014` | 5 new features + sentinel-spike fix in `src/features.py` |
| `627e0ad` | Hyperparameter tuning workflow (`src/tune.py`) |
| `c735222` | BiLSTM as drop-in model alternative (`src/possession_lstm.py`) |
| `8359535` | Frame-by-frame debug dashboard (`src/dashboard/app.py`) |
| `6cc52fb` | Pipeline lifted from single-game to full 3-game workflow |
| `d58ceae` | Smart Tagging XML dead-ball support; LSTM + tune.py expanded |

---

## Key Results

- **HMM baseline**: spellF1 = 0.2433 (smooth50, Game 3 public 5-min slice, BAR-ATH leaderboard floor)
- **LSTM baseline**: spellF1 = 0.649 (lstm_game3, same-game train+eval, Game 3 public slice)
- LSTM beats HMM by a large margin on same-game eval; cross-game generalisation investigated in Practice 2

## Key Findings

- **Same-game eval is not a reliable proxy** — training and evaluating on Game 3 inflates LSTM scores vs true cross-game transfer (explored in Practice 2)
- **Dead-ball masking matters** — forced `D` labels on dead frames meaningfully cleans up spell boundaries
- **Game 3 is the hardest held-out game** — different events format (JSON vs CSV), different distribution; the model trained on Games 1+2 struggles on it (spellF1 ~0.43–0.50 vs ~0.72 on G1/G2)
- **Feature engineering was the highest-leverage early work** — 5 additional features in `a37f014` improved HMM performance more than model tuning

## Oct 20 Checklist (when BAR-ATH data lands in `data/tactical/`)

1. Verify `load_tracking()` on BAR-ATH EPTS files
2. Train on all 3 sample games, predict on BAR-ATH
3. Use Smart Tagging XML for dead-ball (not CSV fallback)
4. Write `X` for ungraded frames via `games.half_video_blocks()`
5. Validate submission (199319 rows, valid labels)
6. Score against public GT — floor: spellF1 0.2433
7. Best model from Practice 2: plain BiLSTM + cosine LR, h=256, e=40, bw=2.5
