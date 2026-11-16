# Source Data

## Overview

The pipeline uses two data sources: three publicly available sample games
for training, and one gated real-match recording (BAR-ATH) for the final
submission prediction.

---

## Training data — Metrica Sports sample games

**Origin:** Metrica Sports public open-data repository on GitHub.
`https://github.com/metrica-sports/sample-data`

**How to obtain:** Run `task sample-data` on a fresh clone. The script
`src/task_sample_data.py` downloads each file from the GitHub raw URL,
places them under `data/sample/`, and generates possession GT CSVs.
No manual steps are required. Files are not committed to this repository
because they are already publicly available upstream.

| Game | Frames | Tracking format | Events format |
|------|--------|----------------|---------------|
| Sample_Game_1 | 145,006 | Two CSVs (home + away separate) | `_RawEventsData.csv` |
| Sample_Game_2 | 141,156 | Two CSVs (home + away separate) | `_RawEventsData.csv` |
| Sample_Game_3 | 143,761 | EPTS unified (`_tracking.txt` + `_metadata.xml`) | `_events.json` |

All coordinates are normalised to `[0, 1]`. Player arrays are padded to
`(N, 46, 2)` — 46 slot slots, `NaN` where a player is absent.

### Possession ground truth (GT)

The raw events files do not ship with possession labels. `task sample-data`
calls `src/task_sample_data.py` to derive them from the event stream:

- `BALL OUT`, `FAULT RECEIVED`, `CARD` → `D` (dead ball)
- Possession events (`SET PIECE`, `CHALLENGE`, etc.) → carry team label
  (`H` or `A`) from the last possessing team
- Frames outside any half → `X`

The initial GT is then corrected by running `src/generate_gt.py`, which
applies competition-rule relabelling:

- `BALL LOST` and `CHALLENGE` frames → carry-forward `H`/`A` (not `D`),
  because competition rules define a contested ball as belonging to the last
  possessing team, not as a dead ball

Corrected GT files are written to
`data/sample/<game>/<game>_possession_gt_corrected.csv`. The pipeline
requires these files to be present before training (`require_corrected_gt()`
preflight check in `src/pipeline.py` and `src/sweep_challenge.py`).

---

## Prediction target — BAR-ATH tactical recording

**Origin:** FC Barcelona vs Athletic Club, LaLiga EA Sports 2024-25,
matchday 2. Provided by Metrica Sports and FC Barcelona Innovation Hub as
part of the More Than A Hack 2027 challenge.

**How to obtain:** Gated — requires hackathon enrolment and acceptance of
the data-use terms. See `DATA.md` for the enrolment link and unpack
instructions. Unpacked files live in `data/tactical/` (gitignored).

| File | Description |
|------|-------------|
| `BAR-ATH_tactical_atd.xml` | EPTS tracking metadata |
| `BAR-ATH_tactical_atd.txt` | EPTS tracking data (199,319 frames) |
| `BAR-ATH_tactical_smart_tagging.xml` | 638 tactical phases + set pieces |
| `BAR-ATH_tactical_homography.csv` | 3×3 pitch-to-pixel matrix per frame |

Frame count: **199,319** at 25 fps. The submission CSV must have exactly
this many rows (one per frame, 0-indexed).

### Graded frames

Only the two halves are graded. Frames outside the halves are `X` and are
never scored.

| Segment | First frame | Last frame | Graded? |
|---------|-------------|------------|---------|
| Pre-match | 0 | 11,184 | No |
| **1st half** | **11,185** | **90,651** | **Yes** |
| Half-time | 90,652 | 110,807 | No |
| **2nd half** | **110,808** | **190,197** | **Yes** |
| Post-match | 190,198 | 199,318 | No |

158,857 graded frames total.

### Public GT window

`evaluation/gt/BAR-ATH_possession_gt_public.csv` — committed in this
repository, no download needed. Covers the first 5 minutes of play
(frames 11,185–18,684): 7,500 non-X frames (`H` 3,454 · `A` 2,577 ·
`D` 1,469). Used **for reporting only** — not for config selection or
training (competition rules, RULES.md §2).

---

## How training data flows into the model

```
task sample-data
  └─ src/task_sample_data.py
       Downloads Game 1-3 raw files from GitHub
       Generates _possession_gt.csv for each game

src/generate_gt.py
  └─ Writes _possession_gt_corrected.csv for each game
       (BALL LOST / CHALLENGE → carry-forward H/A)

src/sweep_challenge.py  (or src/pipeline.py)
  └─ require_corrected_gt()   ← exits if corrected GT absent
  └─ load_tracking(game_dir)  ← auto-detects EPTS vs CSV
  └─ extract_features()       ← 15 features (see features.py)
  └─ dead_ball_mask()         ← BALL OUT / CARD / FAULT RECEIVED only
  └─ PossessionLSTM.fit()     ← trains on all 3 corrected GTs
  └─ PossessionLSTM.predict() ← predicts on BAR-ATH
  └─ clamp_d_to_mask()        ← visible-ball D→H/A carry-forward
  └─ writes submission CSV
```

The final locked config (run-id v7) trains on all three sample games and
predicts on BAR-ATH. See `docs/FINAL_SUBMISSION_COMMAND.txt` for the
exact command.
