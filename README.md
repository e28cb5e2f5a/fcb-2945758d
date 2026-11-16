# More than a Hack 2027 — Ball Possession Submission

Bidirectional LSTM trained on the three Metrica public sample games,
applied frame-by-frame to the BAR-ATH tactical tracking data.

---

## Approach

A single bidirectional LSTM reads a window of per-frame features derived
from the automatic tracking data (player–ball distances, team centroids,
ball velocity, dead-ball indicators) and classifies each frame as
`H` (home) / `A` (away) / `D` (dead ball) / `X` (out of play).

Post-processing applies a boundary-weight penalty and a minimum-spell
filter to suppress sub-second label flickers without over-smoothing
possession transitions.

Training uses leave-one-out cross-validation on the three sample games.
No labels from the BAR-ATH game were used in training or tuning.

Final config: `--bw 5.0 --msf 50 --hidden-size 128 --seed 42`

---

## Reproduce

### Prerequisites

- Python 3.11
- [Task](https://taskfile.dev) runner — `brew install go-task` on macOS

### 1. Install dependencies

```bash
task setup
```

### 2. Download training data

Downloads the three Metrica public sample games from GitHub (~50 MB):

```bash
task sample-data
```

### 3. Generate corrected ground truth (one-time, safe to re-run)

```bash
python3.11 src/generate_gt.py
```

### 4. Place the BAR-ATH challenge data

Unpack the competition data bundle into `data/2027_challenge/`:

```
data/2027_challenge/
└─ tactical/
   ├─ BAR-ATH_tactical_atd.xml
   ├─ BAR-ATH_tactical_atd.txt
   ├─ BAR-ATH_tactical_homography.csv
   └─ BAR-ATH_tactical_smart_tagging.xml
```

Set `MTAH_DATA_DIR` to the parent of `tactical/` if you place it
elsewhere.

### 5. Predict (or reproduce from scratch)

The submission CSV and trained model are committed in the repo.
Run the validator in step 6 directly — no training required.

To regenerate the CSV from the saved model weights:

```bash
python3.11 src/sweep_challenge.py \
    --bw 5.0 --msf 50 \
    --hidden-size 128 --max-epochs 20 \
    --seed 42 --run-id v13 \
    --model-path models/diagonalmar.pt \
    --output-csv evaluation/results/diagonalmar.csv
```

The script loads `models/diagonalmar.pt` and skips training.

To retrain from scratch (5–10 min on CPU), delete the `.pt` file
first, then run the same command.

### 6. Validate

```bash
python3.11 evaluation/validate_submission.py \
    evaluation/results/diagonalmar.csv
```

Expected: `199319 rows, all labels valid`.

### 7. Score public window (reporting only)

`evaluation/gt/BAR-ATH_possession_gt_public.csv` is the
competition-supplied public BAR-ATH GT (199,319 rows, 7,500 non-X
scored frames: H:3454 A:2577 D:1469, D%=19.6%). It is committed in
the repo and is not produced or modified by `generate_gt.py`.

```bash
python3.11 evaluation/run_evaluation.py \
    --gt evaluation/gt/BAR-ATH_possession_gt_public.csv \
    --results evaluation/results \
    --no-detail
```

Public window spellF1: 0.4000 (macroF1 0.6310) — reporting only,
not used for config selection.

---

## Repository layout

```
src/
  sweep_challenge.py    entry point — trains and predicts in one pass
  pipeline.py           training loop and LOO harness
  features.py           feature extraction from tracking data
  possession_lstm.py    bidirectional LSTM model
  challenge_loader.py   loads BAR-ATH tactical ATD
  sample_loader.py      loads Metrica sample games
games.py                per-match constants and asset paths
evaluation/
  run_evaluation.py     scorer (spellF1, macroF1)
  validate_submission.py  pre-submit checks
  possession_metrics.py   metric library
requirements.txt        numpy, pandas (pinned)
requirements-starter.txt  full training stack (kloppy, pytorch-crf, …)
Taskfile.yml            task runner shortcuts
```

---

## Requirements

```
numpy==1.26.4
pandas==3.0.2
kloppy==3.18.0
pytorch-crf
matplotlib
```

Full pins: `requirements.txt` and `requirements-starter.txt`.
