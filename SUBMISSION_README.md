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

**Locked config (v13)**

| parameter | value |
|---|---|
| boundary weight | 5.0 |
| min spell frames | 50 |
| hidden size | 128 |
| epochs | 20 |
| seed | 42 |
| features | 18 (tracking-proxy; no ball interpolation) |
| corrected GT | yes (`generate_gt.py`; GT frame +1 offset fix) |
| BALL LOST in dead ball | excluded |
| visible-ball clamp | threshold-free |
| Game 3 LOO spellF1 | 0.6000 |

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

Downloads the three Metrica public sample games from GitHub (~50 MB)
and builds corrected possession GT CSVs:

```bash
task sample-data
```

### 3. Place the BAR-ATH challenge data

Unpack the competition data bundle into `data/2027_challenge/`:

```
data/2027_challenge/
└─ tactical/
   ├─ BAR-ATH_tactical_atd.xml
   ├─ BAR-ATH_tactical_atd.txt
   └─ BAR-ATH_tactical_smart_tagging.xml
```

Set `MTAH_DATA_DIR` to the parent of `tactical/` if you place it
elsewhere.

### 4. Predict (or reproduce from scratch)

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

### 5. Verify row count

```bash
python3.11 -c "
import csv
with open('evaluation/results/diagonalmar.csv') as f:
    rows = sum(1 for _ in csv.reader(f)) - 1
assert rows == 199319, f'Expected 199319 rows, got {rows}'
print(f'OK: {rows} rows')
"
```

### 6. Validate submission file

```bash
python3.11 evaluation/validate_submission.py \
    evaluation/results/diagonalmar.csv
```

### 7. Score public window (reporting only)

`evaluation/gt/BAR-ATH_possession_gt_public.csv` is the
competition-supplied public GT (199,319 rows; 7,500 graded frames:
H:3454 A:2577 D:1469). It is committed in this repository and is not
modified by any pipeline step.

```bash
python3.11 evaluation/run_evaluation.py \
    --gt evaluation/gt/BAR-ATH_possession_gt_public.csv \
    --results evaluation/results \
    --no-detail
```

Expected: spellF1 **0.4000**, macroF1 **0.6310** (reporting only —
public window is 5 minutes; final ranking uses the full match).

---

## Repository layout

```
src/
  sweep_challenge.py      entry point — trains and predicts in one pass
  pipeline.py             training loop and LOO harness
  features.py             feature extraction from tracking data
  possession_lstm.py      bidirectional LSTM model
  challenge_loader.py     loads BAR-ATH tactical ATD
  sample_loader.py        loads Metrica sample games
  generate_gt.py          builds corrected possession GT from sample events
games.py                  per-match constants and asset paths
evaluation/
  run_evaluation.py       scorer (spellF1, macroF1)
  validate_submission.py  pre-submit checks
  possession_metrics.py   metric library
  gt/                     public 5-minute BAR-ATH ground truth
requirements.txt          numpy, pandas (pinned)
requirements-starter.txt  full training stack (kloppy, pytorch-crf, …)
Taskfile.yml              task runner shortcuts
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
