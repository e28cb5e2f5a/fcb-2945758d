# Reproducing the submission

## Prerequisites

- Python 3.11
- [Task](https://taskfile.dev) runner — `brew install go-task` on macOS

## Steps

### 1. Install dependencies

```
task setup
```

### 2. Download training data

The model trains on the three publicly available Metrica sample games.
This downloads them from GitHub (~50 MB total) and builds the possession
GT CSVs:

```
task sample-data
```

Output: `data/sample/Sample_Game_{1,2,3}/`

### 3. Place the BAR-ATH challenge data

Unpack the competition data bundle (provided by the organisers) into
`data/2027_challenge/`, keeping the names it arrives with:

```
data/2027_challenge/
└─ tactical/
   ├─ BAR-ATH_tactical_atd.xml
   ├─ BAR-ATH_tactical_atd.txt
   ├─ BAR-ATH_tactical_homography.csv
   └─ BAR-ATH_tactical_smart_tagging.xml
```

If you place it elsewhere, set `MTAH_DATA_DIR` to the parent of `tactical/`.

### 4. Regenerate the submission

```
python3.11 src/sweep_challenge.py \
    --bw 5.0 --msf 50 \
    --hidden-size 128 --max-epochs 20 \
    --seed 42 --run-id v13 \
    --model-path models/diagonalmar.pt \
    --output-csv evaluation/results/diagonalmar.csv
```

The script loads `models/diagonalmar.pt` and skips training.
Training from scratch takes roughly 5–10 minutes on a modern laptop (CPU only).

### 5. Validate

```
python3.11 evaluation/validate_submission.py \
    evaluation/results/diagonalmar.csv
```

Expected: 199,319 rows, labels `H`/`A`/`D`/`X`, no blank cells.
