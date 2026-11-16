"""Single-command BAR-ATH submission script.

Usage
-----
    python3.11 src/predict_bar_ath.py \
        --data-root data/tactical \
        --model models/lstm_v2.pt \
        --out evaluation/results/bar_ath_submission.csv

For dry-run testing on sample data:
    python3.11 src/predict_bar_ath.py \
        --data-root data/sample/Sample_Game_3 \
        --model models/lstm_v2.pt \
        --out evaluation/results/bar_ath_dryrun.csv
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.sample_loader import load_tracking
from src.features import extract_features
from src.dead_ball import dead_ball_mask
from src.possession_lstm import PossessionLSTM


def _events_path(data_root: str) -> str:
    """Return the events file path for a game directory (JSON or XML/CSV)."""
    game_name = os.path.basename(data_root.rstrip("/"))
    # JSON (Sample_Game_3 style)
    json_path = os.path.join(data_root, f"{game_name}_events.json")
    if os.path.isfile(json_path):
        return json_path
    # Smart Tagging XML
    xml_path = os.path.join(data_root, f"{game_name}_SmartTagging.xml")
    if os.path.isfile(xml_path):
        return xml_path
    # Raw events CSV fallback
    csv_path = os.path.join(data_root, f"{game_name}_RawEventsData.csv")
    if os.path.isfile(csv_path):
        return csv_path
    raise FileNotFoundError(
        f"No events file found in {data_root}. "
        "Expected *_events.json, *_SmartTagging.xml, or *_RawEventsData.csv."
    )


def _apply_graded_blocks(labels: np.ndarray, game_id: str) -> np.ndarray:
    """Set frames outside graded half-blocks to 'X'. No-op if game_id not found."""
    try:
        from games import half_video_blocks
        blocks = half_video_blocks(game_id)
        out = np.full(len(labels), "X", dtype="<U1")
        for _, first, last in blocks:
            out[first:last + 1] = labels[first:last + 1]
        print(f"  Applied graded blocks ({len(blocks)} halves)")
        return out
    except Exception:
        return labels


from src.pipeline import clamp_d_to_mask as _clamp_d_to_mask


def _write_csv(path: str, labels: np.ndarray) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "label"])
        for i, label in enumerate(labels):
            writer.writerow([i, label])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True,
                        help="directory containing tracking + events files")
    parser.add_argument("--model", required=True, metavar="PATH",
                        help="path to saved lstm_v2.pt model")
    parser.add_argument("--out", required=True, metavar="PATH",
                        help="output CSV path")
    args = parser.parse_args()

    data_root = args.data_root.rstrip("/")
    game_id = os.path.basename(data_root)

    # 1. Load tracking
    print(f"Loading tracking from {data_root}...")
    data = load_tracking(data_root)
    feats = extract_features(data)
    n_frames = feats.shape[0]
    print(f"  {n_frames:,} frames, {feats.shape[1]} features")

    # 2. Load dead-ball mask
    events_file = _events_path(data_root)
    print(f"Loading dead-ball mask from {os.path.basename(events_file)}...")
    dmask = dead_ball_mask(events_file, n_frames=n_frames, ball_xy=data["ball_xy"])
    print(f"  dead frames: {dmask.sum():,} / {n_frames:,} ({100*dmask.mean():.1f}%)")

    # 3. Load model
    print(f"Loading model from {args.model}...")
    model = PossessionLSTM.load(args.model)

    # 4. Predict
    print("Predicting...")
    labels = model.predict(feats, dmask)
    before_d = int((labels == "D").sum())
    labels = _clamp_d_to_mask(labels, dmask, ball_xy=data["ball_xy"])
    after_d = int((labels == "D").sum())
    print(f"  Clamp: D {before_d:,} → {after_d:,} "
          f"({before_d - after_d:,} frames converted to H/A)")

    # 5. Apply X masking for frames outside graded half-blocks
    labels = _apply_graded_blocks(labels, game_id)

    # 6. Write CSV
    _write_csv(args.out, labels)

    total = len(labels)
    counts = {l: int((labels == l).sum()) for l in ("H", "A", "D", "X")}
    print(f"\nWritten: {args.out}")
    print(f"  Rows: {total:,}")
    print(f"  H={counts['H']:,} ({100*counts['H']/total:.1f}%)  "
          f"A={counts['A']:,} ({100*counts['A']/total:.1f}%)  "
          f"D={counts['D']:,} ({100*counts['D']/total:.1f}%)  "
          f"X={counts['X']:,} ({100*counts['X']/total:.1f}%)")


if __name__ == "__main__":
    main()
