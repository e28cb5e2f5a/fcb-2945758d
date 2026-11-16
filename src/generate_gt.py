"""Regenerate possession GT CSVs from competition rules.

Competition rules:
  - BALL LOST, CHALLENGE → last possessing team keeps ball (H or A)
  - BALL OUT, FAULT RECEIVED, CARD → game stopped (D)
  - SET PIECE → game stopped (D) until restart

Practice GT mislabels BALL LOST and CHALLENGE as D. This script reads the
original GT CSV and events file, applies the carry-forward rule for contested
frames, and writes a corrected GT CSV alongside the original (never overwrites).

Usage
-----
    python3.11 src/generate_gt.py           # corrects all three sample games
    python3.11 src/generate_gt.py --dry-run # print stats only, write nothing
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.gt_relabel import relabel_contested

SAMPLE_DIRS = [
    os.path.join(ROOT, "data", "sample", "Sample_Game_1"),
    os.path.join(ROOT, "data", "sample", "Sample_Game_2"),
    os.path.join(ROOT, "data", "sample", "Sample_Game_3"),
]

EVAL_GT_DIR = os.path.join(ROOT, "evaluation", "gt")


def _events_path(game_dir: str) -> str:
    game_name = os.path.basename(game_dir)
    json_path = os.path.join(game_dir, f"{game_name}_events.json")
    if os.path.isfile(json_path):
        return json_path
    return os.path.join(game_dir, f"{game_name}_RawEventsData.csv")


def _corrected_path(original_path: str) -> str:
    base, ext = os.path.splitext(original_path)
    return base + "_corrected" + ext


def correct_game(game_dir: str, dry_run: bool = False) -> dict:
    """Load original GT, apply competition rules, write corrected GT.

    Returns stats dict.
    """
    game_name = os.path.basename(game_dir)
    gt_path = os.path.join(game_dir, f"{game_name}_possession_gt.csv")
    events_path = _events_path(game_dir)

    # Load original GT
    with open(gt_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # GT CSV frames are 0-indexed; event frames are 1-indexed (tracking convention).
    # Add 1 to convert so relabel_contested frame lookups align correctly.
    frames = np.array([int(r[0]) + 1 for r in rows], dtype=np.int32)
    labels = np.array([r[1] for r in rows], dtype="<U1")

    # Apply correction
    corrected = relabel_contested(labels, events_path, frames)

    changed = (labels != corrected).sum()
    d_before = (labels == "D").sum()
    d_after = (corrected == "D").sum()

    stats = {
        "game": game_name,
        "total_frames": len(frames),
        "changed": int(changed),
        "d_before": int(d_before),
        "d_after": int(d_after),
    }

    if not dry_run:
        out_path = _corrected_path(gt_path)
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i, row in enumerate(rows):
                writer.writerow([row[0], corrected[i]])
        stats["written"] = out_path

    return stats


def correct_public_eval_gt(game_name: str, dry_run: bool = False) -> dict | None:
    """Regenerate the public evaluation GT window using the corrected full GT.

    Reads the corrected full GT, then extracts only the frames present in the
    original public GT file, applying corrected labels.
    """
    game_dir = None
    for d in SAMPLE_DIRS:
        if os.path.basename(d) == game_name:
            game_dir = d
            break
    if game_dir is None:
        return None

    corrected_full = _corrected_path(
        os.path.join(game_dir, f"{game_name}_possession_gt.csv")
    )
    if not os.path.isfile(corrected_full):
        return None

    # Load corrected full GT as frame→label map
    with open(corrected_full, newline="") as f:
        corrected_map = {int(r["frame"]): r["label"] for r in csv.DictReader(f)}

    # Load original public GT
    pub_path = os.path.join(EVAL_GT_DIR, f"{game_name}_possession_gt_public.csv")
    if not os.path.isfile(pub_path):
        return None

    with open(pub_path, newline="") as f:
        reader = csv.DictReader(f)
        pub_rows = list(reader)

    changed = sum(
        1 for r in pub_rows
        if r["label"] != "X" and corrected_map.get(int(r["frame"]), r["label"]) != r["label"]
    )

    if not dry_run:
        out_path = _corrected_path(pub_path)
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "label"])
            for r in pub_rows:
                frame = int(r["frame"])
                orig_label = r["label"]
                new_label = corrected_map.get(frame, orig_label) if orig_label != "X" else "X"
                writer.writerow([frame, new_label])
        return {"game": game_name, "changed": changed, "written": out_path}

    return {"game": game_name, "changed": changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="print stats only, write nothing")
    args = parser.parse_args()

    print("Correcting full GT files...")
    for game_dir in SAMPLE_DIRS:
        stats = correct_game(game_dir, dry_run=args.dry_run)
        print(f"  {stats['game']}: {stats['changed']:,} frames changed  "
              f"D {stats['d_before']:,} → {stats['d_after']:,}"
              + (f"  → {stats['written']}" if "written" in stats else " (dry run)"))

    print("\nCorrecting public evaluation GT files...")
    for game_dir in SAMPLE_DIRS:
        game_name = os.path.basename(game_dir)
        result = correct_public_eval_gt(game_name, dry_run=args.dry_run)
        if result:
            print(f"  {result['game']}: {result['changed']:,} frames changed"
                  + (f"  → {result['written']}" if "written" in result else " (dry run)"))
        else:
            print(f"  {game_name}: skipped (corrected full GT not found or no public GT)")


if __name__ == "__main__":
    main()
