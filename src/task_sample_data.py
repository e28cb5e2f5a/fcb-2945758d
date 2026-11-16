#!/usr/bin/env python3
"""
Prepare Metrica sample data for all three games.

Game 3  — EPTS format (metadata.xml + tracking.txt + events.json)
Games 1 & 2 — CSV format  (_RawTrackingData_Home/Away_Team.csv + _RawEventsData.csv)

Downloads any missing files from GitHub, then generates possession GT CSVs.

Output:
  data/sample/Sample_Game_1/Sample_Game_1_possession_gt.csv
  data/sample/Sample_Game_2/Sample_Game_2_possession_gt.csv
  data/sample/Sample_Game_3/Sample_Game_3_possession_gt.csv

Columns: frame (0-indexed), label (H / A / D / X)
  H  home team has possession
  A  away team has possession
  D  dead ball (BALL OUT, BALL LOST, CHALLENGE, FAULT RECEIVED, CARD)
  X  ungraded / outside any half

Usage
-----
  python task_sample_data.py
"""

import csv
import json
import os
import urllib.request



_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT_DIR, "data", "sample")

_GITHUB_RAW = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"

_FILES = {
    "Sample_Game_1": [
        "Sample_Game_1/Sample_Game_1_RawEventsData.csv",
        "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
        "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
    ],
    "Sample_Game_2": [
        "Sample_Game_2/Sample_Game_2_RawEventsData.csv",
        "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
        "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
    ],
    "Sample_Game_3": [
        "Sample_Game_3/Sample_Game_3_events.json",
        "Sample_Game_3/Sample_Game_3_metadata.xml",
        "Sample_Game_3/Sample_Game_3_tracking.txt",
    ],
}

# Event types that indicate dead ball regardless of team
DEAD_BALL_TYPES = {"BALL OUT", "BALL LOST", "CHALLENGE", "FAULT RECEIVED", "CARD"}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_all():
    """Download any missing files for all three games."""
    all_files = [f for files in _FILES.values() for f in files]
    missing = [f for f in all_files
               if not os.path.isfile(os.path.join(DATA_DIR, f))]
    if not missing:
        print("All sample data files already present.")
        return
    print(f"Downloading {len(missing)} missing file(s) from GitHub...")
    for rel_path in missing:
        dest = os.path.join(DATA_DIR, rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = f"{_GITHUB_RAW}/{rel_path}"
        print(f"  {rel_path}")
        urllib.request.urlretrieve(url, dest)
    print("Download complete.\n")


# ---------------------------------------------------------------------------
# Game 3 — EPTS / JSON events
# ---------------------------------------------------------------------------

def convert_game3():
    """Build possession GT CSV for Sample Game 3 (EPTS/JSON format)."""
    game_name = "Sample_Game_3"
    game_dir  = os.path.join(DATA_DIR, game_name)

    events_path   = os.path.join(game_dir, f"{game_name}_events.json")
    tracking_path = os.path.join(game_dir, f"{game_name}_tracking.txt")

    with open(events_path) as f:
        raw_events = json.load(f)["data"]

    # Period boundaries from events
    periods = {}
    for ev in raw_events:
        p = ev["period"]
        sf, ef = ev["start"]["frame"], ev["end"]["frame"]
        if p not in periods:
            periods[p] = [sf, ef]
        else:
            periods[p][0] = min(periods[p][0], sf)
            periods[p][1] = max(periods[p][1], ef)

    # Total frame count = last frame number in tracking.txt
    with open(tracking_path) as f:
        last_line = None
        for last_line in f:
            pass
    n_frames = int(last_line.split(":")[0]) if last_line else max(
        v[1] for v in periods.values()
    )

    labels = ["X"] * n_frames

    events = []
    for ev in raw_events:
        etype = ev["type"]["name"]
        team  = ev["team"]["name"]
        if etype in DEAD_BALL_TYPES:
            label = "D"
        elif team == "Team A":
            label = "H"
        elif team == "Team B":
            label = "A"
        else:
            label = "X"
        events.append({"start": ev["start"]["frame"],
                        "end":   ev["end"]["frame"],
                        "label": label})

    _apply_events_and_ffill(labels, events, periods, n_frames)
    _write_gt(game_dir, game_name, labels)


# ---------------------------------------------------------------------------
# Games 1 & 2 — CSV events
# ---------------------------------------------------------------------------

def convert_game_csv(game_name: str):
    """Build possession GT CSV for a Game 1/2 style directory."""
    game_dir    = os.path.join(DATA_DIR, game_name)
    events_path = os.path.join(game_dir, f"{game_name}_RawEventsData.csv")
    home_path   = os.path.join(game_dir, f"{game_name}_RawTrackingData_Home_Team.csv")

    # Period boundaries + frame count from tracking CSV
    periods = {}
    n_frames = 0
    with open(home_path, newline="") as f:
        reader = csv.reader(f)
        next(reader); next(reader); next(reader)  # 3 header rows
        for row in reader:
            if not row or not row[0].strip():
                continue
            period = int(row[0])
            frame  = int(row[1])
            n_frames = max(n_frames, frame)
            if period not in periods:
                periods[period] = [frame, frame]
            else:
                periods[period][0] = min(periods[period][0], frame)
                periods[period][1] = max(periods[period][1], frame)

    labels = ["X"] * n_frames

    events = []
    with open(events_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            etype = row["Type"].strip()
            team  = row["Team"].strip()
            start = int(row["Start Frame"])
            end_f = row["End Frame"].strip()
            end   = int(end_f) if end_f and end_f != "0" else start

            if etype in DEAD_BALL_TYPES:
                label = "D"
            elif team == "Home":
                label = "H"
            elif team == "Away":
                label = "A"
            else:
                label = "X"

            events.append({"start": start, "end": end, "label": label})

    _apply_events_and_ffill(labels, events, periods, n_frames)
    _write_gt(game_dir, game_name, labels)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _apply_events_and_ffill(labels, events, periods, n_frames):
    """Stamp event labels into the array, then forward-fill within each period."""
    events.sort(key=lambda e: e["start"])
    for ev in events:
        for f in range(ev["start"], ev["end"] + 1):
            if 1 <= f <= n_frames:
                labels[f - 1] = ev["label"]

    for first, last in periods.values():
        current = "X"
        for f in range(first, last + 1):
            idx = f - 1
            if idx >= len(labels):
                break
            if labels[idx] != "X":
                current = labels[idx]
            elif current != "X":
                labels[idx] = current


def _write_gt(game_dir: str, game_name: str, labels: list):
    out_path = os.path.join(game_dir, f"{game_name}_possession_gt.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "label"])
        for i, label in enumerate(labels):
            writer.writerow([i, label])

    counts = {l: labels.count(l) for l in ("H", "A", "D", "X")}
    print(f"{game_name}")
    print(f"  {len(labels):>7,} rows  H={counts['H']:,}  A={counts['A']:,}  "
          f"D={counts['D']:,}  X={counts['X']:,}")
    print(f"  -> {out_path}")
    _write_public_slice(game_name, labels)


def _write_public_slice(game_name: str, labels: list, public_frames: int = 7500):
    """Write evaluation/gt/<game>_possession_gt_public.csv covering first 5 min."""
    eval_gt_dir = os.path.join(
        _ROOT_DIR, "evaluation", "gt"
    )
    os.makedirs(eval_gt_dir, exist_ok=True)

    n_total = len(labels)
    public_labels = ["X"] * n_total
    for i in range(min(public_frames, n_total)):
        public_labels[i] = labels[i]

    out_csv = os.path.join(eval_gt_dir, f"{game_name}_possession_gt_public.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "label"])
        for i, label in enumerate(public_labels):
            writer.writerow([i, label])

    # Count possession spells in window
    spells, prev = 0, None
    for l in public_labels[:public_frames]:
        if l in ("H", "A") and l != prev:
            spells += 1
        prev = l

    meta = {
        "game": game_name,
        "fps": 25,
        "n_video_frames": n_total,
        "n_graded_frames": public_frames,
        "n_possessions": spells,
        "public_window": {
            "first_video_frame": 0,
            "last_video_frame": public_frames - 1,
            "minutes_of_play": round(public_frames / 25 / 60, 1),
            "n_frames": public_frames,
            "note": (
                f"Real labels inside this window only. Every other frame is X "
                f"and is never scored. Submissions must still cover all {n_total} frames."
            ),
        },
        "label_counts_in_window": {
            l: public_labels[:public_frames].count(l) for l in ("H", "A", "D", "X")
        },
    }
    out_meta = out_csv.replace(".csv", "_meta.json")
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  public slice -> {out_csv}  ({spells} spells)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    download_all()
    convert_game_csv("Sample_Game_1")
    print()
    convert_game_csv("Sample_Game_2")
    print()
    convert_game3()
    print("\nDone.")
