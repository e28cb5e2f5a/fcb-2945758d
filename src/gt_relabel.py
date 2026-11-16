"""Relabel contested-ball frames in training GT to match competition rules.

Competition rules state that a ball involved in a duel or lost ball
situation stays with the last possessing team (H or A), not D.
Practice GT labels BALL LOST and CHALLENGE events as D — this module
corrects that in memory before training, without modifying GT files.

Only D frames inside BALL LOST / CHALLENGE event intervals are changed.
Non-D frames and all other event types are left untouched.
"""

import csv
import json
import numpy as np

# Events whose D labels should become the last observed H/A (carry-forward)
_RELABEL_TYPES = {"BALL LOST", "CHALLENGE"}
_FALLBACK_LABEL = "H"   # used when no prior H/A exists (very start of match)


def relabel_contested(
    gt: np.ndarray,
    events_path: str,
    frames: np.ndarray,
) -> np.ndarray:
    """Replace D labels inside BALL LOST/CHALLENGE intervals with carry-forward H/A.

    Parameters
    ----------
    gt          : (N,) array of str labels H/A/D — training GT from load_gt()
    events_path : path to events file (.json or .csv)
    frames      : (N,) int32 array of 1-indexed frame numbers from load_tracking()

    Returns
    -------
    (N,) array of same dtype; D inside contested intervals replaced with
    the last H/A label observed before the interval, or 'H' if none exists.
    GT files on disk are never modified.
    """
    intervals = _load_intervals(events_path)
    if not intervals:
        return gt

    # Build frame → index lookup (frames are 1-indexed, may have gaps)
    frame_to_idx = {int(f): i for i, f in enumerate(frames)}
    n = len(gt)

    gt_out = gt.copy()

    for start_f, end_f in intervals:
        # Find last H/A label before this interval
        carry = _FALLBACK_LABEL
        for f in range(start_f - 1, 0, -1):
            idx = frame_to_idx.get(f)
            if idx is not None and gt_out[idx] in ("H", "A"):
                carry = gt_out[idx]
                break

        # Relabel D frames inside [start_f, end_f]
        for f in range(start_f, end_f + 1):
            idx = frame_to_idx.get(f)
            if idx is not None and gt_out[idx] == "D":
                gt_out[idx] = carry

    return gt_out


def _load_intervals(events_path: str) -> list:
    """Return list of (start_frame, end_frame) 1-indexed pairs for relabelled types."""
    ext = events_path.rsplit(".", 1)[-1].lower()
    if ext == "json":
        return _intervals_from_json(events_path)
    return _intervals_from_csv(events_path)


def _intervals_from_json(events_path: str) -> list:
    intervals = []
    with open(events_path) as f:
        payload = json.load(f)
    events = payload["data"] if isinstance(payload, dict) else payload
    for ev in events:
        type_field = ev.get("type", "")
        if isinstance(type_field, dict):
            name = type_field.get("name", "").strip().upper()
        else:
            name = str(type_field).strip().upper()
        if name not in _RELABEL_TYPES:
            continue
        try:
            start = int(ev["start"]["frame"])
            end_d = ev.get("end", {})
            end = int(end_d["frame"]) if end_d and "frame" in end_d else start
            intervals.append((start, end))
        except (KeyError, TypeError, ValueError):
            continue
    return intervals


def _intervals_from_csv(events_path: str) -> list:
    intervals = []
    with open(events_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if row["Type"].strip().upper() not in _RELABEL_TYPES:
                    continue
                start = int(row["Start Frame"])
                raw_end = row["End Frame"].strip()
                end = int(raw_end) if raw_end else start
                intervals.append((start, end))
            except (KeyError, TypeError, ValueError):
                continue
    return intervals
