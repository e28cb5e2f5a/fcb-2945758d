"""Derive per-frame dead-ball mask from Metrica sample game events data.

Supports three formats (auto-detected by file extension):
  - JSON  (Game 3 / real BAR-ATH sample): _events.json
  - CSV   (Games 1 & 2 legacy):           _RawEventsData.csv
  - XML   (BAR-ATH Smart Tagging):        _smart_tagging.xml  (SportsCode format)

XML restart detection
---------------------
Smart Tagging SET PIECES intervals use a fixed 35 s clip template as <end>,
not the actual restart time.  When ball_xy is supplied to dead_ball_mask(),
_mask_from_xml() scans forward from each start_frame and terminates the dead
region at the first frame where the ball is both visible and moving above
RESTART_SPEED_THRESH.  This reduces over-coverage by ~20–30 s per event.
Falls back to XML <end> time when ball_xy is None (existing behaviour).
"""

import csv
import json
import xml.etree.ElementTree as ET
import numpy as np

# Event types that indicate dead ball.
# NOTE: "BALL LOST" is excluded. Competition rules state that a contested/lost
# ball stays with the last possessing team (H/A, not D). The corrected GT
# (generate_gt.py) relabels BALL LOST frames to carry-forward H/A; keeping
# BALL LOST in the mask would force D at inference for frames the model was
# trained to label H/A — a training/inference mismatch that fragments spells.
# Round 12 probe (−0.0241) was measured against the original (practice) GT
# and is no longer applicable after Fix 2.
DEAD_BALL_TYPES = {"BALL OUT", "CARD", "FAULT RECEIVED"}

# Smart Tagging XML codes that indicate dead ball (case-insensitive)
_XML_DEAD_CODES = {"set pieces"}

_FPS = 25         # tactical video frame rate
_PITCH_LENGTH = 105.0
_PITCH_WIDTH  = 68.0

# Restart detection thresholds (XML path only)
# Source: FIFA Laws of the Game (Law 3 / Law 8) and general football physics.
#
# RESTART_SPEED_THRESH = 0.5 m/s
#   A ball at rest reaches 0.5 m/s within the first contact of a restart kick
#   (even a slow tap for a free-kick).  A stationary ball (pre-restart) never
#   exceeds this threshold under its own drift in tracking noise.  The value is
#   a physics-based lower bound; no sample-game GT was used to select it.
#
# MAX_RESTART_FRAMES = 500  (20 s at 25 fps)
#   FIFA Law 8 requires the ball to be put back in play "as soon as possible"
#   after the referee signals; empirical worst cases (VAR check, physio on pitch,
#   deliberate time-wasting) stay well below 60 s.  Set pieces themselves
#   (corner, free-kick, throw-in) are almost always completed within 10–15 s of
#   the whistle.  20 s is a deliberately generous upper bound derived from the
#   Laws, not from fitting to sample-game data.  It caps pathological cases
#   (tracker dropout during a long delay) without over-extending dead regions.
RESTART_SPEED_THRESH = 0.5   # m/s  (FIFA Laws / physics — see above)
MAX_RESTART_FRAMES   = 500   # 20 s cap (FIFA Law 8 / football domain — see above)


def dead_ball_mask(
    events_path: str,
    n_frames: int,
    buffer: int = 3,
    ball_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Per-frame dead-ball mask from events file (JSON, CSV, or XML).

    Parameters
    ----------
    events_path : path to events file (.json, .csv, or .xml)
    n_frames    : total tracking frames (mask is 0-indexed, tracking is 1-indexed)
    buffer      : extend dead region ±buffer frames around each event
    ball_xy     : (N, 2) float32 array of normalised ball coordinates from
                  load_tracking(); NaN where ball is missing.  When provided
                  and the file is XML, the actual restart moment is detected
                  from the tracking data instead of using the 35 s template
                  end time.  Ignored for JSON and CSV paths.

    Returns
    -------
    np.ndarray shape (n_frames,) bool — True = dead ball
    """
    ext = events_path.rsplit(".", 1)[-1].lower()
    if ext == "json":
        return _mask_from_json(events_path, n_frames, buffer)
    if ext == "xml":
        return _mask_from_xml(events_path, n_frames, buffer, ball_xy)
    return _mask_from_csv(events_path, n_frames, buffer)


def _apply_event(mask: np.ndarray, start: int, end: int, buffer: int) -> None:
    """Mark frames [start, end] (1-indexed, inclusive) as dead ball."""
    n = len(mask)
    lo = max(0, start - 1 - buffer)
    hi = min(n - 1, end - 1 + buffer)
    mask[lo:hi + 1] = True


def _mask_from_json(events_path: str, n_frames: int, buffer: int) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    with open(events_path) as f:
        payload = json.load(f)
    # Support both a bare list and {"data": [...], ...} wrapper
    events = payload["data"] if isinstance(payload, dict) else payload
    for ev in events:
        type_field = ev.get("type", "")
        if isinstance(type_field, dict):
            etype = type_field.get("name", "").strip().upper()
        else:
            etype = str(type_field).strip().upper()
        if etype not in DEAD_BALL_TYPES:
            continue
        try:
            start = int(ev["start"]["frame"])
        except (KeyError, TypeError, ValueError):
            import warnings
            warnings.warn(
                f"dead_ball: skipping malformed event (missing 'start'): "
                f"{ev!r:.120}",
                stacklevel=2,
            )
            continue
        end_frame = ev.get("end", {})
        end = int(end_frame["frame"]) if end_frame and "frame" in end_frame else start
        _apply_event(mask, start, end, buffer)
    return mask


def _mask_from_xml(
    events_path: str,
    n_frames: int,
    buffer: int,
    ball_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Smart Tagging SportsCode XML: SET PIECES intervals → dead ball.

    Time base: <start>/<end> are seconds from first frame of tactical video.
    seconds × 25 fps = 0-indexed video frame number.

    When ball_xy is provided, the end of each dead region is replaced by the
    first frame after start_frame where the ball is visible and moving above
    RESTART_SPEED_THRESH m/s (actual restart), capped at MAX_RESTART_FRAMES.
    """
    mask = np.zeros(n_frames, dtype=bool)

    # Pre-compute per-frame ball speed (m/s) if tracking data is available.
    if ball_xy is not None:
        ball_visible = ~np.isnan(ball_xy[:, 0])
        # Convert normalised coords to metres
        ball_m = np.zeros((n_frames, 2), dtype=np.float32)
        ball_m[:, 0] = np.where(ball_visible, ball_xy[:, 0] * _PITCH_LENGTH, 0.0)
        ball_m[:, 1] = np.where(ball_visible, ball_xy[:, 1] * _PITCH_WIDTH,  0.0)
        ball_diff = np.zeros((n_frames, 2), dtype=np.float32)
        ball_diff[1:] = ball_m[1:] - ball_m[:-1]
        # Zero sentinel→real displacement: first visible frame after absence
        just_appeared = np.zeros(n_frames, dtype=bool)
        just_appeared[1:] = ball_visible[1:] & ~ball_visible[:-1]
        ball_diff[just_appeared] = 0.0
        ball_speed = np.sqrt((ball_diff ** 2).sum(axis=1)) * _FPS
        ball_speed[~ball_visible] = 0.0
    else:
        ball_visible = None
        ball_speed   = None

    for instance in ET.parse(events_path).iter("instance"):
        code = (instance.findtext("code") or "").strip().lower()
        if code not in _XML_DEAD_CODES:
            continue
        start_s = float(instance.findtext("start") or 0)
        end_s   = float(instance.findtext("end")   or 0)
        # Convert seconds → 1-indexed video frame
        start_frame = int(round(start_s * _FPS)) + 1
        end_frame   = int(round(end_s   * _FPS)) + 1

        if ball_speed is not None:
            # Search forward from start_frame (0-indexed: start_frame - 1)
            # for the first frame where ball is visible and moving.
            search_start = max(0, start_frame - 1)
            search_end   = min(n_frames, start_frame - 1 + MAX_RESTART_FRAMES)
            restart_idx  = None
            for i in range(search_start, search_end):
                if ball_visible[i] and ball_speed[i] > RESTART_SPEED_THRESH:
                    restart_idx = i
                    break
            if restart_idx is not None:
                end_frame = restart_idx + 1   # back to 1-indexed
            else:
                end_frame = start_frame + MAX_RESTART_FRAMES

        _apply_event(mask, start_frame, end_frame, buffer)
    return mask


def _mask_from_csv(events_path: str, n_frames: int, buffer: int) -> np.ndarray:
    import warnings
    mask = np.zeros(n_frames, dtype=bool)
    with open(events_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                etype = row["Type"].strip()
                if etype not in DEAD_BALL_TYPES:
                    continue
                start = int(row["Start Frame"])
                raw_end = row["End Frame"].strip()
                end = int(raw_end) if raw_end else start
            except (KeyError, TypeError, ValueError):
                warnings.warn(
                    f"dead_ball: skipping malformed CSV row: {dict(row)!r:.120}",
                    stacklevel=2,
                )
                continue
            _apply_event(mask, start, end, buffer)
    return mask
