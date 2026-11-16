"""Tests for src/gt_relabel.relabel_contested()."""
import csv
import json
import numpy as np
import pytest


def _make_json(events, tmp_path):
    p = tmp_path / "events.json"
    p.write_text(json.dumps({"data": events}))
    return str(p)


def _make_csv(rows, tmp_path):
    p = tmp_path / "events.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Type", "Start Frame", "End Frame"])
        w.writerows(rows)
    return str(p)


def _gt(*labels):
    return np.array(list(labels), dtype="<U1")


def _frames(n, start=1):
    return np.arange(start, start + n, dtype=np.int32)


# ---------------------------------------------------------------------------
# JSON path
# ---------------------------------------------------------------------------

def test_ball_lost_d_frames_become_carry_forward(tmp_path):
    """D frames inside BALL LOST get the last H/A before the event."""
    from src.gt_relabel import relabel_contested
    # frame 1=H, 2=H, 3=D(BALL LOST), 4=D(BALL LOST), 5=A
    gt = _gt("H", "H", "D", "D", "A")
    events = _make_json([
        {"type": {"name": "BALL LOST"}, "start": {"frame": 3}, "end": {"frame": 4}},
    ], tmp_path)
    out = relabel_contested(gt, events, _frames(5))
    assert list(out) == ["H", "H", "H", "H", "A"]


def test_challenge_d_frames_become_carry_forward(tmp_path):
    from src.gt_relabel import relabel_contested
    gt = _gt("A", "D", "D", "H")
    events = _make_json([
        {"type": {"name": "CHALLENGE"}, "start": {"frame": 2}, "end": {"frame": 3}},
    ], tmp_path)
    out = relabel_contested(gt, events, _frames(4))
    assert list(out) == ["A", "A", "A", "H"]


def test_non_d_frames_inside_interval_unchanged(tmp_path):
    """H/A frames inside a CHALLENGE interval must not be overwritten."""
    from src.gt_relabel import relabel_contested
    gt = _gt("H", "A", "D", "H")   # frame 3 is D, frame 2 is already A
    events = _make_json([
        {"type": {"name": "CHALLENGE"}, "start": {"frame": 2}, "end": {"frame": 3}},
    ], tmp_path)
    out = relabel_contested(gt, events, _frames(4))
    # frame 2 was A → unchanged; frame 3 was D → carry-forward from frame 1 (H)
    assert out[1] == "A"
    assert out[2] == "H"


def test_d_outside_interval_unchanged(tmp_path):
    """D frames outside any relabelled interval must stay D."""
    from src.gt_relabel import relabel_contested
    gt = _gt("H", "D", "D", "D", "H")
    events = _make_json([
        {"type": {"name": "BALL LOST"}, "start": {"frame": 2}, "end": {"frame": 2}},
    ], tmp_path)
    out = relabel_contested(gt, events, _frames(5))
    # frame 3 and 4 are D but outside interval — must stay D
    assert out[2] == "D"
    assert out[3] == "D"


def test_fallback_h_when_no_prior_possession(tmp_path):
    """If no H/A exists before the interval, fallback to 'H'."""
    from src.gt_relabel import relabel_contested
    gt = _gt("D", "D", "D", "H")
    events = _make_json([
        {"type": {"name": "CHALLENGE"}, "start": {"frame": 1}, "end": {"frame": 2}},
    ], tmp_path)
    out = relabel_contested(gt, events, _frames(4))
    assert out[0] == "H"
    assert out[1] == "H"
    assert out[2] == "D"   # outside interval


def test_other_event_types_ignored(tmp_path):
    """BALL OUT / PASS events must not trigger relabelling."""
    from src.gt_relabel import relabel_contested
    gt = _gt("H", "D", "D", "H")
    events = _make_json([
        {"type": {"name": "BALL OUT"}, "start": {"frame": 2}, "end": {"frame": 3}},
        {"type": {"name": "PASS"}, "start": {"frame": 2}, "end": {"frame": 3}},
    ], tmp_path)
    out = relabel_contested(gt, events, _frames(4))
    assert list(out) == ["H", "D", "D", "H"]   # unchanged


def test_no_events_returns_original(tmp_path):
    from src.gt_relabel import relabel_contested
    gt = _gt("H", "D", "A")
    events = _make_json([], tmp_path)
    out = relabel_contested(gt, events, _frames(3))
    np.testing.assert_array_equal(out, gt)


# ---------------------------------------------------------------------------
# CSV path
# ---------------------------------------------------------------------------

def test_csv_ball_lost_relabelled(tmp_path):
    from src.gt_relabel import relabel_contested
    gt = _gt("A", "D", "D", "H")
    events = _make_csv([
        ["BALL LOST", "2", "3"],
        ["PASS", "1", "1"],
    ], tmp_path)
    out = relabel_contested(gt, events, _frames(4))
    assert list(out) == ["A", "A", "A", "H"]


def test_csv_challenge_relabelled(tmp_path):
    from src.gt_relabel import relabel_contested
    gt = _gt("H", "H", "D", "A")
    events = _make_csv([["CHALLENGE", "3", "3"]], tmp_path)
    out = relabel_contested(gt, events, _frames(4))
    assert out[2] == "H"


def test_original_array_not_mutated(tmp_path):
    """relabel_contested must return a copy, not mutate the input."""
    from src.gt_relabel import relabel_contested
    gt = _gt("H", "D", "D")
    events = _make_json([
        {"type": {"name": "BALL LOST"}, "start": {"frame": 2}, "end": {"frame": 3}},
    ], tmp_path)
    out = relabel_contested(gt, events, _frames(3))
    assert gt[1] == "D"   # original unchanged
    assert out[1] == "H"
