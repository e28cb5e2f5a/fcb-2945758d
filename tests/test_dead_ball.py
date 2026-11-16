import csv as csv_mod
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# CSV (Legacy RawEventsData) fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth_csv_events_path(tmp_path_factory):
    """Minimal RawEventsData CSV with one dead-ball event (BALL OUT)."""
    p = tmp_path_factory.mktemp("csv_events") / "events.csv"
    rows = [
        ["Type", "Start Frame", "End Frame"],
        ["BALL OUT", "50", "60"],
        ["FAULT RECEIVED", "200", "210"],
        ["PASS", "300", "310"],
    ]
    with open(p, "w", newline="") as f:
        w = csv_mod.writer(f)
        for row in rows:
            w.writerow(row)
    return str(p)


# ---------------------------------------------------------------------------
# CSV dead-ball tests
# ---------------------------------------------------------------------------

def test_csv_dead_ball_mask_shape(synth_csv_events_path):
    from src.dead_ball import dead_ball_mask
    mask = dead_ball_mask(synth_csv_events_path, n_frames=500)
    assert mask.shape == (500,)
    assert mask.dtype == bool


def test_csv_dead_ball_marks_event(synth_csv_events_path):
    from src.dead_ball import dead_ball_mask
    mask = dead_ball_mask(synth_csv_events_path, n_frames=500, buffer=0)
    # BALL OUT at 1-indexed frames 50-60 → 0-indexed 49-59.
    # _apply_event: lo = start-1 = 49, hi = end-1 = 59 → mask[49:60] is set.
    assert mask[49:60].all(), "BALL OUT interval not marked dead"
    # At least one frame should be dead
    assert mask.sum() > 0


def test_csv_dead_ball_non_dead_event_ignored(synth_csv_events_path):
    from src.dead_ball import dead_ball_mask
    mask = dead_ball_mask(synth_csv_events_path, n_frames=500, buffer=0)
    # PASS at frames 300-310 is not a dead-ball type — must not be masked
    assert not mask[299:311].all(), "PASS interval incorrectly marked as dead"


# ---------------------------------------------------------------------------
# XML (Smart Tagging) fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth_xml_path(tmp_path_factory):
    """Minimal SportsCode XML with two SET PIECES (dead) and one BUILD UP (alive)."""
    xml = """<?xml version="1.0" encoding="utf-8"?>
<file>
  <instance>
    <code>SET PIECES</code>
    <start>2.0</start>
    <end>4.0</end>
    <label><group>Team</group><text>Home Team</text></label>
    <label><group>Type</group><text>Kick Off (Start)</text></label>
    <label><group>Half</group><text>1st Half</text></label>
  </instance>
  <instance>
    <code>SET PIECES</code>
    <start>8.0</start>
    <end>9.0</end>
    <label><group>Team</group><text>Away Team</text></label>
    <label><group>Type</group><text>Free Kick</text></label>
    <label><group>Half</group><text>1st Half</text></label>
  </instance>
  <instance>
    <code>BUILD UP</code>
    <start>5.0</start>
    <end>7.0</end>
    <label><group>Team</group><text>Home Team</text></label>
    <label><group>Half</group><text>1st Half</text></label>
  </instance>
</file>
"""
    p = tmp_path_factory.mktemp("xml_events") / "smart_tagging.xml"
    p.write_text(xml)
    return str(p)


# ---------------------------------------------------------------------------
# XML dead-ball tests
# ---------------------------------------------------------------------------

def test_xml_dead_ball_mask_shape(synth_xml_path):
    from src.dead_ball import dead_ball_mask
    mask = dead_ball_mask(synth_xml_path, n_frames=500)
    assert mask.shape == (500,)
    assert mask.dtype == bool


def test_xml_dead_ball_marks_set_pieces(synth_xml_path):
    """SET PIECES intervals (frames 50-100 and 200-225 at 25fps) must be dead."""
    from src.dead_ball import dead_ball_mask
    mask = dead_ball_mask(synth_xml_path, n_frames=500, buffer=0)
    # 2.0s * 25fps = frame 50, 4.0s = frame 100
    assert mask[50:101].all(), "First SET PIECE interval not marked dead"
    # 8.0s = 200, 9.0s = 225
    assert mask[200:226].all(), "Second SET PIECE interval not marked dead"


def test_xml_dead_ball_build_up_is_alive(synth_xml_path):
    """BUILD UP is not a dead-ball event — must not be masked."""
    from src.dead_ball import dead_ball_mask
    mask = dead_ball_mask(synth_xml_path, n_frames=500, buffer=0)
    # 5.0s-7.0s = frames 125-175, no overlap with SET PIECES
    assert not mask[140:160].all(), "BUILD UP interval incorrectly marked as dead"


def test_xml_dead_ball_buffer_extends(synth_xml_path):
    from src.dead_ball import dead_ball_mask
    mask0 = dead_ball_mask(synth_xml_path, n_frames=500, buffer=0)
    mask3 = dead_ball_mask(synth_xml_path, n_frames=500, buffer=3)
    assert mask3.sum() >= mask0.sum()


# ---------------------------------------------------------------------------
# Original JSON/CSV tests
# ---------------------------------------------------------------------------

def test_dead_ball_mask_shape(synth_events_path, synth_data):
    from src.dead_ball import dead_ball_mask
    n = len(synth_data["frames"])
    mask = dead_ball_mask(synth_events_path, n_frames=n)
    assert mask.shape == (n,)
    assert mask.dtype == bool


def test_dead_ball_has_some_dead_frames(synth_events_path, synth_data):
    from src.dead_ball import dead_ball_mask
    n = len(synth_data["frames"])
    mask = dead_ball_mask(synth_events_path, n_frames=n)
    assert mask.sum() > 0


def test_dead_ball_not_all_dead(synth_events_path, synth_data):
    from src.dead_ball import dead_ball_mask
    n = len(synth_data["frames"])
    mask = dead_ball_mask(synth_events_path, n_frames=n)
    assert (~mask).sum() > 0


def test_dead_ball_buffer_extends_mask(synth_events_path, synth_data):
    from src.dead_ball import dead_ball_mask
    n = len(synth_data["frames"])
    mask_no_buf = dead_ball_mask(synth_events_path, n_frames=n, buffer=0)
    mask_buf    = dead_ball_mask(synth_events_path, n_frames=n, buffer=3)
    assert mask_buf.sum() >= mask_no_buf.sum()


def test_dead_ball_within_bounds(synth_events_path, synth_data):
    from src.dead_ball import dead_ball_mask
    n = len(synth_data["frames"])
    mask = dead_ball_mask(synth_events_path, n_frames=n)
    assert len(mask) == n


# ---------------------------------------------------------------------------
# XML restart-detection tests (ball_xy supplied)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth_xml_long_path(tmp_path_factory):
    """XML with one 35 s SET PIECES event (template window) and one BUILD UP."""
    xml = """<?xml version="1.0" encoding="utf-8"?>
<file>
  <instance>
    <code>SET PIECES</code>
    <start>2.0</start>
    <end>37.0</end>
    <label><group>Type</group><text>Throw-in</text></label>
  </instance>
  <instance>
    <code>BUILD UP</code>
    <start>5.0</start>
    <end>7.0</end>
  </instance>
</file>
"""
    p = tmp_path_factory.mktemp("xml_long") / "smart_tagging.xml"
    p.write_text(xml)
    return str(p)


def test_restart_detection_shortens_mask(synth_xml_long_path):
    """With ball_xy, dead region ends at restart, not at 35 s template."""
    from src.dead_ball import dead_ball_mask, _FPS, RESTART_SPEED_THRESH
    import src.dead_ball as _db

    n = 1500
    # start_s=2.0 → start_frame = int(round(2.0*25))+1 = 51  (0-indexed: 50)
    # Ball is missing for frames 0-74, then becomes visible and moving at 75.
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)
    # Ball appears at frame 75 (0-indexed) and is stationary for 2 frames,
    # then moves fast enough at frame 77.
    ball_xy[75:, 0] = 0.5
    ball_xy[75:, 1] = 0.5
    # Give frame 77 a large displacement so speed > threshold
    ball_xy[77, 0] = 0.5 + (RESTART_SPEED_THRESH + 1.0) / (_FPS * 105.0)

    mask_no_bxy = dead_ball_mask(synth_xml_long_path, n_frames=n, buffer=0)
    mask_bxy    = dead_ball_mask(synth_xml_long_path, n_frames=n, buffer=0,
                                 ball_xy=ball_xy)

    assert mask_no_bxy.sum() > mask_bxy.sum(), (
        "restart detection must reduce dead-frame count vs 35 s template"
    )
    # Without ball_xy: end_frame = int(round(37.0*25))+1 = 926 → dead up to 925
    assert mask_no_bxy[900], "no-ball_xy mask should still cover frame 900"
    # With ball_xy: dead region must end well before frame 900
    assert not mask_bxy[900], "ball_xy mask must NOT cover frame 900"


def test_restart_detection_fallback_when_no_restart(synth_xml_long_path):
    """When ball never moves, end_frame is capped at MAX_RESTART_FRAMES."""
    from src.dead_ball import dead_ball_mask, MAX_RESTART_FRAMES
    n = 2000
    # Ball is always missing → no restart detected
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    mask = dead_ball_mask(synth_xml_long_path, n_frames=n, buffer=0,
                          ball_xy=ball_xy)
    # start_frame=51, cap = 51 + MAX_RESTART_FRAMES = 551 (1-indexed)
    # → mask should be True up to frame 550 (0-indexed) and False beyond
    assert mask[549], "cap frame should be dead"
    assert not mask[560], "frame beyond cap should be alive"


def test_csv_malformed_row_skipped(tmp_path):
    """Malformed CSV row (missing 'Start Frame') must be skipped, not crash."""
    import warnings
    from src.dead_ball import dead_ball_mask
    p = tmp_path / "events.csv"
    p.write_text("Type,Start Frame,End Frame\n"
                 "BALL OUT,10,20\n"
                 "BALL OUT,,\n")   # malformed — empty Start Frame
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mask = dead_ball_mask(str(p), n_frames=50)
    assert mask[9:20].all(), "well-formed row must be masked"
    assert len(w) == 1, "one warning expected for malformed row"


def test_json_malformed_event_skipped(tmp_path):
    """Malformed JSON event (missing 'start') must be skipped, not crash."""
    import json, warnings
    from src.dead_ball import dead_ball_mask
    events = [
        {"type": "BALL OUT", "start": {"frame": 10}, "end": {"frame": 20}},
        {"type": "BALL OUT"},   # malformed — no 'start' key
    ]
    p = tmp_path / "events.json"
    p.write_text(json.dumps(events))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mask = dead_ball_mask(str(p), n_frames=50)
    assert mask[9:20].all(), "well-formed event must be masked"
    assert len(w) == 1, "one warning expected for malformed event"
