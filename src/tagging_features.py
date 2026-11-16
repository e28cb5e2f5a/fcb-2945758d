"""Binary per-frame features derived from the Smart Tagging XML.

Four channels (all 0/1 float32):
  0  in_attacking_transition   — frame is within an ATTACKING TRANSITION interval
  1  in_defensive_transition   — frame is within a DEFENSIVE TRANSITION interval
  2  in_ball_final_third       — frame is within a BALL IN FINAL THIRD interval
  3  in_set_piece              — frame is within a SET PIECES interval

Time base: <start>/<end> in the XML are seconds from the start of the tactical
video. Frame index = int(round(seconds * fps)), 0-indexed.
"""

import xml.etree.ElementTree as ET

import numpy as np

# Maps XML <code> text (case-insensitive) to column index
_CODE_TO_COL = {
    "attacking transition": 0,
    "defensive transition": 1,
    "ball in final third": 2,
    "set pieces": 3,
}


def tagging_features(xml_path: str, n_frames: int, fps: float = 25.0) -> np.ndarray:
    """Compute binary tagging feature matrix from Smart Tagging XML.

    Parameters
    ----------
    xml_path  : path to *_smart_tagging.xml
    n_frames  : total number of video frames (length of the output array)
    fps       : video frame rate (default 25)

    Returns
    -------
    np.ndarray shape (n_frames, 4) float32, values 0 or 1
    """
    feats = np.zeros((n_frames, 4), dtype=np.float32)
    tree = ET.parse(xml_path)
    for instance in tree.iter("instance"):
        code = (instance.findtext("code") or "").strip().lower()
        col = _CODE_TO_COL.get(code)
        if col is None:
            continue
        start_s = float(instance.findtext("start") or 0)
        end_s = float(instance.findtext("end") or 0)
        start_f = int(round(start_s * fps))
        end_f = int(round(end_s * fps))
        # Clip to valid frame range
        start_f = max(0, min(start_f, n_frames - 1))
        end_f = max(0, min(end_f, n_frames - 1))
        feats[start_f:end_f + 1, col] = 1.0
    return feats
