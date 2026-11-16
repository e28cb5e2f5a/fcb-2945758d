"""Load BAR-ATH challenge tracking data into the pipeline's numpy dict.

The challenge data lives under data/2027_challenge/{tactical,broadcast}/ and
uses the FIFA/EPTS ATD format with Metrica-specific filenames.  This is a thin
shim that reads those files directly (no kloppy dependency) and returns the
same dict shape that sample_loader.load_tracking() and features.extract_features()
expect:

    frames   (N,)                int32   — video frame numbers (1-indexed)
    periods  (N,)                int8    — 1 or 2; 0 for halftime / outside match
    home_xy  (N, N_PLAYERS_MAX, 2) float32 — normalised [0,1], NaN if missing
    away_xy  (N, N_PLAYERS_MAX, 2) float32 — normalised [0,1], NaN if missing
    ball_xy  (N, 2)              float32 — normalised [0,1], NaN if missing

Usage
-----
    from src.challenge_loader import load_challenge_tracking
    data = load_challenge_tracking()              # tactical (default)
    data = load_challenge_tracking(source="broadcast")
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import asset_paths, half_of_video_frame, GAME, DATA_DIR
from src.sample_loader import N_PLAYERS_MAX

_CACHE_FILE = ".challenge_tracking_cache.npz"

# The challenge bundle lives one level below DATA_DIR
_CHALLENGE_DATA_DIR = os.environ.get("MTAH_DATA_DIR") or os.path.join(DATA_DIR, "2027_challenge")

# Home team is always Barcelona in this fixture (attack_direction_first_half=left_to_right)
_HOME_TEAM_NAME = "Barcelona"


def load_challenge_tracking(source: str = "tactical", game: str = GAME,
                             data_dir: str = None) -> dict:
    """Load challenge tracking data for one source (tactical or broadcast).

    Parameters
    ----------
    source   : "tactical" (default) or "broadcast"
    game     : game key, defaults to GAME ("BAR-ATH")
    data_dir : override for the data root (defaults to data/2027_challenge);
               set MTAH_DATA_DIR env var to point elsewhere permanently

    Returns
    -------
    dict with keys: frames, periods, home_xy, away_xy, ball_xy
    """
    paths = asset_paths(game=game, source=source, data_dir=data_dir or _CHALLENGE_DATA_DIR)
    meta_path = paths["atd_meta"]
    raw_path  = paths["atd_raw"]

    # Cache next to the raw file so it travels with the data
    cache_path = os.path.join(os.path.dirname(raw_path), _CACHE_FILE)
    if source == "broadcast":
        cache_path = cache_path.replace(".npz", "_broadcast.npz")

    missing = [p for p in (meta_path, raw_path) if not os.path.isfile(p)]
    if missing:
        tactical_dir = os.path.join(data_dir or _CHALLENGE_DATA_DIR, "tactical")
        sys.exit(
            "\n"
            "  Challenge data not found.\n"
            "\n"
            "  Copy the BAR-ATH competition files into:\n"
            f"    {tactical_dir}/\n"
            "\n"
            "  Required files:\n"
            "    BAR-ATH_tactical_atd.xml\n"
            "    BAR-ATH_tactical_atd.txt\n"
        )

    if os.path.isfile(cache_path):
        cache_mtime = os.path.getmtime(cache_path)
        if (os.path.getmtime(raw_path) <= cache_mtime and
                os.path.getmtime(meta_path) <= cache_mtime):
            c = np.load(cache_path)
            return {k: c[k] for k in ("frames", "periods", "home_xy", "away_xy", "ball_xy")}

    data = _parse_atd(meta_path, raw_path, game=game)
    try:
        np.savez_compressed(cache_path, **data)
    except OSError:
        pass
    return data


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _parse_meta(meta_path: str) -> tuple[list[int], list[int]]:
    """Return (home_slot_indices, away_slot_indices) — 0-indexed into the 46-slot player block.

    The DataFormatSpecification lists PlayerChannelRefs in slot order; the XML
    cross-references PlayerChannel → Player → Team to identify which team each
    slot belongs to.
    """
    tree = ET.parse(meta_path)
    root = tree.getroot()

    # Team name → team id
    teams = {t.get("id"): (t.findtext("Name") or "").strip()
             for t in root.findall(".//Team")}

    # player_id → team name
    p_team = {p.get("id"): teams.get(p.get("teamId"), "")
              for p in root.findall(".//Player")}

    # PlayerChannel id "playerN_x" → player_id (for x-channel only)
    slot_player: dict[int, str] = {}
    for pc in root.findall(".//PlayerChannel"):
        if pc.get("channelId") == "x":
            slot_str = pc.get("id", "")  # e.g. "player3_x"
            try:
                slot_num = int(slot_str.split("_")[0].replace("player", ""))
            except ValueError:
                continue
            slot_player[slot_num] = pc.get("playerId", "")

    # Slot order from DataFormatSpecification (1-indexed slot numbers)
    dfs = root.find(".//DataFormatSpecification")
    import re
    ordered_slots = [int(m) for m in
                     re.findall(r'playerChannelId="player(\d+)_x"',
                                ET.tostring(dfs, encoding="unicode"))]

    home_slots, away_slots = [], []
    for slot_num in ordered_slots:
        pid = slot_player.get(slot_num, "")
        team_name = p_team.get(pid, "")
        idx = slot_num - 1  # 0-indexed
        if team_name == _HOME_TEAM_NAME:
            home_slots.append(idx)
        else:
            away_slots.append(idx)

    return home_slots, away_slots


def _parse_atd(meta_path: str, raw_path: str, game: str = GAME) -> dict:
    """Parse ATD txt + XML into the pipeline numpy dict."""
    home_slots, away_slots = _parse_meta(meta_path)

    # Count lines first for pre-allocation
    n_frames = sum(1 for _ in open(raw_path))

    frames  = np.zeros(n_frames, dtype=np.int32)
    periods = np.zeros(n_frames, dtype=np.int8)
    all_xy  = np.full((n_frames, 46, 2), np.nan, dtype=np.float32)
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)

    print(f"Parsing {n_frames:,} frames from {os.path.basename(raw_path)}...")
    with open(raw_path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            # Format: frame_id:p1x,p1y,p1s;p2x,p2y,p2s;...;p46x,p46y,p46s:ballx,bally,balls
            colon_parts = line.split(":")
            frame_id = int(colon_parts[0])
            frames[i] = frame_id

            half = half_of_video_frame(game, frame_id)
            periods[i] = half if half is not None else 0

            # Player block
            player_slots = colon_parts[1].split(";")
            for slot_idx, slot_str in enumerate(player_slots):
                if slot_idx >= 46:
                    break
                vals = slot_str.split(",")
                if vals[0] != "NaN":
                    all_xy[i, slot_idx, 0] = float(vals[0])
                if vals[1] != "NaN":
                    all_xy[i, slot_idx, 1] = float(vals[1])

            # Ball (last colon part)
            ball_vals = colon_parts[2].split(",")
            if ball_vals[0] != "NaN":
                ball_xy[i, 0] = float(ball_vals[0])
            if ball_vals[1] != "NaN":
                ball_xy[i, 1] = float(ball_vals[1])

    home_xy = np.full((n_frames, N_PLAYERS_MAX, 2), np.nan, dtype=np.float32)
    away_xy = np.full((n_frames, N_PLAYERS_MAX, 2), np.nan, dtype=np.float32)

    for dst_idx, src_idx in enumerate(home_slots[:N_PLAYERS_MAX]):
        home_xy[:, dst_idx, :] = all_xy[:, src_idx, :]

    for dst_idx, src_idx in enumerate(away_slots[:N_PLAYERS_MAX]):
        away_xy[:, dst_idx, :] = all_xy[:, src_idx, :]

    print(f"  Home slots: {len(home_slots)}, Away slots: {len(away_slots)}")
    ball_visible = ~np.isnan(ball_xy[:, 0])
    print(f"  Ball visible: {ball_visible.sum():,} / {n_frames:,} "
          f"({100 * ball_visible.mean():.1f}%)")

    return {
        "frames":  frames,
        "periods": periods,
        "home_xy": home_xy,
        "away_xy": away_xy,
        "ball_xy": ball_xy,
    }
