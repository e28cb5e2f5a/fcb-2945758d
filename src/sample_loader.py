"""Load Metrica sample game tracking data into numpy arrays.

Supports two formats:
  - EPTS (Game 3 / real BAR-ATH format): _metadata.xml + _tracking.txt
  - CSV  (Games 1 & 2 legacy):           _RawTrackingData_Home_Team.csv + Away_Team.csv

Format is auto-detected: if _tracking.txt exists → EPTS, else → CSV.
"""

import csv
import os
import warnings
import numpy as np

PITCH = (105.0, 68.0)   # metres — standard assumption for sample games
N_PLAYERS_MAX = 46      # real ATD has 46 track slots; arrays always padded to this size


def load_tracking(game_dir: str) -> dict:
    """Load home/away/ball tracking from a sample game directory.

    Auto-detects EPTS vs CSV format.

    Returns dict with:
      frames   (N,)              int32   — tracking frame numbers (1-indexed)
      periods  (N,)              int8    — period (1 or 2)
      home_xy  (N, N_PLAYERS_MAX, 2) float32 — home player positions, NaN if missing
      away_xy  (N, N_PLAYERS_MAX, 2) float32 — away player positions, NaN if missing
      ball_xy  (N, 2)            float32 — ball position, NaN if missing
    """
    game_name = os.path.basename(game_dir)
    tracking_txt = os.path.join(game_dir, f"{game_name}_tracking.txt")
    if os.path.isfile(tracking_txt):
        return _load_epts(game_dir, game_name)
    return _load_csv(game_dir, game_name)


# ---------------------------------------------------------------------------
# EPTS loader (Game 3 / real BAR-ATH format)
# ---------------------------------------------------------------------------

_CACHE_FILE = ".tracking_cache.npz"


def _load_epts(game_dir: str, game_name: str) -> dict:
    cache_path = os.path.join(game_dir, _CACHE_FILE)
    if os.path.isfile(cache_path):
        source_files = [
            os.path.join(game_dir, f"{game_name}_tracking.txt"),
            os.path.join(game_dir, f"{game_name}_metadata.xml"),
        ]
        cache_mtime = os.path.getmtime(cache_path)
        if all(os.path.getmtime(p) <= cache_mtime for p in source_files if os.path.isfile(p)):
            c = np.load(cache_path)
            return {k: c[k] for k in ("frames", "periods", "home_xy", "away_xy", "ball_xy")}

    data = _parse_epts(game_dir, game_name)
    try:
        np.savez_compressed(cache_path, **data)
    except OSError:
        pass
    return data


def _dataset_to_numpy(dataset) -> dict:
    """Convert a kloppy TrackingDataset to the pipeline's numpy dict."""
    df = dataset.to_df()

    home_team = dataset.metadata.teams[0]
    away_team = dataset.metadata.teams[1]
    home_ids = [p.player_id for p in home_team.players]
    away_ids = [p.player_id for p in away_team.players]

    N = len(df)
    frames  = df["frame_id"].values.astype(np.int32)
    periods = df["period_id"].values.astype(np.int8)

    home_xy = np.full((N, N_PLAYERS_MAX, 2), np.nan, dtype=np.float32)
    away_xy = np.full((N, N_PLAYERS_MAX, 2), np.nan, dtype=np.float32)

    for i, pid in enumerate(home_ids[:N_PLAYERS_MAX]):
        xcol, ycol = f"{pid}_x", f"{pid}_y"
        if xcol in df.columns:
            home_xy[:, i, 0] = df[xcol].values.astype(np.float32)
            home_xy[:, i, 1] = df[ycol].values.astype(np.float32)

    for i, pid in enumerate(away_ids[:N_PLAYERS_MAX]):
        xcol, ycol = f"{pid}_x", f"{pid}_y"
        if xcol in df.columns:
            away_xy[:, i, 0] = df[xcol].values.astype(np.float32)
            away_xy[:, i, 1] = df[ycol].values.astype(np.float32)

    ball_xy = np.full((N, 2), np.nan, dtype=np.float32)
    ball_xy[:, 0] = df["ball_x"].values.astype(np.float32)
    ball_xy[:, 1] = df["ball_y"].values.astype(np.float32)

    return {
        "frames":  frames,
        "periods": periods,
        "home_xy": home_xy,
        "away_xy": away_xy,
        "ball_xy": ball_xy,
    }


def _parse_epts(game_dir: str, game_name: str) -> dict:
    from kloppy import metrica
    dataset = metrica.load_tracking_epts(
        meta_data=os.path.join(game_dir, f"{game_name}_metadata.xml"),
        raw_data=os.path.join(game_dir, f"{game_name}_tracking.txt"),
    )
    return _dataset_to_numpy(dataset)


# ---------------------------------------------------------------------------
# CSV loader (Games 1 & 2 legacy format)
# ---------------------------------------------------------------------------

def _load_csv(game_dir: str, game_name: str) -> dict:
    from kloppy import metrica
    dataset = metrica.load_tracking_csv(
        home_data=os.path.join(game_dir, f"{game_name}_RawTrackingData_Home_Team.csv"),
        away_data=os.path.join(game_dir, f"{game_name}_RawTrackingData_Away_Team.csv"),
    )
    result = _dataset_to_numpy(dataset)
    # kloppy's CSV loader applies y → (1 − y); undo it so Games 1 & 2 use the
    # same raw [0, 1] Metrica coordinate system as the EPTS loader (Game 3).
    # NaN values are preserved: 1 − NaN = NaN.
    result["home_xy"][..., 1] = 1.0 - result["home_xy"][..., 1]
    result["away_xy"][..., 1] = 1.0 - result["away_xy"][..., 1]
    result["ball_xy"][:, 1]   = 1.0 - result["ball_xy"][:, 1]
    return result


# ---------------------------------------------------------------------------
# GT loader (same for both formats)
# ---------------------------------------------------------------------------

def load_gt(game_dir: str, use_corrected: bool = True) -> np.ndarray:
    """Load possession GT CSV → label array aligned to tracking frames.

    When use_corrected=True (default), loads the competition-rules-corrected
    GT (BALL LOST / CHALLENGE relabelled to carry-forward H/A).  Raises
    FileNotFoundError if the corrected file is absent — run generate_gt.py
    first.

    Precondition: use_corrected=False must NOT be used in any training or
    evaluation path.  BALL LOST is no longer in DEAD_BALL_TYPES; the
    original GT labels those frames as D while the dead-ball mask no longer
    suppresses them — a training/inference mismatch that fragments spells.
    use_corrected=False is only safe for unit tests that check file I/O
    mechanics, not possession semantics.

    Returns array shape (N,) dtype '<U1', values 'H'/'A'/'D'/'X'.
    """
    game_name = os.path.basename(game_dir)
    corrected_path = os.path.join(game_dir, f"{game_name}_possession_gt_corrected.csv")
    original_path  = os.path.join(game_dir, f"{game_name}_possession_gt.csv")
    if use_corrected:
        if not os.path.isfile(corrected_path):
            raise FileNotFoundError(
                f"Corrected GT not found: {corrected_path}\n"
                "Run 'python3.11 src/generate_gt.py' to generate it."
            )
        gt_path = corrected_path
    else:
        warnings.warn(
            "load_gt(use_corrected=False) loads the original GT where BALL LOST "
            "frames are labelled D. BALL LOST is no longer in DEAD_BALL_TYPES — "
            "using this GT for training or evaluation will produce a "
            "training/inference mismatch. Only use use_corrected=False for "
            "mechanical I/O tests.",
            UserWarning,
            stacklevel=2,
        )
        gt_path = original_path
    with open(gt_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        labels = [row[1] for row in reader]
    return np.array(labels, dtype="<U1")


def require_corrected_gt(game_dirs: list) -> None:
    """Raise SystemExit with a clear message if any corrected GT file is missing.

    Call this at the top of main() in scripts that use corrected GT for
    training, before any expensive work starts.
    """
    missing = []
    for game_dir in game_dirs:
        game_name = os.path.basename(game_dir)
        corrected = os.path.join(game_dir,
                                 f"{game_name}_possession_gt_corrected.csv")
        if not os.path.isfile(corrected):
            missing.append(corrected)
    if missing:
        paths = "\n  ".join(missing)
        print(
            f"ERROR: corrected GT files not found:\n  {paths}\n\n"
            "Run this first:\n"
            "  python3.11 src/generate_gt.py\n",
            file=__import__("sys").stderr,
        )
        raise SystemExit(1)
