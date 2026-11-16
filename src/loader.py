"""Unified load_tracking() — auto-detects EPTS vs. sample-game format.

Detects challenge (EPTS/ATD) data by the presence of a file matching
*_atd.txt in the given directory. Falls back to sample_loader otherwise.
"""

import glob
import os


def _is_epts_dir(game_dir: str) -> bool:
    return bool(glob.glob(os.path.join(game_dir, "*_atd.txt")))


def _sample_load_tracking(game_dir, **kwargs):
    from src.sample_loader import load_tracking
    return load_tracking(game_dir, **kwargs)


def _challenge_load_tracking(game_dir, **kwargs):
    from src.challenge_loader import load_challenge_tracking
    return load_challenge_tracking(data_dir=game_dir, **kwargs)


def load_tracking(game_dir: str, **kwargs) -> dict:
    """Load tracking data, auto-detecting format from directory contents.

    Parameters
    ----------
    game_dir : path to game directory (sample game) or challenge data dir

    Returns
    -------
    dict with keys: frames, periods, home_xy, away_xy, ball_xy
    """
    if _is_epts_dir(game_dir):
        return _challenge_load_tracking(game_dir, **kwargs)
    return _sample_load_tracking(game_dir, **kwargs)
