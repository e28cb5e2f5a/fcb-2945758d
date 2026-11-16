import os
import numpy as np
from unittest.mock import patch


def test_dispatches_to_sample_loader_for_sample_game(tmp_path):
    """Non-EPTS directory routes to sample_loader."""
    from src.loader import load_tracking
    with patch("src.loader._sample_load_tracking") as mock:
        mock.return_value = {"frames": np.array([1])}
        load_tracking(str(tmp_path))
        mock.assert_called_once()


def test_dispatches_to_challenge_loader_for_epts_dir(tmp_path):
    """Directory containing _atd.txt routes to challenge_loader."""
    (tmp_path / "BAR-ATH_tactical_atd.txt").touch()
    from src.loader import load_tracking
    with patch("src.loader._challenge_load_tracking") as mock:
        mock.return_value = {"frames": np.array([1])}
        load_tracking(str(tmp_path))
        mock.assert_called_once()
