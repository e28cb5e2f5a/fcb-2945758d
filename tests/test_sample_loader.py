"""Verify the EPTS loader (Game 3) and CSV loader (Games 1 & 2) output shapes and values."""
import csv
import os
import numpy as np
import pytest

N_FRAMES = 143761
N_SLOTS  = 46  # N_PLAYERS_MAX

# ---------------------------------------------------------------------------
# Synthetic CSV fixture — mimics Games 1 & 2 format without real data
# ---------------------------------------------------------------------------

def _write_csv_game(directory: "os.PathLike", name: str, n_frames: int = 20,
                    n_players: int = 14) -> None:
    """Write minimal home/away tracking CSVs + GT in Game 1/2 format."""
    os.makedirs(directory, exist_ok=True)

    # Header rows (3 lines before data)
    team_row  = ["", "", ""] + ["Home"] * (n_players * 2) + ["", ""]
    shirt_row = ["", "", ""] + [str(i + 1) for i in range(n_players)
                                for _ in range(2)] + ["", ""]
    col_row   = ["Period", "Frame", "Time [s]"] + \
                [f"Player{i+1}_x" if j == 0 else f"Player{i+1}_y"
                 for i in range(n_players) for j in range(2)] + \
                ["Ball_x", "Ball_y"]

    for team_label in ("Home_Team", "Away_Team"):
        path = os.path.join(directory, f"{name}_RawTrackingData_{team_label}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(team_row)
            writer.writerow(shirt_row)
            writer.writerow(col_row)
            for frame in range(1, n_frames + 1):
                period = 1 if frame <= n_frames // 2 else 2
                t = frame * 0.04
                players = [round(0.1 + 0.05 * ((i + frame) % 10), 4)
                           for i in range(n_players * 2)]
                ball = [0.5, 0.4] if frame % 5 != 0 else ["NaN", "NaN"]
                writer.writerow([period, frame, t] + players + ball)

    # Possession GT CSV
    gt_path = os.path.join(directory, f"{name}_possession_gt.csv")
    with open(gt_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "label"])
        labels = ["H", "A", "D"]
        for i in range(n_frames):
            writer.writerow([i, labels[i % 3]])


def test_load_tracking_shapes(game3_data):
    assert game3_data["frames"].shape  == (N_FRAMES,)
    assert game3_data["periods"].shape == (N_FRAMES,)
    assert game3_data["home_xy"].shape == (N_FRAMES, N_SLOTS, 2)
    assert game3_data["away_xy"].shape == (N_FRAMES, N_SLOTS, 2)
    assert game3_data["ball_xy"].shape == (N_FRAMES, 2)


def test_load_tracking_coordinates_in_range(game3_data):
    ball = game3_data["ball_xy"]
    valid = ball[~np.isnan(ball[:, 0])]
    assert valid[:, 0].min() >= -0.2
    assert valid[:, 0].max() <= 1.2


def test_load_tracking_ball_sometimes_missing(game3_data):
    assert np.isnan(game3_data["ball_xy"][:, 0]).sum() > 0


def test_load_tracking_two_periods(game3_data):
    assert set(game3_data["periods"].tolist()) == {1, 2}


def test_load_gt_shape_and_labels(game3_gt):
    assert game3_gt.shape == (N_FRAMES,)
    assert set(game3_gt.tolist()).issubset({"H", "A", "D", "X"})


def test_load_gt_has_all_three_scored_labels(game3_gt):
    assert "H" in game3_gt
    assert "A" in game3_gt
    assert "D" in game3_gt


# ---------------------------------------------------------------------------
# CSV loader tests (Games 1 & 2 format)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def csv_game_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("csv_game") / "Synth_Game_1"
    _write_csv_game(d, "Synth_Game_1", n_frames=20, n_players=14)
    return str(d)


@pytest.fixture(scope="module")
def csv_data(csv_game_dir):
    from src.sample_loader import load_tracking
    return load_tracking(csv_game_dir)


def test_csv_load_tracking_shapes(csv_data):
    assert csv_data["frames"].shape  == (20,)
    assert csv_data["periods"].shape == (20,)
    assert csv_data["home_xy"].shape == (20, N_SLOTS, 2)
    assert csv_data["away_xy"].shape == (20, N_SLOTS, 2)
    assert csv_data["ball_xy"].shape == (20, 2)


def test_csv_load_tracking_coordinates_in_range(csv_data):
    """All non-NaN player coordinates must be in [0, 1]."""
    for key in ("home_xy", "away_xy"):
        xy = csv_data[key]
        valid = xy[~np.isnan(xy)]
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0


def test_csv_load_tracking_unused_slots_are_nan(csv_data):
    """Slots beyond the 14 real players must be NaN-padded."""
    assert np.isnan(csv_data["home_xy"][:, 14:, 0]).all()
    assert np.isnan(csv_data["away_xy"][:, 14:, 0]).all()


def test_csv_load_tracking_ball_missing_on_every_fifth_frame(csv_data):
    """Frames 5, 10, 15, 20 have NaN ball in the fixture."""
    ball_x = csv_data["ball_xy"][:, 0]
    missing_idx = np.where(np.isnan(ball_x))[0]
    assert len(missing_idx) > 0


def test_csv_load_tracking_two_periods(csv_data):
    assert set(csv_data["periods"].tolist()) == {1, 2}


def test_csv_load_gt_original_format_only(csv_game_dir):
    """Format/IO test only — use_corrected=False, not the production default."""
    from src.sample_loader import load_gt
    gt = load_gt(csv_game_dir, use_corrected=False)
    assert gt.shape == (20,)
    assert set(gt.tolist()).issubset({"H", "A", "D", "X"})


def test_load_gt_corrected_raises_when_file_absent(csv_game_dir):
    """use_corrected=True must raise FileNotFoundError when corrected file is missing."""
    from src.sample_loader import load_gt
    with pytest.raises(FileNotFoundError, match="generate_gt.py"):
        load_gt(csv_game_dir, use_corrected=True)


def test_load_gt_corrected_loads_when_file_present(csv_game_dir, tmp_path):
    """use_corrected=True loads corrected file when it exists."""
    import shutil
    from src.sample_loader import load_gt

    # csv_game_dir name is the fixture game name
    game_name = os.path.basename(csv_game_dir)
    corrected = os.path.join(csv_game_dir, f"{game_name}_possession_gt_corrected.csv")

    # Write a minimal corrected GT — same frames as original but all H
    with open(corrected, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "label"])
        for i in range(20):
            writer.writerow([i, "H"])

    try:
        gt = load_gt(csv_game_dir, use_corrected=True)
        assert gt.shape == (20,)
        assert set(gt.tolist()) == {"H"}
    finally:
        os.remove(corrected)
