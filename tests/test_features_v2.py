"""Tests for the 18-feature extract_features() — Round 13.

History:
- ball_visible removed in Round 11 (Fix 3: break missing-ball→D shortcut).
- Reverted back to 14-feat in Round 12 after 15-feat cross-check failed (§C12):
  v10 (15-feat) won only G3 (1/3 folds), weighted delta = −0.027.
- 4 tracking-derived proxy features added in Round 13 (cols 14–17):
  ball_in_final_third, ball_vx, set_piece_proxy, h_centroid_adv_x.

Feature n_A_within_3m remains absent (banned — Round 7 T1: −0.0215).
Feature dist_ratio (index 13) was added in iteration 8.
"""
import numpy as np
import pytest
from src.features import extract_features, FEATURE_NAMES


def _make_data(n=10, ball_visible=True):
    home_xy = np.random.rand(n, 46, 2).astype(np.float32)
    away_xy = np.random.rand(n, 46, 2).astype(np.float32)
    ball_xy = np.random.rand(n, 2).astype(np.float32)
    if not ball_visible:
        ball_xy[:] = np.nan
    return {
        "home_xy": home_xy,
        "away_xy": away_xy,
        "ball_xy": ball_xy,
        "periods": np.ones(n, dtype=np.int8),
    }


def test_feature_count():
    feats = extract_features(_make_data())
    assert feats.shape[1] == 18
    assert len(FEATURE_NAMES) == 18


def test_ball_visible_not_present():
    """Confirm ball_visible is absent (18-feat config, §C13)."""
    assert "ball_visible" not in FEATURE_NAMES


def test_n_A_within_3m_not_present():
    """Confirm n_A_within_3m (banned Round 7) is absent."""
    assert "n_A_within_3m" not in FEATURE_NAMES


def test_dist_ratio_present():
    """Confirm dist_ratio (col 13) is present."""
    assert "dist_ratio" in FEATURE_NAMES
    assert FEATURE_NAMES.index("dist_ratio") == 13


def test_dist_ratio_sentinel_when_ball_missing():
    """dist_ratio must be 1.0 (neutral sentinel) when ball is not visible."""
    data = _make_data(n=20, ball_visible=False)
    feats = extract_features(data)
    idx = FEATURE_NAMES.index("dist_ratio")
    assert (feats[:, idx] == 1.0).all(), \
        "dist_ratio should be 1.0 sentinel when ball is missing"


# ---------------------------------------------------------------------------
# Round 13: tracking-derived proxy features (cols 14–17)
# ---------------------------------------------------------------------------

def test_new_features_present():
    """Confirm all four R13 proxy features are in FEATURE_NAMES at the right indices."""
    assert FEATURE_NAMES.index("ball_in_final_third") == 14
    assert FEATURE_NAMES.index("ball_vx") == 15
    assert FEATURE_NAMES.index("set_piece_proxy") == 16
    assert FEATURE_NAMES.index("h_centroid_adv_x") == 17


def test_ball_in_final_third_binary():
    """ball_in_final_third must be 0.0 or 1.0 only."""
    feats = extract_features(_make_data(n=50))
    vals = feats[:, 14]
    assert set(np.unique(vals)).issubset({0.0, 1.0})


def test_ball_in_final_third_zero_when_missing():
    """ball_in_final_third must be 0.0 when ball is missing (sentinel 0.5 → midfield)."""
    data = _make_data(n=20, ball_visible=False)
    feats = extract_features(data)
    assert (feats[:, 14] == 0.0).all()


def test_ball_vx_zero_when_missing():
    """ball_vx must be 0.0 when ball is not visible."""
    data = _make_data(n=20, ball_visible=False)
    feats = extract_features(data)
    assert (feats[:, 15] == 0.0).all()


def test_set_piece_proxy_binary():
    """set_piece_proxy must be 0.0 or 1.0 only."""
    feats = extract_features(_make_data(n=50))
    vals = feats[:, 16]
    assert set(np.unique(vals)).issubset({0.0, 1.0})


def test_set_piece_proxy_zero_when_missing():
    """set_piece_proxy must be 0.0 when ball is not visible."""
    data = _make_data(n=20, ball_visible=False)
    feats = extract_features(data)
    assert (feats[:, 16] == 0.0).all()


def test_set_piece_proxy_fires_on_slow_ball():
    """set_piece_proxy = 1.0 when ball has been slow for ≥20 of last 25 frames."""
    n = 50
    data = _make_data(n=n, ball_visible=True)
    # Force ball to stay near centre (no displacement → speed ≈ 0)
    data["ball_xy"][:] = np.array([0.5, 0.5], dtype=np.float32)
    feats = extract_features(data)
    # Frames ≥ 24 should fire (window filled with slow frames)
    assert feats[24, 16] == 1.0, "set_piece_proxy should fire after 25 slow frames"
    assert (feats[24:, 16] == 1.0).all()


def test_h_centroid_adv_x_finite():
    """h_centroid_adv_x must be finite for all frames."""
    feats = extract_features(_make_data(n=30))
    assert np.isfinite(feats[:, 17]).all()


def test_h_centroid_adv_x_independent_of_ball():
    """h_centroid_adv_x must not change when ball visibility changes."""
    rng = np.random.default_rng(0)
    n = 20
    home_xy = rng.random((n, 46, 2)).astype(np.float32)
    away_xy = rng.random((n, 46, 2)).astype(np.float32)
    ball_vis = {
        "home_xy": home_xy, "away_xy": away_xy,
        "ball_xy": rng.random((n, 2)).astype(np.float32),
        "periods": np.ones(n, dtype=np.int8),
    }
    ball_miss = {
        "home_xy": home_xy, "away_xy": away_xy,
        "ball_xy": np.full((n, 2), np.nan, dtype=np.float32),
        "periods": np.ones(n, dtype=np.int8),
    }
    feats_vis  = extract_features(ball_vis)
    feats_miss = extract_features(ball_miss)
    np.testing.assert_array_equal(feats_vis[:, 17], feats_miss[:, 17])
