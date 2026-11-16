import numpy as np
import pytest


def test_feature_shape(synth_data, synth_feats):
    n = len(synth_data["frames"])
    assert synth_feats.shape == (n, 18)


def test_no_nan_in_features(synth_feats):
    assert not np.isnan(synth_feats).any()


def test_distances_positive(synth_feats):
    assert (synth_feats[:, 2] >= 0).all()  # dist_nearest_H
    assert (synth_feats[:, 3] >= 0).all()  # dist_nearest_A


def test_sentinel_when_ball_missing(synth_data, synth_feats):
    missing_mask = np.isnan(synth_data["ball_xy"][:, 0])
    assert (synth_feats[missing_mask, 2] == 999.0).all()  # dist_nearest_H = sentinel
    assert (synth_feats[missing_mask, 3] == 999.0).all()  # dist_nearest_A = sentinel


def test_delta_resets_at_period_boundary(synth_data, synth_feats):
    period2_start = int(np.where(synth_data["periods"] == 2)[0][0])
    assert synth_feats[period2_start, 4] == 0.0  # delta_dist_H
    assert synth_feats[period2_start, 5] == 0.0  # delta_dist_A


def test_ball_accel_zero_when_missing(synth_data, synth_feats):
    missing_mask = np.isnan(synth_data["ball_xy"][:, 0])
    assert (synth_feats[missing_mask, 8] == 0.0).all()


def test_ball_speed_zero_at_first_visible_after_missing(synth_data, synth_feats):
    # At frames where ball reappears after a missing run, ball_m[i-1] was the
    # sentinel (0.5*L, 0.5*W). Without the fix, ball_diff[i] would be a large
    # spurious displacement. The fix zeros ball_diff[just_appeared], so
    # ball_speed must be 0.0 at every such frame.
    ball_vis = ~np.isnan(synth_data["ball_xy"][:, 0])
    just_appeared = np.zeros(len(ball_vis), dtype=bool)
    just_appeared[1:] = ball_vis[1:] & ~ball_vis[:-1]
    assert (synth_feats[just_appeared, 6] == 0.0).all(), \
        "ball_speed non-zero at missing→visible transition (sentinel spike)"


def test_spread_nonnegative(synth_feats):
    assert (synth_feats[:, 9] >= 0).all()   # spread_H
    assert (synth_feats[:, 10] >= 0).all()  # spread_A


def test_centroid_dist_sentinel_when_missing(synth_data, synth_feats):
    missing_mask = np.isnan(synth_data["ball_xy"][:, 0])
    assert (synth_feats[missing_mask, 11] == 999.0).all()  # centroid_dist_H
    assert (synth_feats[missing_mask, 12] == 999.0).all()  # centroid_dist_A
