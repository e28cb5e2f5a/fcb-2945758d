"""Per-frame feature extraction from tracking data arrays.

Feature vector (18 columns):
  0  ball_x             normalised pitch x (0-1), 0.5 if ball missing
  1  ball_y             normalised pitch y (0-1), 0.5 if ball missing
  2  dist_nearest_H     metres from ball to nearest home player, 999 if ball missing
  3  dist_nearest_A     metres from ball to nearest away player, 999 if ball missing
  4  delta_dist_H       change in dist_nearest_H from previous frame, 0 at period start
  5  delta_dist_A       change in dist_nearest_A from previous frame, 0 at period start
  6  ball_speed         ball speed m/s (euclidean * 25 fps), 0 if ball missing
  7  n_H_within_3m      count of home players within 3m of ball, 0 if ball missing
  8  ball_accel         ‖Δv_ball‖ m/s² (2nd derivative), 0 if ball missing or period start
  9  spread_H           std of home player x-positions in metres (nanstd), 0 if no players
  10 spread_A           std of away player x-positions in metres (nanstd), 0 if no players
  11 centroid_dist_H    metres from home centroid to ball, 999 if ball missing
  12 centroid_dist_A    metres from away centroid to ball, 999 if ball missing
  13 dist_ratio         dist_nearest_H / (dist_nearest_A + 1e-3);
                       <1 ball nearer Home, >1 ball nearer Away,
                       1.0 sentinel when ball missing
  14 ball_in_final_third  1.0 if ball_x < 0.33 or ball_x > 0.67, else 0.0;
                          0.0 if ball missing (sentinel 0.5 → midfield → 0.0)
  15 ball_vx            signed ball x-velocity in m/s (ball_diff_x * FPS);
                        0.0 at period start, just-appeared frames, or ball missing
  16 set_piece_proxy    1.0 if ball was slow (<1.5 m/s, visible) for ≥20 of
                        the last 25 frames; 0.0 otherwise or ball missing
  17 h_centroid_adv_x   home centroid x minus away centroid x in metres;
                        positive = home team further toward x=1 side;
                        does not depend on ball visibility

Note: ball_visible was removed in Round 11 (Fix 3: drop ball_visible to
  break the missing-ball→D shortcut). Restored in Round 12 for 15-feat
  experiment (v10), but reverted back to 14-feat after cross-check failed:
  v10 won only G3 (1/3 folds), weighted delta = −0.027 → rejected (§C12).
  Cols 14–17 added in Round 13 as tracking-derived proxies for Smart Tagging
  channels (ball_in_final_third, transitions, set_pieces). These are LOO-valid
  because they require only tracking data (no XML), making 18-feat the base.

Note on home/away symmetry (intentional, confirmed):
  Feature 7 (n_H_within_3m) has no away-team counterpart. n_A_within_3m was
  tested in isolation in iteration 7 (Game 3 LOO, bw=3, msf=125, seed=42):
    14-feat baseline (with ball_visible): spellF1 = 0.5997
    +n_A_within_3m only:                 spellF1 = 0.5782  (−0.0215)
    +n_A_within_3m + ratio:              spellF1 = 0.5846  (−0.0151)
  n_A_within_3m is the primary culprit; dist_ratio partially recovers.
  The asymmetry is intentional and confirmed. See docs/CHALLENGE7.md T1.

  Feature 13 (dist_ratio) was added in iteration 8 (T1 solo probe).
  Both the Game 3 LOO gate (+0.0808) and the Game 1 LOO cross-check
  (+0.0557) passed.
"""

import numpy as np

FEATURE_NAMES = [
    "ball_x",
    "ball_y",
    "dist_nearest_H",
    "dist_nearest_A",
    "delta_dist_H",
    "delta_dist_A",
    "ball_speed",
    "n_H_within_3m",
    "ball_accel",
    "spread_H",
    "spread_A",
    "centroid_dist_H",
    "centroid_dist_A",
    "dist_ratio",
    "ball_in_final_third",
    "ball_vx",
    "set_piece_proxy",
    "h_centroid_adv_x",
]

PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0
SENTINEL_DIST = 999.0
FPS = 25.0
CONTACT_RADIUS_M = 3.0


def extract_features(data: dict) -> np.ndarray:
    """Compute per-frame feature matrix from tracking arrays.

    Parameters
    ----------
    data : dict returned by sample_loader.load_tracking()

    Returns
    -------
    np.ndarray of shape (N, 18) float32, no NaN values.
    """
    home_xy = data["home_xy"]   # (N, N_PLAYERS_MAX, 2)
    away_xy = data["away_xy"]   # (N, N_PLAYERS_MAX, 2)
    ball_xy = data["ball_xy"]   # (N, 2)
    periods = data["periods"]   # (N,)

    N = len(periods)
    feats = np.zeros((N, 18), dtype=np.float32)

    ball_visible = ~np.isnan(ball_xy[:, 0])

    # Features 0-1: ball position (sentinel 0.5 when missing)
    feats[:, 0] = np.where(ball_visible, ball_xy[:, 0], 0.5)
    feats[:, 1] = np.where(ball_visible, ball_xy[:, 1], 0.5)

    # Convert to metres for distance calculations
    ball_m = np.stack([
        feats[:, 0] * PITCH_LENGTH,
        feats[:, 1] * PITCH_WIDTH,
    ], axis=1)  # (N, 2)

    home_m = home_xy * np.array([PITCH_LENGTH, PITCH_WIDTH], dtype=np.float32)
    away_m = away_xy * np.array([PITCH_LENGTH, PITCH_WIDTH], dtype=np.float32)

    # Features 2-3: minimum distance from ball to nearest player per team
    dist_H = _min_dist(home_m, ball_m)
    dist_A = _min_dist(away_m, ball_m)
    dist_H[~ball_visible] = SENTINEL_DIST
    dist_A[~ball_visible] = SENTINEL_DIST
    feats[:, 2] = dist_H
    feats[:, 3] = dist_A

    # Period boundary mask: True at first frame of each period
    period_start = np.zeros(N, dtype=bool)
    period_start[0] = True
    period_start[1:] = periods[1:] != periods[:-1]

    # Features 4-5: delta distances, reset at period boundaries
    delta_H = np.zeros(N, dtype=np.float32)
    delta_A = np.zeros(N, dtype=np.float32)
    delta_H[1:] = dist_H[1:] - dist_H[:-1]
    delta_A[1:] = dist_A[1:] - dist_A[:-1]
    delta_H[period_start] = 0.0
    delta_A[period_start] = 0.0
    feats[:, 4] = delta_H
    feats[:, 5] = delta_A

    # Feature 6: ball speed (m/s), reset at period boundaries
    # Also zero at first-visible frames: when ball reappears, ball_m[i-1] was the
    # (0.5*L, 0.5*W) sentinel, making ball_diff[i] a large spurious displacement.
    just_appeared = np.zeros(N, dtype=bool)
    just_appeared[1:] = ball_visible[1:] & ~ball_visible[:-1]

    ball_diff = np.zeros((N, 2), dtype=np.float32)
    ball_diff[1:] = ball_m[1:] - ball_m[:-1]
    ball_diff[period_start] = 0.0
    ball_diff[just_appeared] = 0.0   # suppress sentinel→real displacement spike
    ball_speed = np.sqrt((ball_diff ** 2).sum(axis=1)) * FPS
    ball_speed[~ball_visible] = 0.0
    feats[:, 6] = ball_speed

    # Feature 7: count of home players within CONTACT_RADIUS_M of ball
    diff_H = home_m - ball_m[:, np.newaxis, :]   # (N, N_PLAYERS_MAX, 2)
    dist_H_all = np.sqrt((diff_H ** 2).sum(axis=2))  # (N, N_PLAYERS_MAX)
    player_missing_H = np.isnan(home_xy[:, :, 0])
    dist_H_all[player_missing_H] = np.inf
    n_H = (dist_H_all < CONTACT_RADIUS_M).sum(axis=1).astype(np.float32)
    n_H[~ball_visible] = 0.0
    feats[:, 7] = n_H

    # Feature 8: ball acceleration ‖Δv_ball‖ m/s² (2nd derivative of position)
    # ball_diff already has sentinel spikes zeroed, so vel inherits clean values.
    # The frame after just_appeared still gets vel[i]-vel[i-1] where vel[i-1]=0,
    # which is the real velocity at i — acceptable, not a sentinel artefact.
    vel = ball_diff * FPS                             # (N, 2) velocity vectors
    accel_vec = np.zeros((N, 2), dtype=np.float32)
    accel_vec[1:] = vel[1:] - vel[:-1]
    accel_vec[period_start] = 0.0
    ball_accel = np.sqrt((accel_vec ** 2).sum(axis=1)) * FPS
    ball_accel[~ball_visible] = 0.0
    feats[:, 8] = ball_accel

    # Features 9-10: team spread — std of x-positions in metres
    feats[:, 9]  = np.nanstd(home_m[:, :, 0], axis=1).astype(np.float32)
    feats[:, 10] = np.nanstd(away_m[:, :, 0], axis=1).astype(np.float32)

    # Features 11-12: centroid-to-ball distance in metres, sentinel when ball missing
    home_centroid = np.nanmean(home_m, axis=1)   # (N, 2)
    away_centroid = np.nanmean(away_m, axis=1)   # (N, 2)
    centroid_dist_H = np.sqrt(((home_centroid - ball_m) ** 2).sum(axis=1)).astype(np.float32)
    centroid_dist_A = np.sqrt(((away_centroid - ball_m) ** 2).sum(axis=1)).astype(np.float32)
    centroid_dist_H[~ball_visible] = SENTINEL_DIST
    centroid_dist_A[~ball_visible] = SENTINEL_DIST
    feats[:, 11] = centroid_dist_H
    feats[:, 12] = centroid_dist_A

    # Feature 13: dist_ratio — ball closer to H (<1) or A (>1)
    _EPS = 1e-3
    dist_ratio = (dist_H / (dist_A + _EPS)).astype(np.float32)
    dist_ratio[~ball_visible] = 1.0   # neutral sentinel when no ball
    feats[:, 13] = dist_ratio

    # Feature 14: ball_in_final_third
    # ball_m[:,0] is already 0.5*PITCH_LENGTH sentinel when missing, which
    # is midfield → neither final third → 0.0 naturally.
    bx_norm = feats[:, 0]  # normalised [0,1], 0.5 when missing
    feats[:, 14] = ((bx_norm < 0.33) | (bx_norm > 0.67)).astype(np.float32)

    # Feature 15: ball_vx — signed x-velocity in m/s
    # ball_diff already has period-start, just-appeared, and missing zeroed.
    feats[:, 15] = (ball_diff[:, 0] * FPS).astype(np.float32)
    feats[~ball_visible, 15] = 0.0

    # Feature 16: set_piece_proxy
    # 1.0 if ball was slow (<1.5 m/s, visible) for ≥20 of the last 25 frames.
    _SP_WINDOW = 25
    _SP_THRESH = 20
    _SP_SPEED  = 1.5   # m/s
    slow_ball = ((ball_speed < _SP_SPEED) & ball_visible).astype(np.int32)
    cum = np.zeros(N + 1, dtype=np.int32)
    np.cumsum(slow_ball, out=cum[1:])
    # rolling[i] = sum of slow_ball[max(0, i-W+1) : i+1]
    starts = np.maximum(0, np.arange(N) - _SP_WINDOW + 1)
    rolling = cum[1:] - cum[starts]
    feats[:, 16] = (rolling >= _SP_THRESH).astype(np.float32)
    feats[~ball_visible, 16] = 0.0

    # Feature 17: h_centroid_adv_x — home centroid x minus away centroid x (m)
    # home_centroid and away_centroid already computed above (nanmean).
    # Not zeroed when ball missing — team advancement is ball-independent.
    feats[:, 17] = (home_centroid[:, 0] - away_centroid[:, 0]).astype(np.float32)

    return feats


def _min_dist(players_m: np.ndarray, ball_m: np.ndarray) -> np.ndarray:
    """Minimum distance (metres) from ball to any visible player. (N,) float32."""
    diff = players_m - ball_m[:, np.newaxis, :]   # (N, N_PLAYERS_MAX, 2)
    dist = np.sqrt((diff ** 2).sum(axis=2))        # (N, N_PLAYERS_MAX)
    # Mask missing players
    player_missing = np.isnan(players_m[:, :, 0])
    dist[player_missing] = np.inf
    result = dist.min(axis=1)
    result[~np.isfinite(result)] = SENTINEL_DIST
    return result.astype(np.float32)
