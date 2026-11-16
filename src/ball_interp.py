"""Linear interpolation of ball position within half-blocks.

Only fills NaN values that are bounded by known positions on both sides
within the same half (period 1 or 2). Leading/trailing NaNs at the edges
of a half are left as NaN (no extrapolation).
"""

import numpy as np


def interpolate_ball(data: dict, max_gap: int = 10) -> dict:
    """Fill ball_xy NaNs by linear interpolation within each half-block.

    Parameters
    ----------
    data    : dict with keys ball_xy (N, 2) float32 and periods (N,) int8
    max_gap : maximum consecutive NaN frames to fill.  Longer gaps are left
              as NaN (ball genuinely out of view for an extended period).
              0 or None disables the cap and fills all interior NaN runs.
              Default 10 frames (0.4 s at 25 fps) targets brief occlusions.

    Returns
    -------
    New dict (copy of data) with ball_xy NaNs filled where bounded by
    known values within the same half and within the gap limit.
    Other keys are unchanged.
    """
    ball_xy = data["ball_xy"].copy()
    periods = data["periods"]
    cap = max_gap if (max_gap and max_gap > 0) else None

    for half in (1, 2):
        mask = periods == half
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        ball_half = ball_xy[indices]  # (M, 2)
        visible = ~np.isnan(ball_half[:, 0])  # (M,)

        if visible.sum() < 2:
            continue

        known_idx = np.where(visible)[0]

        if cap is None:
            # Fill all interior NaN runs
            first_known, last_known = known_idx[0], known_idx[-1]
            interp_range = np.arange(first_known, last_known + 1)
            for coord in (0, 1):
                known_vals = ball_half[known_idx, coord]
                interp_vals = np.interp(interp_range, known_idx, known_vals)
                ball_half[interp_range, coord] = interp_vals
        else:
            # Only fill NaN runs bounded on both sides AND <= cap frames long
            M = len(visible)
            fill_mask = np.zeros(M, dtype=bool)
            i = 0
            while i < M:
                if not visible[i]:
                    j = i
                    while j < M and not visible[j]:
                        j += 1
                    if (j - i) <= cap and i > 0 and j < M:
                        fill_mask[i:j] = True
                    i = j
                else:
                    i += 1
            interp_indices = np.where(fill_mask)[0]
            if len(interp_indices) > 0:
                for coord in (0, 1):
                    known_vals = ball_half[known_idx, coord]
                    ball_half[interp_indices, coord] = np.interp(
                        interp_indices, known_idx, known_vals
                    )

        ball_xy[indices] = ball_half

    return {**data, "ball_xy": ball_xy}
