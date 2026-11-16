import numpy as np
import pytest
from src.ball_interp import interpolate_ball


def test_interpolates_between_visible_frames():
    ball = np.array([[0.3, 0.5], [np.nan, np.nan], [0.5, 0.5]], dtype=np.float32)
    periods = np.array([1, 1, 1], dtype=np.int8)
    result = interpolate_ball({"ball_xy": ball, "periods": periods})
    assert not np.isnan(result["ball_xy"][1, 0])
    np.testing.assert_allclose(result["ball_xy"][1], [0.4, 0.5], atol=1e-5)


def test_does_not_interpolate_across_half_boundary():
    ball = np.array([[0.3, 0.5], [np.nan, np.nan], [0.5, 0.5]], dtype=np.float32)
    periods = np.array([1, 0, 2], dtype=np.int8)  # period 0 = halftime gap
    result = interpolate_ball({"ball_xy": ball, "periods": periods})
    assert np.isnan(result["ball_xy"][1, 0])


def test_trailing_nan_at_end_of_half_stays_nan():
    ball = np.array([[0.3, 0.5], [0.4, 0.5], [np.nan, np.nan]], dtype=np.float32)
    periods = np.array([1, 1, 1], dtype=np.int8)
    result = interpolate_ball({"ball_xy": ball, "periods": periods})
    assert np.isnan(result["ball_xy"][2, 0])


def test_returns_copy_not_mutation():
    ball = np.array([[0.3, 0.5], [np.nan, np.nan], [0.5, 0.5]], dtype=np.float32)
    periods = np.array([1, 1, 1], dtype=np.int8)
    data = {"ball_xy": ball.copy(), "periods": periods}
    result = interpolate_ball(data)
    assert np.isnan(data["ball_xy"][1, 0])  # original unchanged
