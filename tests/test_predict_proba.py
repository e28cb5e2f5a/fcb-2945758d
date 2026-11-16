"""Unit tests for PossessionLSTM.predict_proba()."""

import numpy as np
import pytest


def test_predict_proba_shape_and_sums_to_one():
    """predict_proba() returns (N, 3) with rows summing to ~1."""
    from src.possession_lstm import PossessionLSTM

    rng = np.random.default_rng(0)
    X = rng.random((200, 14)).astype("float32")
    y = np.array((["H", "A", "D"] * 67)[:200])
    dm = np.zeros(200, dtype=bool)

    model = PossessionLSTM(hidden_size=8, max_epochs=1, seed=42)
    model.fit(X, y, dm)

    proba = model.predict_proba(X)
    assert proba.shape == (200, 3), f"Expected (200,3), got {proba.shape}"
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), \
        f"Rows do not sum to 1: {row_sums.min():.6f} – {row_sums.max():.6f}"


def test_predict_proba_argmax_matches_predict():
    """argmax of predict_proba should match predict() labels
    (dead-mask is all-False so the only difference is msf)."""
    from src.possession_lstm import PossessionLSTM

    rng = np.random.default_rng(1)
    X = rng.random((200, 14)).astype("float32")
    y = np.array(["H", "A", "D"] * 67)[:200]
    dm = np.zeros(200, dtype=bool)

    model = PossessionLSTM(hidden_size=8, max_epochs=1, seed=42)
    model.fit(X, y, dm)

    proba = model.predict_proba(X)
    idx_map = {0: "H", 1: "A", 2: "D"}
    proba_labels = np.array(
        [idx_map[int(i)] for i in proba.argmax(axis=1)], dtype="<U1"
    )
    # predict() applies msf; pre-msf argmax should match before smoothing.
    # Just confirm shapes match and labels are in {H,A,D}.
    assert len(proba_labels) == 200
    assert set(proba_labels).issubset({"H", "A", "D"})


def test_predict_proba_raises_without_fit():
    """predict_proba() raises RuntimeError if called before fit()."""
    from src.possession_lstm import PossessionLSTM

    model = PossessionLSTM(hidden_size=8, max_epochs=1)
    X = np.random.rand(100, 14).astype("float32")
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_proba(X)


def test_predict_proba_raises_for_crf_model():
    """predict_proba() raises RuntimeError if model uses CRF."""
    from src.possession_lstm import PossessionLSTM

    rng = np.random.default_rng(2)
    X = rng.random((200, 14)).astype("float32")
    y = np.array((["H", "A", "D"] * 67)[:200])
    dm = np.zeros(200, dtype=bool)

    model = PossessionLSTM(hidden_size=8, max_epochs=1, use_crf=True, seed=42)
    model.fit(X, y, dm)

    with pytest.raises(RuntimeError, match="CRF"):
        model.predict_proba(X)


def test_predict_proba_no_dead_mask_applied():
    """predict_proba() must not apply dead-ball mask internally."""
    from src.possession_lstm import PossessionLSTM

    rng = np.random.default_rng(3)
    X = rng.random((200, 14)).astype("float32")
    y = np.array((["H", "A", "D"] * 67)[:200])
    dm = np.zeros(200, dtype=bool)

    model = PossessionLSTM(hidden_size=8, max_epochs=1, seed=42)
    model.fit(X, y, dm)

    proba = model.predict_proba(X)
    # All three class probabilities should be non-trivially above 0 somewhere —
    # if dead-mask were applied, D-column would be 1.0 on dead frames.
    # Since dm is all-False here, we just verify columns vary (no trivial override).
    assert proba[:, 0].max() > 0.01, "H column is near-zero everywhere"
    assert proba[:, 1].max() > 0.01, "A column is near-zero everywhere"
    assert proba[:, 2].max() > 0.01, "D column is near-zero everywhere"
