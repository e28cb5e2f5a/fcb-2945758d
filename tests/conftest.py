"""Session-scoped fixtures for tests.

Two tiers:
  game3_*  — real Game 3 data (loaded from .npz cache, ~3 s); used only by
             test_sample_loader.py to verify the loader's output shapes/values.
  synth_*  — small synthetic dataset (5 000 frames, instant); used by all
             logic-testing modules (features, dead_ball, hmm, pipeline).
"""
import os
import json
import numpy as np
import pytest

GAME3_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "Sample_Game_3")
EVENTS_PATH = os.path.join(GAME3_DIR, "Sample_Game_3_events.json")

_N = 2_000   # frames in synthetic dataset
_SEED = 42


# ---------------------------------------------------------------------------
# Real Game 3 fixtures (loader verification only)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def game3_data():
    from src.sample_loader import load_tracking
    return load_tracking(GAME3_DIR)


@pytest.fixture(scope="session")
def game3_gt():
    from src.sample_loader import load_gt
    return load_gt(GAME3_DIR)


# ---------------------------------------------------------------------------
# Synthetic fixtures (fast logic tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synth_data():
    rng = np.random.default_rng(_SEED)
    n = _N
    periods = np.ones(n, dtype=np.int8)
    periods[n // 2:] = 2
    frames = np.arange(1, n + 1, dtype=np.int32)
    home_xy = rng.random((n, 46, 2)).astype(np.float32)
    away_xy = rng.random((n, 46, 2)).astype(np.float32)
    ball_xy = rng.random((n, 2)).astype(np.float32)
    ball_xy[rng.random(n) < 0.32] = np.nan   # ~30 % missing, like real game
    return {"frames": frames, "periods": periods,
            "home_xy": home_xy, "away_xy": away_xy, "ball_xy": ball_xy}


@pytest.fixture(scope="session")
def synth_feats(synth_data):
    from src.features import extract_features
    return extract_features(synth_data)


@pytest.fixture(scope="session")
def synth_gt(synth_data):
    """Synthetic GT: roughly realistic H/A/D distribution."""
    rng = np.random.default_rng(_SEED + 1)
    n = len(synth_data["frames"])
    choices = rng.choice(["H", "A", "D"], size=n, p=[0.35, 0.35, 0.30])
    return np.array(choices, dtype="<U1")


@pytest.fixture(scope="session")
def synth_dmask(synth_data):
    """Simple dead-ball mask: first 100 and last 100 frames dead."""
    n = len(synth_data["frames"])
    mask = np.zeros(n, dtype=bool)
    mask[:100] = True
    mask[-100:] = True
    return mask


@pytest.fixture(scope="session")
def synth_hmm(synth_feats, synth_gt, synth_dmask):
    from src.possession_hmm import PossessionHMM
    model = PossessionHMM()
    model.fit(synth_feats, synth_gt, synth_dmask)
    return model


@pytest.fixture(scope="session")
def synth_events_path(tmp_path_factory):
    """Write a tiny events JSON for dead_ball tests."""
    events = {
        "data": [
            {"type": {"name": "BALL OUT"}, "start": {"frame": 50}, "end": {"frame": 60}},
            {"type": {"name": "FAULT RECEIVED"}, "start": {"frame": 200}, "end": {"frame": 210}},
            {"type": {"name": "PASS"}, "start": {"frame": 300}, "end": {"frame": 310}},
        ],
        "metadata": []
    }
    p = tmp_path_factory.mktemp("events") / "events.json"
    p.write_text(json.dumps(events))
    return str(p)


# ---------------------------------------------------------------------------
# Pipeline fixture (end-to-end, uses real cached data)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pipeline_csv(tmp_path_factory, synth_feats, synth_gt, synth_dmask):
    """Run HMM pipeline on synthetic data; validates CSV format (not accuracy).

    Uses synthetic data so the fixture completes in milliseconds rather than
    loading + fitting on the 143k-frame Game 3 cache (~7 s).
    """
    from src.possession_hmm import PossessionHMM
    from src.pipeline import _write_csv
    out = str(tmp_path_factory.mktemp("pipeline") / "submission.csv")
    # train == predict: intentional — tests CSV format only, not generalisation.
    # Use PossessionHMM directly (avoids the `import torch` inside train_model
    # that would add ~2 s on the first test run before torch is warm).
    model = PossessionHMM()
    model.fit(synth_feats, synth_gt, synth_dmask)
    labels = model.predict(synth_feats, synth_dmask)
    _write_csv(out, labels)
    return out
