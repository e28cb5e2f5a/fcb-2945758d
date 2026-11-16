import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth_lstm(synth_feats, synth_gt, synth_dmask):
    from src.possession_lstm import PossessionLSTM
    model = PossessionLSTM(hidden_size=16, max_epochs=1)
    model.fit(synth_feats, synth_gt, synth_dmask)
    return model


@pytest.fixture(scope="module")
def synth_crf_lstm(synth_feats, synth_gt, synth_dmask):
    from src.possession_lstm import PossessionLSTM
    model = PossessionLSTM(hidden_size=16, max_epochs=1, use_crf=True)
    model.fit(synth_feats, synth_gt, synth_dmask)
    return model


@pytest.fixture(scope="module")
def synth_bw_lstm(synth_feats, synth_gt, synth_dmask):
    from src.possession_lstm import PossessionLSTM
    model = PossessionLSTM(hidden_size=16, max_epochs=1,
                           boundary_weight=5.0, boundary_radius=8)
    model.fit(synth_feats, synth_gt, synth_dmask)
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_lstm_fit_predict_shape(synth_feats, synth_dmask, synth_lstm):
    pred = synth_lstm.predict(synth_feats, synth_dmask)
    assert pred.shape == (len(synth_feats),)


def test_lstm_predict_labels_valid(synth_feats, synth_dmask, synth_lstm):
    pred = synth_lstm.predict(synth_feats, synth_dmask)
    assert set(pred.tolist()).issubset({"H", "A", "D"})  # X is never emitted


def test_lstm_dead_frames_are_D(synth_feats, synth_dmask, synth_lstm):
    pred = synth_lstm.predict(synth_feats, synth_dmask)
    assert (pred[synth_dmask] == "D").all()


def test_lstm_min_spell_filter(synth_feats, synth_dmask, synth_lstm):
    from src.possession_hmm import MIN_SPELL_FRAMES
    pred = synth_lstm.predict(synth_feats, synth_dmask)
    runs = _run_lengths(pred, include_labels={"H", "A"})
    assert all(r >= MIN_SPELL_FRAMES for r in runs), \
        f"Found spell shorter than {MIN_SPELL_FRAMES}: {min(runs) if runs else 'none'}"


def test_lstm_save_load(tmp_path, synth_feats, synth_dmask, synth_lstm):
    from src.possession_lstm import PossessionLSTM
    path = str(tmp_path / "lstm.pt")
    synth_lstm.save(path)
    model2 = PossessionLSTM.load(path)
    pred1 = synth_lstm.predict(synth_feats[:500], synth_dmask[:500])
    pred2 = model2.predict(synth_feats[:500], synth_dmask[:500])
    assert (pred1 == pred2).all()


def test_lstm_constructor_args():
    from src.possession_lstm import PossessionLSTM
    model = PossessionLSTM(hidden_size=64, max_epochs=5)
    assert model.hidden_size == 64
    assert model.max_epochs == 5


def test_lstm_larger_chunk_size_produces_valid_output(synth_feats, synth_dmask, synth_gt):
    """LSTM with chunk_size larger than dataset still produces valid predictions."""
    from src.possession_lstm import PossessionLSTM
    # 300 frames, chunk_size=1000 → single oversized chunk; tests the same
    # "chunk_size > len(data)" path without allocating an 8192-step sequence.
    feats, gt, dmask = synth_feats[:300], synth_gt[:300], synth_dmask[:300]
    model = PossessionLSTM(hidden_size=32, max_epochs=2, chunk_size=1000)
    model.fit(feats, gt, dmask)
    pred = model.predict(feats, dmask)
    assert pred.shape == (len(feats),)
    assert set(pred.tolist()).issubset({"H", "A", "D"})


def test_lstm_tune_returns_best_params(synth_feats, synth_gt, synth_dmask):
    """tune_lstm returns a dict with all expected keys including new boundary/crf fields."""
    from src.tune import tune_lstm
    result = tune_lstm(
        feats=synth_feats[:200],
        gt=synth_gt[:200],
        dmask=synth_dmask[:200],
        hidden_sizes=[16],
        max_epochs_values=[2],
        boundary_weight_values=[5.0],
        use_crf_values=[False],
    )
    for key in ("hidden_size", "max_epochs", "chunk_size",
                "boundary_weight", "use_crf", "spell_f1", "macro_f1"):
        assert key in result, f"missing key: {key}"
    assert result["hidden_size"] == 16
    assert result["max_epochs"] == 2
    assert result["boundary_weight"] == 5.0
    assert result["use_crf"] is False


def test_boundary_weights_marks_transition_frames():
    """Frames adjacent to a label transition receive the upweight; others stay 1.0."""
    from src.possession_lstm import _boundary_weights
    # Single chunk: labels 0,0,0,1,1,1 — transition at index 3
    y = np.array([[0, 0, 0, 1, 1, 1]], dtype=np.int64)
    w = _boundary_weights(y, radius=1, weight=5.0)
    assert w[0, 0] == 1.0   # far from transition
    assert w[0, 2] == 5.0   # within radius=1 of transition at index 3
    assert w[0, 3] == 5.0   # the transition frame itself
    assert w[0, 4] == 5.0   # within radius=1 on the other side
    assert w[0, 5] == 1.0   # far from transition


def test_boundary_weights_ignores_padding():
    """Padding frames (y==-1) never receive upweight."""
    from src.possession_lstm import _boundary_weights
    y = np.array([[0, 1, -1, -1]], dtype=np.int64)
    w = _boundary_weights(y, radius=2, weight=5.0)
    assert (w[0, 2:] == 1.0).all()


def test_boundary_weights_disabled_when_weight_one():
    from src.possession_lstm import _boundary_weights
    y = np.array([[0, 0, 1, 1]], dtype=np.int64)
    w = _boundary_weights(y, radius=1, weight=1.0)
    assert (w == 1.0).all()


def test_boundary_weight_fit_predict_valid(synth_feats, synth_dmask, synth_bw_lstm):
    """boundary_weight != 1.0 trains and predicts without error."""
    pred = synth_bw_lstm.predict(synth_feats, synth_dmask)
    assert pred.shape == (len(synth_feats),)
    assert set(pred.tolist()).issubset({"H", "A", "D"})


def test_boundary_weight_save_load_roundtrip(tmp_path, synth_feats, synth_dmask, synth_bw_lstm):
    from src.possession_lstm import PossessionLSTM
    path = str(tmp_path / "lstm_bw.pt")
    synth_bw_lstm.save(path)
    loaded = PossessionLSTM.load(path)
    assert loaded.boundary_weight == 5.0
    assert loaded.boundary_radius == 8
    pred1 = synth_bw_lstm.predict(synth_feats[:500], synth_dmask[:500])
    pred2 = loaded.predict(synth_feats[:500], synth_dmask[:500])
    assert (pred1 == pred2).all()


def test_crf_fit_predict_valid(synth_feats, synth_dmask, synth_crf_lstm):
    """use_crf=True trains (Viterbi) and produces valid labels."""
    pred = synth_crf_lstm.predict(synth_feats, synth_dmask)
    assert pred.shape == (len(synth_feats),)
    assert set(pred.tolist()).issubset({"H", "A", "D"})


def test_crf_dead_frames_are_D(synth_feats, synth_dmask, synth_crf_lstm):
    pred = synth_crf_lstm.predict(synth_feats, synth_dmask)
    assert (pred[synth_dmask] == "D").all()


def test_crf_save_load_roundtrip(tmp_path, synth_feats, synth_dmask, synth_crf_lstm):
    from src.possession_lstm import PossessionLSTM
    path = str(tmp_path / "lstm_crf.pt")
    synth_crf_lstm.save(path)
    loaded = PossessionLSTM.load(path)
    assert loaded.use_crf is True
    assert loaded._crf is not None
    pred1 = synth_crf_lstm.predict(synth_feats[:500], synth_dmask[:500])
    pred2 = loaded.predict(synth_feats[:500], synth_dmask[:500])
    assert (pred1 == pred2).all()


def test_crf_boundary_weight_affects_training(synth_feats, synth_gt, synth_dmask):
    """boundary_weight must affect CRF training — models with w=1.0 vs w=5.0 must differ.

    Uses 2 epochs — enough for gradient differences to diverge across
    the larger model (BiLSTM + attention + CRF).
    """
    from src.possession_lstm import PossessionLSTM
    import torch

    def _head_weights(bw):
        model = PossessionLSTM(hidden_size=16, max_epochs=2,
                               use_crf=True, boundary_weight=bw, boundary_radius=4)
        model.fit(synth_feats[:200], synth_gt[:200], synth_dmask[:200])
        return list(model._model.head.parameters())[0].detach().clone()

    w1 = _head_weights(1.0)
    w5 = _head_weights(5.0)
    assert not torch.allclose(w1, w5), \
        "boundary_weight has no effect on CRF training (weights are identical)"


# ---------------------------------------------------------------------------
# Phase 2A — Overlapping inference
# ---------------------------------------------------------------------------

def test_overlap_inference_valid_shape_and_labels(synth_feats, synth_dmask, synth_gt):
    """Overlapping inference (stride=chunk//2) produces correct shape and valid labels."""
    from src.possession_lstm import PossessionLSTM
    feats, gt, dmask = synth_feats[:400], synth_gt[:400], synth_dmask[:400]
    # smoke test — same-split intentional, checks shape/labels only not accuracy
    # chunk_size=128 → multiple overlapping windows across 400 frames
    model = PossessionLSTM(hidden_size=16, max_epochs=1, chunk_size=128, use_overlap=True)
    model.fit(feats, gt, dmask)
    pred = model.predict(feats, dmask)
    assert pred.shape == (400,)
    assert set(pred.tolist()).issubset({"H", "A", "D"})


def test_overlap_inference_dead_frames_remain_D(synth_feats, synth_dmask, synth_gt):
    """Dead-ball override still applied after overlapping inference."""
    from src.possession_lstm import PossessionLSTM
    feats, gt, dmask = synth_feats[:400], synth_gt[:400], synth_dmask[:400]
    # smoke test — same-split intentional, checks dead-mask override only not accuracy
    model = PossessionLSTM(hidden_size=16, max_epochs=1, chunk_size=128, use_overlap=True)
    model.fit(feats, gt, dmask)
    pred = model.predict(feats, dmask)
    assert (pred[dmask[:400]] == "D").all()


def test_overlap_inference_uses_half_chunk_stride(synth_feats, synth_gt, synth_dmask):
    """predict() uses stride=chunk_size//2, making ~2x more forward passes than non-overlapping.

    Counts actual calls to model.forward — the clearest proof the overlap stride is active.
    """
    import math
    from src.possession_lstm import PossessionLSTM

    N = 200
    C = 40   # chunk_size; stride should be 20
    feats, gt, dmask = synth_feats[:N], synth_gt[:N], synth_dmask[:N]

    model = PossessionLSTM(hidden_size=8, max_epochs=1, chunk_size=C, use_overlap=True)
    model.fit(feats, gt, dmask)

    call_count = [0]
    orig_forward = model._model.forward

    def counting_forward(x):
        call_count[0] += 1
        return orig_forward(x)

    model._model.forward = counting_forward
    try:
        model.predict(feats, dmask)
    finally:
        model._model.forward = orig_forward

    stride = C // 2  # = 20
    expected = math.ceil(N / stride)  # ceil(200/20) = 10
    assert call_count[0] == expected, (
        f"Expected {expected} forward passes with stride={stride} (overlap, chunk={C}), "
        f"got {call_count[0]} — stride may still be chunk_size ({C}) = non-overlapping"
    )


# ---------------------------------------------------------------------------
# Phase 2B — Cosine LR scheduling
# ---------------------------------------------------------------------------

def test_cosine_lr_steps_once_per_epoch(synth_feats, synth_gt, synth_dmask):
    """CosineAnnealingLR.step() is called exactly once per epoch during fit()."""
    from unittest.mock import patch
    import torch.optim.lr_scheduler as sched
    from src.possession_lstm import PossessionLSTM

    step_count = []
    orig_step = sched.CosineAnnealingLR.step

    def counting_step(self):
        step_count.append(1)
        return orig_step(self)

    n_epochs = 3
    with patch.object(sched.CosineAnnealingLR, "step", counting_step):
        model = PossessionLSTM(hidden_size=8, max_epochs=n_epochs)
        model.fit(synth_feats[:80], synth_gt[:80], synth_dmask[:80])

    # PyTorch's LR scheduler calls step() once during __init__ (sets initial LR),
    # plus once per epoch during training → n_epochs + 1 total.
    assert len(step_count) == n_epochs + 1, (
        f"Expected {n_epochs + 1} scheduler.step() calls "
        f"(1 init + {n_epochs} training), got {len(step_count)}"
    )


def test_cosine_lr_no_constructor_change():
    """Constructor signature unchanged — no new public params added."""
    from src.possession_lstm import PossessionLSTM
    import inspect
    sig = inspect.signature(PossessionLSTM.__init__)
    params = set(sig.parameters) - {"self"}
    expected = {"hidden_size", "max_epochs", "chunk_size",
                "boundary_weight", "boundary_radius", "use_crf",
                "use_attention", "use_overlap", "lr", "seed", "save_best"}
    assert params == expected, f"Unexpected constructor params: {params - expected}"


# ---------------------------------------------------------------------------
# Phase 2C — Self-attention layer
# ---------------------------------------------------------------------------

def test_attention_layer_exists_in_bilstm():
    """_BiLSTM has an 'attn' nn.MultiheadAttention attribute."""
    import torch.nn as nn
    from src.possession_lstm import _BiLSTM
    model = _BiLSTM(n_features=10, hidden_size=32)
    assert hasattr(model, "attn"), "_BiLSTM missing 'attn' attribute"
    assert isinstance(model.attn, nn.MultiheadAttention)


def test_attention_embed_dim_and_heads():
    """_BiLSTM attention uses embed_dim=hidden_size*2 and num_heads=4."""
    from src.possession_lstm import _BiLSTM
    model = _BiLSTM(n_features=10, hidden_size=64)
    assert model.attn.embed_dim == 128        # 64 * 2
    assert model.attn.num_heads == 4


def test_attention_forward_shape():
    """_BiLSTM with attention still produces (B, T, C) output."""
    import torch
    from src.possession_lstm import _BiLSTM, _N_CLASSES
    model = _BiLSTM(n_features=10, hidden_size=32)
    x = torch.randn(2, 50, 10)
    out = model(x)
    assert out.shape == (2, 50, _N_CLASSES)


def test_attention_fit_predict_valid(synth_feats, synth_dmask, synth_gt):
    """LSTM with attention layer trains and predicts valid labels."""
    from src.possession_lstm import PossessionLSTM
    feats, gt, dmask = synth_feats[:300], synth_gt[:300], synth_dmask[:300]
    # smoke test — same-split intentional, checks shape/labels only not accuracy
    model = PossessionLSTM(hidden_size=32, max_epochs=1, use_attention=True)
    model.fit(feats, gt, dmask)
    pred = model.predict(feats, dmask)
    assert pred.shape == (300,)
    assert set(pred.tolist()).issubset({"H", "A", "D"})


def _run_lengths(labels, include_labels):
    runs = []
    i = 0
    while i < len(labels):
        if labels[i] in include_labels:
            j = i + 1
            while j < len(labels) and labels[j] == labels[i]:
                j += 1
            runs.append(j - i)
            i = j
        else:
            i += 1
    return runs
