import numpy as np
import pytest


def test_hmm_fit_predict_shape(synth_feats, synth_dmask, synth_hmm):
    pred = synth_hmm.predict(synth_feats, synth_dmask)
    assert pred.shape == (len(synth_feats),)


def test_hmm_predict_labels_valid(synth_feats, synth_dmask, synth_hmm):
    pred = synth_hmm.predict(synth_feats, synth_dmask)
    assert set(pred.tolist()).issubset({"H", "A", "D", "X"})


def test_hmm_dead_frames_are_D(synth_feats, synth_dmask, synth_hmm):
    pred = synth_hmm.predict(synth_feats, synth_dmask)
    assert (pred[synth_dmask] == "D").all()


def test_hmm_min_spell_filter(synth_feats, synth_dmask, synth_hmm):
    from src.possession_hmm import MIN_SPELL_FRAMES
    pred = synth_hmm.predict(synth_feats, synth_dmask)
    runs = _run_lengths(pred, include_labels={"H", "A"})
    assert all(r >= MIN_SPELL_FRAMES for r in runs), \
        f"Found spell shorter than {MIN_SPELL_FRAMES}: {min(runs) if runs else 'none'}"


def test_hmm_save_load(tmp_path, synth_feats, synth_dmask, synth_hmm):
    from src.possession_hmm import PossessionHMM
    path = str(tmp_path / "model.joblib")
    synth_hmm.save(path)
    model2 = PossessionHMM.load(path)
    pred1 = synth_hmm.predict(synth_feats[:500], synth_dmask[:500])
    pred2 = model2.predict(synth_feats[:500], synth_dmask[:500])
    assert (pred1 == pred2).all()


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
