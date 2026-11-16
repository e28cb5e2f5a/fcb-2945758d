import numpy as np
import pytest
from src.tagging_features import tagging_features

SMART_TAG_XML = "data/2027_challenge/tactical/BAR-ATH_tactical_smart_tagging.xml"


def test_shape():
    feats = tagging_features(SMART_TAG_XML, n_frames=199319)
    assert feats.shape == (199319, 4)
    assert feats.dtype == np.float32


def test_values_binary():
    feats = tagging_features(SMART_TAG_XML, n_frames=199319)
    assert set(np.unique(feats)).issubset({0.0, 1.0})


def test_set_piece_nonzero():
    # 59 SET PIECES instances → some frames must be 1
    feats = tagging_features(SMART_TAG_XML, n_frames=199319)
    assert feats[:, 3].sum() > 0
