"""HMM-based possession classifier.

Three hidden states: H (home), A (away), D (dead ball).
Features are modelled as diagonal-covariance Gaussians per state.
Viterbi decoding produces the globally most-likely label sequence.
Dead-ball frames override the HMM output.
Short H/A spells (<MIN_SPELL_FRAMES) are merged into their neighbours.
"""

import numpy as np
import joblib
from hmmlearn import hmm

MIN_SPELL_FRAMES = 10   # possessions shorter than 0.4 s at 25 fps are artefacts

_LABEL_TO_IDX = {"H": 0, "A": 1, "D": 2}
_IDX_TO_LABEL = {0: "H", 1: "A", 2: "D"}
_N_STATES = 3


class PossessionHMM:
    """Gaussian HMM for frame-level possession labelling."""

    def __init__(self, n_iter: int = 50):
        self.n_iter = n_iter
        self._model = None

    def fit(self, features: np.ndarray, gt_labels: np.ndarray,
            dead_mask: np.ndarray) -> "PossessionHMM":
        """Train on frames where gt_labels != 'X'.

        Parameters
        ----------
        features  : (N, 9) float32
        gt_labels : (N,) '<U1'  H/A/D/X
        dead_mask : (N,) bool   — dead ball mask (not used for training, kept for API consistency)
        """
        scored = gt_labels != "X"
        X_train = features[scored].astype(np.float64)
        y_train = gt_labels[scored]

        # Transition matrix from GT label sequence (Laplace smoothed)
        trans = np.ones((_N_STATES, _N_STATES))
        prev = None
        for label in y_train:
            idx = _LABEL_TO_IDX[label]
            if prev is not None:
                trans[prev, idx] += 1
            prev = idx
        trans = trans / trans.sum(axis=1, keepdims=True)

        # Start probabilities from GT distribution
        start = np.array([
            (y_train == "H").mean(),
            (y_train == "A").mean(),
            (y_train == "D").mean(),
        ]) + 1e-6
        start /= start.sum()

        model = hmm.GaussianHMM(
            n_components=_N_STATES,
            covariance_type="diag",
            n_iter=self.n_iter,
            init_params="mc",   # initialise means and covars from data
            params="mcst",      # learn means, covars, startprob, transmat
            random_state=42,    # reproducible EM initialization
        )
        model.startprob_ = start
        model.transmat_ = trans
        model.fit(X_train)
        self._model = model
        return self

    def predict(self, features: np.ndarray, dead_mask: np.ndarray) -> np.ndarray:
        """Viterbi decode → dead-mask override → min-spell filter.

        Parameters
        ----------
        features  : (N, F) float32
        dead_mask : (N,) bool

        Returns
        -------
        np.ndarray (N,) '<U1'  H/A/D  (X is never emitted)
        """
        labels = self._predict_raw(features, dead_mask)
        return _merge_short_spells(labels, MIN_SPELL_FRAMES)

    def _predict_raw(self, features: np.ndarray, dead_mask: np.ndarray) -> np.ndarray:
        """Viterbi decode + dead-mask override, without the min-spell filter.

        Used by tuning scripts that apply _merge_short_spells with varying parameters
        without re-running the expensive Viterbi pass each time.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")

        state_seq = self._model.predict(features.astype(np.float64))
        labels = np.array([_IDX_TO_LABEL[s] for s in state_seq], dtype="<U1")
        labels[dead_mask] = "D"
        return labels

    def save(self, path: str) -> None:
        """Persist model to disk."""
        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: str) -> "PossessionHMM":
        """Load a previously saved model."""
        obj = cls()
        obj._model = joblib.load(path)
        return obj


def _merge_short_spells(labels: np.ndarray, min_frames: int) -> np.ndarray:
    """Replace H/A spells shorter than min_frames with their longer neighbour's label.

    Iterates until no short spells remain (rare cases require 2 passes when
    two short spells are adjacent).
    """
    out = labels.copy()
    N = len(out)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < N:
            label = out[i]
            if label not in ("H", "A"):
                i += 1
                continue
            # Find end of this run
            j = i + 1
            while j < N and out[j] == label:
                j += 1
            if j - i < min_frames:
                left  = out[i - 1] if i > 0 else "D"
                right = out[j]     if j < N else "D"
                # Prefer a non-X neighbour; fall back to "D"
                replacement = right if right not in ("X",) else left
                if replacement == "X":
                    replacement = "D"
                out[i:j] = replacement
                changed = True
            i = j
    return out
