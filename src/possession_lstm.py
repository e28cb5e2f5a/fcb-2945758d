"""Bidirectional LSTM possession classifier.

Same interface as PossessionHMM — drop-in replacement.
Two-layer BiLSTM → linear → [optional CRF] → H/A/D.
Dead-mask override and min-spell filter applied in predict().

Optional features
-----------------
boundary_weight / boundary_radius
    Upweight loss on frames near possession transitions during training.
    Frames within `boundary_radius` of a label change receive
    `boundary_weight` × the standard cross-entropy loss.  Default 5×
    within 8 frames.  Set boundary_weight=1.0 to disable.

use_crf
    Add a linear-chain CRF decode layer on top of the LSTM.  Training
    uses CRF NLL; inference uses Viterbi decoding instead of argmax.
    Both techniques target the same spell-boundary problem; they compose
    but also individually improve spellF1.  Requires pytorch-crf
    (`pip install pytorch-crf`).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.possession_hmm import MIN_SPELL_FRAMES, _merge_short_spells

_LABEL_TO_IDX = {"H": 0, "A": 1, "D": 2}
_IDX_TO_LABEL = {0: "H", 1: "A", 2: "D"}
_N_CLASSES = 3

_SEED = 42
_BATCH_FRAMES = 2048  # sequence chunk length — larger = more temporal context per forward pass
_LR = 1e-3


class _BiLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, use_attention: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self._use_attention = use_attention
        if use_attention:
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_size * 2,
                num_heads=4,
                batch_first=True,
            )
        self.head = nn.Linear(hidden_size * 2, _N_CLASSES)

    def forward(self, x):                     # x: (B, T, F)
        out, _ = self.lstm(x)                 # (B, T, 2*H)
        if self._use_attention:
            out, _ = self.attn(out, out, out) # (B, T, 2*H) single-head self-attention
        return self.head(out)                 # (B, T, C)


def _boundary_weights(y_chunks: np.ndarray, radius: int, weight: float) -> np.ndarray:
    """Per-frame loss weights that upweight frames near possession transitions.

    Parameters
    ----------
    y_chunks : (n_chunks, T) int64 — -1 = padding/ignore
    radius   : frames either side of a transition to upweight
    weight   : loss multiplier for transition-adjacent frames (>= 1.0)

    Returns
    -------
    (n_chunks, T) float32, all values >= 1.0
    """
    weights = np.ones_like(y_chunks, dtype=np.float32)
    if weight == 1.0:
        return weights

    valid = y_chunks != -1
    # A transition occurs at frame t when y[t] != y[t-1] and both are valid.
    transitions = np.zeros_like(y_chunks, dtype=bool)
    transitions[:, 1:] = (
        valid[:, 1:] & valid[:, :-1] & (y_chunks[:, 1:] != y_chunks[:, :-1])
    )

    # Dilate: any valid frame within `radius` of a transition gets upweighted.
    # sliding_window_view ships with numpy >= 1.20 (python 3.11 env).
    from numpy.lib.stride_tricks import sliding_window_view
    t_pad = np.pad(transitions, ((0, 0), (radius, radius)), mode="constant")
    windows = sliding_window_view(t_pad, window_shape=2 * radius + 1, axis=1)  # (n, T, 2r+1)
    near = windows.any(axis=-1)  # (n, T)

    weights[near & valid] = weight
    return weights


class PossessionLSTM:
    """Bidirectional LSTM for frame-level possession labelling."""

    def __init__(self, hidden_size: int = 128, max_epochs: int = 40,
                 chunk_size: int = _BATCH_FRAMES,
                 boundary_weight: float = 5.0, boundary_radius: int = 8,
                 use_crf: bool = False,
                 use_attention: bool = False, use_overlap: bool = False,
                 lr: float = _LR, seed: int = 42,
                 save_best: bool = False):
        self.hidden_size     = hidden_size
        self.max_epochs      = max_epochs
        self.chunk_size      = chunk_size
        self.boundary_weight = boundary_weight
        self.boundary_radius = boundary_radius
        self.use_crf         = use_crf
        self.use_attention   = use_attention
        self.use_overlap     = use_overlap
        self.lr              = lr
        self._seed           = seed
        self.save_best       = save_best
        self._model          = None
        self._crf            = None
        self._n_features     = None

    def fit(self, features: np.ndarray, gt_labels: np.ndarray,
            dead_mask: np.ndarray) -> "PossessionLSTM":
        """Train on frames where gt_labels != 'X'.

        Parameters
        ----------
        features  : (N, F) float32
        gt_labels : (N,) '<U1'  H/A/D/X
        dead_mask : (N,) bool   — kept for API consistency, not used in training
        """
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

        scored = gt_labels != "X"
        X = features[scored].astype(np.float32)
        y = np.array([_LABEL_TO_IDX[l] for l in gt_labels[scored]], dtype=np.int64)

        self._n_features = X.shape[1]
        model = _BiLSTM(self._n_features, self.hidden_size, self.use_attention)

        X_chunks, y_chunks = _chunk(X, y, self.chunk_size)
        bw = _boundary_weights(y_chunks, self.boundary_radius, self.boundary_weight)

        # Temporal split: first 80% of chronological chunks for training,
        # last 20% for held-out validation loss tracking.
        n_total = len(X_chunks)
        n_train = max(1, int(n_total * 0.8))
        X_tr, y_tr, bw_tr = X_chunks[:n_train], y_chunks[:n_train], bw[:n_train]
        X_val, y_val       = X_chunks[n_train:], y_chunks[n_train:]

        train_ds = TensorDataset(
            torch.from_numpy(X_tr),
            torch.from_numpy(y_tr),
            torch.from_numpy(bw_tr),
        )
        train_loader = DataLoader(
            train_ds, batch_size=32, shuffle=True,
            generator=torch.Generator().manual_seed(self._seed),
        )

        # Val loader: only built when there are held-out chunks.
        if n_total > n_train:
            val_ds = TensorDataset(
                torch.from_numpy(X_val),
                torch.from_numpy(y_val),
            )
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        else:
            import warnings
            warnings.warn(
                "PossessionLSTM: fewer than 5 chunks total — skipping val split.",
                stacklevel=2,
            )
            val_loader = None

        crf = None
        if self.use_crf:
            try:
                from torchcrf import CRF
            except ImportError as exc:
                raise ImportError(
                    "use_crf=True requires 'pytorch-crf': pip install pytorch-crf"
                ) from exc
            crf = CRF(_N_CLASSES, batch_first=True)

        params = list(model.parameters())
        if crf is not None:
            params += list(crf.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )

        model.train()
        if crf is not None:
            crf.train()

        best_val_loss = float("inf")
        best_state    = None

        for _ in range(self.max_epochs):
            for xb, yb, wb in train_loader:
                optimizer.zero_grad()
                emissions = model(xb)                        # (B, T, C)

                if crf is not None:
                    mask = yb != -1                          # (B, T) bool
                    yb_crf = yb.clone()
                    yb_crf[~mask] = 0                        # CRF requires valid indices
                    # Scale emissions by boundary weights so transition frames
                    # contribute more to the path score (same intent as weighted CE).
                    weighted_emissions = emissions * wb.unsqueeze(-1)  # (B, T, C)
                    loss = -crf(weighted_emissions, yb_crf, mask=mask, reduction="mean")
                else:
                    flat_e = emissions.reshape(-1, _N_CLASSES)
                    flat_y = yb.reshape(-1)
                    flat_w = wb.reshape(-1)
                    valid  = flat_y != -1
                    per_frame = F.cross_entropy(flat_e, flat_y,
                                                ignore_index=-1, reduction="none")
                    loss = (per_frame * flat_w * valid.float()).sum() / \
                           valid.float().sum().clamp(min=1)

                loss.backward()
                optimizer.step()
            scheduler.step()

            # Val loss — printed for monitoring; optionally checkpoint best epoch
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    val_losses = []
                    for xb, yb in val_loader:
                        em = model(xb)
                        flat_e = em.reshape(-1, _N_CLASSES)
                        flat_y = yb.reshape(-1)
                        l = F.cross_entropy(flat_e, flat_y, ignore_index=-1)
                        val_losses.append(l.item())
                    if val_losses:
                        mean_vl = np.mean(val_losses)
                        print(f"  epoch val_loss={mean_vl:.4f}")
                        if self.save_best and mean_vl < best_val_loss:
                            best_val_loss = mean_vl
                            best_state = {k: v.cpu().clone()
                                          for k, v in model.state_dict().items()}
                model.train()

        if self.save_best and best_state is not None:
            model.load_state_dict(best_state)
            print(f"  [save_best] restored best epoch (val_loss={best_val_loss:.4f})")

        model.eval()
        self._model = model
        if crf is not None:
            crf.eval()
            self._crf = crf
        return self

    def predict(self, features: np.ndarray, dead_mask: np.ndarray) -> np.ndarray:
        """Forward pass → dead-mask override → min-spell filter.

        Parameters
        ----------
        features  : (N, F) float32
        dead_mask : (N,) bool

        Returns
        -------
        np.ndarray (N,) '<U1'  H/A/D  (X is never emitted)
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")
        if features.shape[1] != self._n_features:
            raise ValueError(
                f"Feature dimension mismatch: model expects {self._n_features} "
                f"features, got {features.shape[1]}. "
                f"A model saved under a different feature config was likely loaded."
            )

        self._model.eval()
        N = len(features)
        X = torch.from_numpy(features.astype(np.float32))  # (N, F)

        stride = max(1, self.chunk_size // 2) if self.use_overlap else self.chunk_size
        logit_sum = np.zeros((N, _N_CLASSES), dtype=np.float32)
        counts    = np.zeros(N, dtype=np.float32)

        with torch.no_grad():
            for start in range(0, N, stride):
                end    = min(start + self.chunk_size, N)
                chunk  = X[start:end].unsqueeze(0)         # (1, T, F)
                em     = self._model(chunk)                 # (1, T, C)
                length = end - start
                if self._crf is not None:
                    # Average raw emissions across overlapping windows; final
                    # Viterbi pass happens once on the accumulated average.
                    logit_sum[start:end] += em[0].numpy()[:length]
                else:
                    import torch.nn.functional as F
                    probs = F.softmax(em[0], dim=-1).numpy()[:length]
                    logit_sum[start:end] += probs
                counts[start:end] += 1

        averaged = logit_sum / counts[:, None].clip(min=1)

        if self._crf is not None:
            avg_tensor = torch.from_numpy(averaged).unsqueeze(0)  # (1, N, C)
            decoded    = self._crf.decode(avg_tensor)
            preds      = np.array(decoded[0], dtype=np.int64)
        else:
            preds = averaged.argmax(axis=1).astype(np.int64)

        labels = np.array([_IDX_TO_LABEL[int(p)] for p in preds], dtype="<U1")
        labels[dead_mask] = "D"
        return _merge_short_spells(labels, MIN_SPELL_FRAMES)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return averaged softmax probabilities (N, 3) without
        applying dead-mask or min-spell filter.

        Column order: [H_prob, A_prob, D_prob].

        Parameters
        ----------
        features : (N, F) float32

        Returns
        -------
        np.ndarray (N, 3) float32, rows sum to 1.0
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict_proba()")
        if self._crf is not None:
            raise RuntimeError(
                "predict_proba() is not supported for CRF models: CRF decoding "
                "produces hard labels, not probabilities. Use use_crf=False."
            )

        self._model.eval()
        N = len(features)
        X = torch.from_numpy(features.astype(np.float32))  # (N, F)

        stride = max(1, self.chunk_size // 2) if self.use_overlap \
                 else self.chunk_size
        prob_sum = np.zeros((N, _N_CLASSES), dtype=np.float32)
        counts   = np.zeros(N, dtype=np.float32)

        with torch.no_grad():
            for start in range(0, N, stride):
                end   = min(start + self.chunk_size, N)
                chunk = X[start:end].unsqueeze(0)          # (1, T, F)
                em    = self._model(chunk)                  # (1, T, C)
                length = end - start
                probs = F.softmax(em[0], dim=-1).numpy()[:length]
                prob_sum[start:end] += probs
                counts[start:end] += 1

        return prob_sum / counts[:, None].clip(min=1)

    def save(self, path: str) -> None:
        """Persist model weights and config to disk."""
        payload = {
            "hidden_size":     self.hidden_size,
            "max_epochs":      self.max_epochs,
            "chunk_size":      self.chunk_size,
            "boundary_weight": self.boundary_weight,
            "boundary_radius": self.boundary_radius,
            "use_crf":         self.use_crf,
            "use_attention":   self.use_attention,
            "use_overlap":     self.use_overlap,
            "n_features":      self._n_features,
            "state_dict":      self._model.state_dict(),
        }
        if self._crf is not None:
            payload["crf_state_dict"] = self._crf.state_dict()
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str) -> "PossessionLSTM":
        """Load a previously saved model."""
        checkpoint = torch.load(path, weights_only=True)
        obj = cls(
            hidden_size     = checkpoint["hidden_size"],
            max_epochs      = checkpoint["max_epochs"],
            chunk_size      = checkpoint.get("chunk_size", _BATCH_FRAMES),
            boundary_weight = checkpoint.get("boundary_weight", 1.0),
            boundary_radius = checkpoint.get("boundary_radius", 8),
            use_crf         = checkpoint.get("use_crf", False),
            use_attention   = checkpoint.get("use_attention", False),
            use_overlap     = checkpoint.get("use_overlap", False),
        )
        obj._n_features = checkpoint["n_features"]
        obj._model = _BiLSTM(obj._n_features, obj.hidden_size, obj.use_attention)
        obj._model.load_state_dict(checkpoint["state_dict"])
        obj._model.eval()
        if obj.use_crf and "crf_state_dict" in checkpoint:
            from torchcrf import CRF
            obj._crf = CRF(_N_CLASSES, batch_first=True)
            obj._crf.load_state_dict(checkpoint["crf_state_dict"])
            obj._crf.eval()
        return obj


def _chunk(X: np.ndarray, y: np.ndarray, chunk_len: int):
    """Split (N, F) and (N,) into non-overlapping chunks of chunk_len.

    The last partial chunk is zero-padded to chunk_len.
    Returns (n_chunks, chunk_len, F) and (n_chunks, chunk_len).
    """
    N, F = X.shape
    n_chunks = (N + chunk_len - 1) // chunk_len
    X_pad = np.zeros((n_chunks * chunk_len, F), dtype=np.float32)
    y_pad = np.full(n_chunks * chunk_len, -1, dtype=np.int64)  # -1 = ignored by loss
    X_pad[:N] = X
    y_pad[:N] = y
    return (
        X_pad.reshape(n_chunks, chunk_len, F),
        y_pad.reshape(n_chunks, chunk_len),
    )
