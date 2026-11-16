"""Phase 2 ablation experiments.

Measures the individual contribution of each improvement by training on
Games 1+2 and evaluating on Game 3's public 5-min slice (first 7,500 frames).

Ablation variants
-----------------
  baseline : no overlap, no cosine LR, no attention
  A        : overlap inference only          (stride = chunk//2)
  B        : cosine LR only
  C        : single-head self-attention only
  combined : A + B + C  (= current possession_lstm.py)

Usage
-----
    python3.11 src/phase2_ablations.py
    python3.11 src/phase2_ablations.py 256 40 2048 5.0  # hidden epochs chunk bw
"""

import csv
import datetime
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import possession_metrics as pm
from src.sample_loader import load_tracking, load_gt
from src.features import extract_features
from src.dead_ball import dead_ball_mask
from src.possession_lstm import (
    _boundary_weights, _chunk,
    _LABEL_TO_IDX, _IDX_TO_LABEL, _N_CLASSES, _LR, _SEED,
)
from src.possession_hmm import MIN_SPELL_FRAMES, _merge_short_spells

# ---------------------------------------------------------------------------
# Network variants
# ---------------------------------------------------------------------------

class _BiLSTMBase(nn.Module):
    """Two-layer BiLSTM → linear head. No self-attention."""
    def __init__(self, n_features: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=2, batch_first=True, bidirectional=True, dropout=0.2,
        )
        self.head = nn.Linear(hidden_size * 2, _N_CLASSES)

    def forward(self, x):           # x: (B, T, F)
        out, _ = self.lstm(x)       # (B, T, 2*H)
        return self.head(out)       # (B, T, C)


class _BiLSTMAttn(nn.Module):
    """Two-layer BiLSTM → single-head self-attention → linear head."""
    def __init__(self, n_features: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=2, batch_first=True, bidirectional=True, dropout=0.2,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, num_heads=4, batch_first=True,
        )
        self.head = nn.Linear(hidden_size * 2, _N_CLASSES)

    def forward(self, x):                       # x: (B, T, F)
        out, _ = self.lstm(x)                   # (B, T, 2*H)
        out, _ = self.attn(out, out, out)       # (B, T, 2*H)
        return self.head(out)                   # (B, T, C)


# ---------------------------------------------------------------------------
# Training and inference helpers
# ---------------------------------------------------------------------------

def _train(net_class, features: np.ndarray, gt_labels: np.ndarray,
           dead_mask: np.ndarray,
           hidden_size: int, max_epochs: int, chunk_size: int,
           boundary_weight: float, use_cosine_lr: bool) -> nn.Module:
    torch.manual_seed(_SEED)
    np.random.seed(_SEED)

    scored = gt_labels != "X"
    X = features[scored].astype(np.float32)
    y = np.array([_LABEL_TO_IDX[l] for l in gt_labels[scored]], dtype=np.int64)

    n_features = X.shape[1]
    model = net_class(n_features, hidden_size)

    X_chunks, y_chunks = _chunk(X, y, chunk_size)
    bw = _boundary_weights(y_chunks, radius=8, weight=boundary_weight)

    dataset = TensorDataset(
        torch.from_numpy(X_chunks),
        torch.from_numpy(y_chunks),
        torch.from_numpy(bw),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(_SEED))

    optimizer = torch.optim.Adam(model.parameters(), lr=_LR)
    if use_cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs
        )

    model.train()
    for _ in range(max_epochs):
        for xb, yb, wb in loader:
            optimizer.zero_grad()
            emissions = model(xb)
            flat_e = emissions.reshape(-1, _N_CLASSES)
            flat_y = yb.reshape(-1)
            flat_w = wb.reshape(-1)
            valid = flat_y != -1
            per_frame = F.cross_entropy(flat_e, flat_y, ignore_index=-1,
                                        reduction="none")
            loss = (per_frame * flat_w * valid.float()).sum() / \
                   valid.float().sum().clamp(min=1)
            loss.backward()
            optimizer.step()
        if use_cosine_lr:
            scheduler.step()

    model.eval()
    return model


def _predict(model: nn.Module, features: np.ndarray, dead_mask: np.ndarray,
             chunk_size: int, use_overlap: bool) -> np.ndarray:
    N = len(features)
    X = torch.from_numpy(features.astype(np.float32))

    stride = max(1, chunk_size // 2) if use_overlap else chunk_size
    logit_sum = np.zeros((N, _N_CLASSES), dtype=np.float32)
    counts = np.zeros(N, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, N, stride):
            end = min(start + chunk_size, N)
            chunk = X[start:end].unsqueeze(0)       # (1, T, F)
            em = model(chunk)                        # (1, T, C)
            probs = F.softmax(em[0], dim=-1).numpy()[:end - start]
            logit_sum[start:end] += probs
            counts[start:end] += 1

    averaged = logit_sum / counts[:, None].clip(min=1)
    preds = averaged.argmax(axis=1).astype(np.int64)
    labels = np.array([_IDX_TO_LABEL[int(p)] for p in preds], dtype="<U1")
    labels[dead_mask] = "D"
    return _merge_short_spells(labels, MIN_SPELL_FRAMES)


# ---------------------------------------------------------------------------
# Data and logging
# ---------------------------------------------------------------------------

_DEFAULT_DATA_ROOT = os.path.join(ROOT, "data", "sample")


def _build_games(data_root: str) -> list:
    """Return GAMES list for data_root.

    Expects subdirectories Sample_Game_1, Sample_Game_2, Sample_Game_3.
    Events file is auto-detected: .csv preferred, falls back to .json.
    Override via --data-root or PRACTICE_DATA_ROOT env var.
    """
    games = []
    for i, name in enumerate(["Sample_Game_1", "Sample_Game_2", "Sample_Game_3"], 1):
        d = os.path.join(data_root, name)
        csv_path  = os.path.join(d, f"{name}_RawEventsData.csv")
        json_path = os.path.join(d, f"{name}_events.json")
        events = csv_path if os.path.exists(csv_path) else json_path
        games.append({"name": f"Game{i}", "dir": d, "events": events})
    return games


GAMES = _build_games(_DEFAULT_DATA_ROOT)
PUBLIC_FRAMES = 7_500
LOG_PATH   = os.path.join(ROOT, "docs", "overnight_log.md")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def _log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_PATH, "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def _write_csv(path: str, labels: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "label"])
        for i, lbl in enumerate(labels):
            writer.writerow([i, lbl])


def _load_game(game: dict):
    data  = load_tracking(game["dir"])
    feats = extract_features(data)
    gt    = load_gt(game["dir"])
    n     = feats.shape[0]
    dmask = dead_ball_mask(game["events"], n_frames=n)
    return feats, gt, dmask


def _score(labels: np.ndarray, gt: np.ndarray, n_pub: int):
    gt_codes   = pm.label_codes(gt[:n_pub])
    pred_codes = pm.label_codes(labels)
    result = pm.evaluate(gt_codes, pred_codes)
    return float(result["spell_f1_mean"]), float(result["macro_f1"])


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

VARIANTS = [
    # (label,           net_class,     cosine_lr, overlap)
    ("baseline",        _BiLSTMBase,   False,     False),
    ("A_overlap",       _BiLSTMBase,   False,     True),
    ("B_cosine_lr",     _BiLSTMBase,   True,      False),
    ("C_attention",     _BiLSTMAttn,   False,     False),
    ("combined_ABC",    _BiLSTMAttn,   True,      True),
]


def main(hidden_size: int = 128, max_epochs: int = 40,
         chunk_size: int = 2048, boundary_weight: float = 5.0):
    _log("=== Phase 2: Architecture ablations ===")
    _log(f"  hidden={hidden_size} epochs={max_epochs} chunk={chunk_size} bw={boundary_weight}")

    print("Loading all three games...")
    game_data  = [_load_game(g) for g in GAMES]
    train_data = [game_data[0], game_data[1]]   # Games 1+2
    val_feats, val_gt, val_dmask = game_data[2] # Game 3
    n_pub = min(PUBLIC_FRAMES, len(val_gt))
    _log("  Data loaded.")

    all_feats = np.concatenate([d[0] for d in train_data])
    all_gt    = np.concatenate([d[1] for d in train_data])

    kw = dict(
        hidden_size=hidden_size, max_epochs=max_epochs,
        chunk_size=chunk_size, boundary_weight=boundary_weight,
    )

    results = {}
    for label, net_class, cosine_lr, overlap in VARIANTS:
        _log(f"  --- {label}: training on Games 1+2 ---")
        model = _train(net_class, all_feats, all_gt,
                       dead_mask=np.zeros(len(all_feats), dtype=bool),
                       use_cosine_lr=cosine_lr, **kw)
        labels = _predict(model, val_feats[:n_pub], val_dmask[:n_pub],
                          chunk_size=chunk_size, use_overlap=overlap)

        csv_path = os.path.join(RESULTS_DIR, f"lstm_{label}.csv")
        _write_csv(csv_path, labels)

        sf, mf = _score(labels, val_gt, n_pub)
        results[label] = (sf, mf)
        _log(f"  {label}: spellF1={sf:.4f}  macroF1={mf:.4f}  ({csv_path})")

    _log("=" * 60)
    _log("## Phase 2 complete")
    _log(f"  baseline (no A/B/C):   spellF1={results['baseline'][0]:.4f}  macroF1={results['baseline'][1]:.4f}")
    _log(f"  A (overlap only):      spellF1={results['A_overlap'][0]:.4f}  macroF1={results['A_overlap'][1]:.4f}")
    _log(f"  B (cosine LR only):    spellF1={results['B_cosine_lr'][0]:.4f}  macroF1={results['B_cosine_lr'][1]:.4f}")
    _log(f"  C (attention only):    spellF1={results['C_attention'][0]:.4f}  macroF1={results['C_attention'][1]:.4f}")
    _log(f"  Combined A+B+C:        spellF1={results['combined_ABC'][0]:.4f}  macroF1={results['combined_ABC'][1]:.4f}")
    _log("=" * 60)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None,
                        help="root dir containing Sample_Game_1/2/3 "
                             "(default: $PRACTICE_DATA_ROOT or data/sample)")
    parser.add_argument("--hidden-size",     type=int,   default=128)
    parser.add_argument("--epochs",          type=int,   default=40)
    parser.add_argument("--chunk-size",      type=int,   default=2048)
    parser.add_argument("--boundary-weight", type=float, default=5.0)
    args = parser.parse_args()

    data_root = (args.data_root
                 or os.environ.get("PRACTICE_DATA_ROOT")
                 or _DEFAULT_DATA_ROOT)
    GAMES = _build_games(data_root)

    main(hidden_size=args.hidden_size, max_epochs=args.epochs,
         chunk_size=args.chunk_size, boundary_weight=args.boundary_weight)
