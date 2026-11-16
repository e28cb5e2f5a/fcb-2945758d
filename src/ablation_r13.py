"""Round 13 one-out ablation: drop each of cols 14-17 individually.

Purpose: identify the marginal contribution of each tracking-proxy
feature added in Round 13 (§C13). Zeroes one column at a time;
no retraining of the base 18-feat model needed conceptually, but
each probe trains a fresh model so the LSTM can redistribute weight.

v13 baseline (all 4 cols present): G3 spellF1 = 0.6000, msf=50.

Usage
-----
    python3.11 src/ablation_r13.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import numpy as np
import possession_metrics as pm

from src.sample_loader import load_tracking, load_gt
from src.features import extract_features, FEATURE_NAMES
from src.dead_ball import dead_ball_mask
from src.possession_lstm import PossessionLSTM
from src.possession_hmm import _merge_short_spells

# Locked v13 config
HIDDEN  = 128
EPOCHS  = 20
CHUNK   = 2048
BW      = 5.0
MSF     = 50
CRF     = False

V13_G3  = 0.6000   # baseline — all 4 proxy cols present

PUBLIC_FRAMES      = 7_500
_DEFAULT_DATA_ROOT = os.path.join(ROOT, "data", "sample")

# Cols to ablate (one at a time)
ABLATE_COLS = [
    (14, "ball_in_final_third"),
    (15, "ball_vx"),
    (16, "set_piece_proxy"),
    (17, "h_centroid_adv_x"),
]


def _build_games(data_root):
    # Practice-data–only script. Sample_Game_N paths are intentional;
    # override via --data-root or PRACTICE_DATA_ROOT env var.
    games = []
    for i, name in enumerate(["Sample_Game_1", "Sample_Game_2", "Sample_Game_3"], 1):
        d = os.path.join(data_root, name)
        csv_path  = os.path.join(d, f"{name}_RawEventsData.csv")
        json_path = os.path.join(d, f"{name}_events.json")
        events = csv_path if os.path.exists(csv_path) else json_path
        games.append({"name": f"Game{i}", "dir": d, "events": events})
    return games


def _load_game(game):
    data  = load_tracking(game["dir"])
    feats = extract_features(data)
    gt    = load_gt(game["dir"])
    n     = feats.shape[0]
    dmask = dead_ball_mask(game["events"], n_frames=n)
    return feats, gt, dmask


def _run_g3_loo(game_data, drop_col):
    """Train on G1+G2, eval on G3, with column drop_col zeroed."""
    train_games = [game_data[0], game_data[1]]   # G1, G2
    val_game    = game_data[2]                    # G3

    def _zero(feats):
        f = feats.copy()
        if drop_col is not None:
            f[:, drop_col] = 0.0
        return f

    all_feats = np.concatenate([_zero(d[0]) for d in train_games])
    all_gt    = np.concatenate([d[1] for d in train_games])
    all_dmask = np.concatenate([d[2] for d in train_games])

    model = PossessionLSTM(
        hidden_size=HIDDEN, max_epochs=EPOCHS,
        chunk_size=CHUNK, boundary_weight=BW, use_crf=CRF,
    )
    model.fit(all_feats, all_gt, all_dmask)

    val_feats, val_gt, val_dmask = val_game
    n_pub      = min(PUBLIC_FRAMES, len(val_gt))
    raw_labels = model.predict(_zero(val_feats[:n_pub]), val_dmask[:n_pub])
    labels     = _merge_short_spells(raw_labels.copy(), MSF)
    gt_codes   = pm.label_codes(val_gt[:n_pub])
    pred_codes = pm.label_codes(labels)
    result     = pm.evaluate(gt_codes, pred_codes)
    return float(result["spell_f1_mean"])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    data_root = (args.data_root
                 or os.environ.get("PRACTICE_DATA_ROOT")
                 or _DEFAULT_DATA_ROOT)

    games = _build_games(data_root)
    print("Loading all three games...")
    game_data = [_load_game(g) for g in games]
    print()

    print(f"Round 13 one-out ablation  (v13 baseline G3={V13_G3:.4f}, msf={MSF})")
    print(f"Drop one column, retrain, eval G3 LOO.")
    print()

    results = []
    for col, name in ABLATE_COLS:
        print(f"Probe: drop col {col} ({name})...")
        sf = _run_g3_loo(game_data, drop_col=col)
        delta = sf - V13_G3
        print(f"  G3 spellF1 = {sf:.4f}  delta vs v13 = {delta:+.4f}")
        results.append((col, name, sf, delta))

    print()
    print("=" * 60)
    print(f"{'col':<4} {'feature':<22} {'G3_sf':<8} {'delta':>8}  {'impact'}")
    print("-" * 60)
    for col, name, sf, delta in sorted(results, key=lambda r: r[2]):
        impact = "highest loss" if delta == min(r[3] for r in results) else ""
        print(f"{col:<4} {name:<22} {sf:.4f}   {delta:>+.4f}  {impact}")
    print(f"{'—':<4} {'v13 baseline (all)':<22} {V13_G3:.4f}   {'—':>+6}")
    print("=" * 60)


if __name__ == "__main__":
    main()
