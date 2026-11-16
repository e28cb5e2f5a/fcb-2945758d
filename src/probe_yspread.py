"""§11a probe: add y-spread features (spread_H_y, spread_A_y) to 18-feat base.

Derives std of player y-positions (cols 18-19) alongside the existing
x-spread features (cols 9-10). Tests whether 2-axis spread information
improves G3 LOO spellF1 over the v13 18-feat baseline.

v13 baseline: G3 spellF1 = 0.6000, msf=50.

Does NOT modify src/features.py — computes y-spread inline and stacks
onto the 18-feat output. Features.py is unchanged; cache is unaffected.

Usage
-----
    python3.11 src/probe_yspread.py
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
from src.features import extract_features
from src.dead_ball import dead_ball_mask
from src.possession_lstm import PossessionLSTM
from src.possession_hmm import _merge_short_spells

PITCH_WIDTH = 68.0

# Locked v13 config
HIDDEN  = 128
EPOCHS  = 20
CHUNK   = 2048
BW      = 5.0
MSF     = 50
CRF     = False

V13_G3  = 0.6000

PUBLIC_FRAMES      = 7_500
_DEFAULT_DATA_ROOT = os.path.join(ROOT, "data", "sample")


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


def _load_game_20feat(game):
    """Load game and append spread_H_y, spread_A_y as cols 18-19."""
    data  = load_tracking(game["dir"])
    feats = extract_features(data)   # (N, 18)

    home_xy = data["home_xy"]  # (N, N_PLAYERS_MAX, 2)
    away_xy = data["away_xy"]

    home_m_y = home_xy[:, :, 1] * PITCH_WIDTH   # (N, N_PLAYERS_MAX)
    away_m_y = away_xy[:, :, 1] * PITCH_WIDTH

    spread_H_y = np.nanstd(home_m_y, axis=1).astype(np.float32)   # (N,)
    spread_A_y = np.nanstd(away_m_y, axis=1).astype(np.float32)

    feats = np.hstack([feats,
                       spread_H_y[:, np.newaxis],
                       spread_A_y[:, np.newaxis]])   # (N, 20)

    gt    = load_gt(game["dir"])
    n     = feats.shape[0]
    dmask = dead_ball_mask(game["events"], n_frames=n)
    return feats, gt, dmask


def _run_g3_loo(game_data):
    train_games = [game_data[0], game_data[1]]
    val_game    = game_data[2]

    all_feats = np.concatenate([d[0] for d in train_games])
    all_gt    = np.concatenate([d[1] for d in train_games])
    all_dmask = np.concatenate([d[2] for d in train_games])

    model = PossessionLSTM(
        hidden_size=HIDDEN, max_epochs=EPOCHS,
        chunk_size=CHUNK, boundary_weight=BW, use_crf=CRF,
    )
    model.fit(all_feats, all_gt, all_dmask)

    val_feats, val_gt, val_dmask = val_game
    n_pub      = min(PUBLIC_FRAMES, len(val_gt))
    raw_labels = model.predict(val_feats[:n_pub], val_dmask[:n_pub])
    labels     = _merge_short_spells(raw_labels.copy(), MSF)
    gt_codes   = pm.label_codes(val_gt[:n_pub])
    pred_codes = pm.label_codes(labels)
    result     = pm.evaluate(gt_codes, pred_codes)
    return float(result["spell_f1_mean"]), float(result["macro_f1"])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    data_root = (args.data_root
                 or os.environ.get("PRACTICE_DATA_ROOT")
                 or _DEFAULT_DATA_ROOT)

    games = _build_games(data_root)
    print("Loading all three games (20-feat: 18-feat + spread_H_y + spread_A_y)...")
    game_data = [_load_game_20feat(g) for g in games]
    print(f"Feature shape: {game_data[0][0].shape}")
    print()

    print(f"§11a probe: 20-feat y-spread  (v13 baseline G3={V13_G3:.4f})")
    print(f"Config: h={HIDDEN}, e={EPOCHS}, bw={BW}, msf={MSF}")
    print()

    sf, mf = _run_g3_loo(game_data)
    delta = sf - V13_G3

    print()
    print("=" * 60)
    print(f"20-feat (+ spread_H_y, spread_A_y):")
    print(f"  G3 spellF1 = {sf:.4f}  macroF1 = {mf:.4f}")
    print(f"  vs v13 baseline = {V13_G3:.4f}  delta = {delta:+.4f}")
    decision = "ACCEPT — run cross-check" if delta > 0 else "REJECT — no G3 gain"
    print(f"  G3 gate: {decision}")
    print("=" * 60)


if __name__ == "__main__":
    main()
