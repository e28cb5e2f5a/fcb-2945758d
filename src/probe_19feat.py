"""§10 probe: 19-feat = 18-feat (v13) + ball_visible at col 18.

§C11 showed 15-feat > 14-feat on all three corrected-GT LOO folds
(weighted delta +0.046). Round 13 built on the 14-feat base; this
probe tests whether restoring ball_visible on top of the 4 proxy
features (cols 14-17) improves G3 LOO further.

Does NOT modify src/features.py — appends ball_visible inline.

v13 baseline: G3 spellF1 = 0.6000, msf=50.

Usage
-----
    python3.11 src/probe_19feat.py
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


def _load_game_19feat(game):
    """Load game and append ball_visible as col 18 (19-feat total)."""
    data  = load_tracking(game["dir"])
    feats = extract_features(data)   # (N, 18)

    ball_xy = data["ball_xy"]
    ball_visible = (~np.isnan(ball_xy[:, 0])).astype(np.float32)   # (N,)

    feats = np.hstack([feats, ball_visible[:, np.newaxis]])   # (N, 19)

    gt    = load_gt(game["dir"])
    n     = feats.shape[0]
    dmask = dead_ball_mask(game["events"], n_frames=n)
    return feats, gt, dmask


def _run_g3_loo(game_data):
    train_games = [game_data[0], game_data[1]]   # G1, G2
    val_game    = game_data[2]                    # G3

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
    sf = float(result["spell_f1_mean"])
    mf = float(result["macro_f1"])

    # D-rate check: guard against ball_visible becoming a new D-proxy
    d_rate = float(np.mean(np.array(labels) == "D"))
    return sf, mf, d_rate


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    data_root = (args.data_root
                 or os.environ.get("PRACTICE_DATA_ROOT")
                 or _DEFAULT_DATA_ROOT)

    games = _build_games(data_root)
    print("Loading all three games (19-feat: 18-feat + ball_visible col 18)...")
    game_data = [_load_game_19feat(g) for g in games]
    print(f"Feature shape: {game_data[0][0].shape}")
    print()

    print(f"§10 probe: 19-feat  (v13 baseline G3={V13_G3:.4f}, msf={MSF})")
    print(f"Config: h={HIDDEN}, e={EPOCHS}, bw={BW}, msf={MSF}")
    print()

    sf, mf, d_rate = _run_g3_loo(game_data)
    delta = sf - V13_G3

    print()
    print("=" * 60)
    print(f"19-feat (18-feat + ball_visible):")
    print(f"  G3 spellF1 = {sf:.4f}  macroF1 = {mf:.4f}  D-rate = {d_rate:.3f}")
    print(f"  vs v13 baseline = {V13_G3:.4f}  delta = {delta:+.4f}")
    decision = "PASS G3 gate — run cross-check" if delta > 0 else "FAIL G3 gate — keep v13"
    print(f"  G3 gate: {decision}")
    print("=" * 60)


if __name__ == "__main__":
    main()
