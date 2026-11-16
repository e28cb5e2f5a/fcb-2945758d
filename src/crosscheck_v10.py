"""Cross-check v10 winning config (h=128, e=20, bw=5.0, msf=100) on G1 and G2 folds.

Purpose: confirm the G3 LOO signal (+0.019 over v8) is not noise.
Selection was already made on G3 — these folds are cross-checks only.

v8 baselines (14-feat, corrected GT, §C11):
  G1: 0.4999
  G2: 0.6028
  G3: 0.5325 (selection fold — already run)

Usage
-----
    python3.11 src/crosscheck_v10.py
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

# ---------------------------------------------------------------------------
# Winning config (selected by G3 LOO)
# ---------------------------------------------------------------------------
HIDDEN       = 128
EPOCHS       = 20
CHUNK        = 2048
BW           = 5.0
MSF          = 100
CRF          = False

# v8 baselines for comparison
V8 = {"G1": 0.4999, "G2": 0.6028, "G3": 0.5325}

PUBLIC_FRAMES = 7_500
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


def _load_game(game):
    print(f"  Loading {game['name']}...")
    data  = load_tracking(game["dir"])
    feats = extract_features(data)
    gt    = load_gt(game["dir"])
    n     = feats.shape[0]
    dmask = dead_ball_mask(game["events"], n_frames=n)
    return feats, gt, dmask


def _run_fold(train_data, val_data, val_name):
    all_feats = np.concatenate([d[0] for d in train_data])
    all_gt    = np.concatenate([d[1] for d in train_data])
    all_dmask = np.concatenate([d[2] for d in train_data])

    model = PossessionLSTM(
        hidden_size=HIDDEN, max_epochs=EPOCHS,
        chunk_size=CHUNK, boundary_weight=BW, use_crf=CRF,
    )
    model.fit(all_feats, all_gt, all_dmask)

    val_feats, val_gt, val_dmask = val_data
    n_pub      = min(PUBLIC_FRAMES, len(val_gt))
    raw_labels = model.predict(val_feats[:n_pub], val_dmask[:n_pub])
    labels     = _merge_short_spells(raw_labels.copy(), MSF)
    gt_codes   = pm.label_codes(val_gt[:n_pub])
    pred_codes = pm.label_codes(labels)
    result     = pm.evaluate(gt_codes, pred_codes)
    sf = float(result["spell_f1_mean"])
    mf = float(result["macro_f1"])
    v8 = V8[val_name]
    print(f"  {val_name}: spellF1={sf:.4f}  macroF1={mf:.4f}  "
          f"vs v8={v8:.4f}  delta={sf-v8:+.4f}")
    return sf, mf


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

    print(f"Cross-check: v10 (h={HIDDEN}, e={EPOCHS}, bw={BW}, msf={MSF})")
    print(f"Config selected on G3 LOO — G1/G2 are cross-checks only.")
    print()

    # G1 fold: train G2+G3, eval G1
    print("Fold G1 (train G2+G3, eval G1):")
    sf_g1, mf_g1 = _run_fold([game_data[1], game_data[2]], game_data[0], "G1")

    # G2 fold: train G1+G3, eval G2
    print("Fold G2 (train G1+G3, eval G2):")
    sf_g2, mf_g2 = _run_fold([game_data[0], game_data[2]], game_data[1], "G2")

    print()
    print("=" * 60)
    print("Summary (v10 vs v8 on all three folds):")
    print(f"  G3 (selection): v10=0.5517  v8={V8['G3']:.4f}  delta={0.5517-V8['G3']:+.4f}")
    print(f"  G1 (cross-chk): v10={sf_g1:.4f}  v8={V8['G1']:.4f}  delta={sf_g1-V8['G1']:+.4f}")
    print(f"  G2 (cross-chk): v10={sf_g2:.4f}  v8={V8['G2']:.4f}  delta={sf_g2-V8['G2']:+.4f}")
    wins = sum([0.5517 > V8["G3"], sf_g1 > V8["G1"], sf_g2 > V8["G2"]])
    print(f"  v10 wins {wins}/3 folds")
    print("=" * 60)

    return sf_g1, mf_g1, sf_g2, mf_g2


if __name__ == "__main__":
    main()
