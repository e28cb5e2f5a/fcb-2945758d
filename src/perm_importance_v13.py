"""Per-feature permutation importance for the v13 18-feat model.

Shuffles each feature column independently (5 reps), measures mean
spellF1 drop and per-label D-rate change on the G3 LOO fold
(public 7,500-frame slice). Focuses on cols 14-17 (new R13 proxies)
but reports all 18 features for comparison with C8 (14-feat baseline).

Usage
-----
    python3.11 src/perm_importance_v13.py
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

MSF           = 50
N_REPS        = 5
PUBLIC_FRAMES = 7_500
_DEFAULT_DATA_ROOT = os.path.join(ROOT, "data", "sample")
MODEL_PATH    = os.path.join(ROOT, "models", "diagonalmar.pt")


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


def _score(feats, gt, dmask, rng, col=None):
    """Predict with optional column shuffle; return (spellF1, D_rate)."""
    f = feats.copy()
    if col is not None:
        rng.shuffle(f[:, col])
    model = PossessionLSTM.load(MODEL_PATH)
    n_pub = min(PUBLIC_FRAMES, len(gt))
    raw   = model.predict(f[:n_pub], dmask[:n_pub])
    labels = _merge_short_spells(raw.copy(), MSF)
    gt_codes   = pm.label_codes(gt[:n_pub])
    pred_codes = pm.label_codes(labels)
    result = pm.evaluate(gt_codes, pred_codes)
    sf = float(result["spell_f1_mean"])
    d_rate = float(np.mean(np.array(labels) == "D"))
    return sf, d_rate


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    data_root = (args.data_root
                 or os.environ.get("PRACTICE_DATA_ROOT")
                 or _DEFAULT_DATA_ROOT)

    games = _build_games(data_root)
    # G3 is the validation fold (train G1+G2 → v13 model already saved)
    g3 = games[2]
    print(f"Loading G3 ({g3['name']})...")
    data  = load_tracking(g3["dir"])
    feats = extract_features(data)
    gt    = load_gt(g3["dir"])
    n     = feats.shape[0]
    dmask = dead_ball_mask(g3["events"], n_frames=n)
    print(f"  {n:,} frames, {feats.shape[1]} features")
    print(f"  Model: {MODEL_PATH}")
    print()

    rng = np.random.default_rng(42)

    # Baseline (no shuffle)
    sf_base, dr_base = _score(feats, gt, dmask, rng, col=None)
    print(f"Baseline: spellF1={sf_base:.4f}  D-rate={dr_base:.3f}")
    print()

    results = []
    for col, name in enumerate(FEATURE_NAMES):
        drops, dr_changes = [], []
        for rep in range(N_REPS):
            sf, dr = _score(feats, gt, dmask, rng, col=col)
            drops.append(sf_base - sf)
            dr_changes.append(dr - dr_base)
        mean_drop = float(np.mean(drops))
        mean_dr   = float(np.mean(dr_changes))
        marker = " ◄ NEW R13" if col >= 14 else ""
        print(f"  col {col:2d} {name:<22}  drop={mean_drop:+.4f}  ΔD-rate={mean_dr:+.4f}{marker}")
        results.append((col, name, mean_drop, mean_dr))

    # Sort by drop descending
    results_sorted = sorted(results, key=lambda r: -r[2])

    print()
    print("=" * 72)
    print(f"{'col':<4} {'feature':<22} {'sf_drop':>8} {'ΔD-rate':>9}  note")
    print("-" * 72)
    for col, name, drop, dr in results_sorted:
        new_tag = "NEW R13" if col >= 14 else ""
        print(f"{col:<4} {name:<22} {drop:>+8.4f} {dr:>+9.4f}  {new_tag}")
    print(f"{'—':<4} {'baseline':<22} {'0.0000':>8} {'0.0000':>9}")
    print(f"      spellF1={sf_base:.4f}  D-rate={dr_base:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
