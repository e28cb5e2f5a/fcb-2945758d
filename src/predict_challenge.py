"""Train on sample games, predict BAR-ATH challenge data, write a scoreable CSV.

Writes to evaluation/results/ so run_evaluation.py picks it up automatically.

Usage
-----
    # Train HMM on all 3 sample games, predict BAR-ATH
    python3.11 src/predict_challenge.py

    # Use LSTM (trains from scratch each run — use --load-model to skip)
    python3.11 src/predict_challenge.py --model lstm

    # Load a previously saved model
    python3.11 src/predict_challenge.py --load-model models/lstm_v2.pt --model lstm

    # Custom output name (goes into evaluation/results/<name>.csv)
    python3.11 src/predict_challenge.py --name hmm_challenge_v1
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.challenge_loader import load_challenge_tracking
from src.dead_ball import dead_ball_mask
from src.features import extract_features, FEATURE_NAMES
from src.pipeline import train_model, apply_graded_blocks, GAME_DIRS
from games import asset_paths, half_video_blocks, GAME
from src.challenge_loader import _CHALLENGE_DATA_DIR


def _build_dead_mask(n_frames: int, ball_xy: np.ndarray | None = None) -> np.ndarray:
    paths = asset_paths(game=GAME, source="tactical", data_dir=_CHALLENGE_DATA_DIR)
    xml_path = paths["smart_tagging"]
    mask = dead_ball_mask(xml_path, n_frames=n_frames, ball_xy=ball_xy)
    print(f"  Dead-ball frames: {mask.sum():,} / {n_frames:,} ({100 * mask.mean():.1f}%)")
    return mask


def _write_csv(path: str, labels: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "label"])
        for i, label in enumerate(labels):
            writer.writerow([i, label])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=["hmm", "lstm"], default="hmm")
    parser.add_argument("--train", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3],
                        help="sample game numbers to train on (default: all three)")
    parser.add_argument("--load-model", default=None, metavar="PATH",
                        help="skip training and load a saved model from PATH")
    parser.add_argument("--name", default=None,
                        help="output filename stem (default: <model>_challenge)")
    parser.add_argument("--save-model", default=None, metavar="PATH",
                        help="save the trained model to PATH after training")
    parser.add_argument("--min-spell-frames", type=int, default=None, metavar="N",
                        help="merge H/A spells shorter than N frames into neighbours (default: model default of 10)")
    parser.add_argument("--interp-ball", action="store_true",
                        help="linearly interpolate missing ball positions within each half-block")
    parser.add_argument("--tagging-feats", action="store_true",
                        help="append 4 binary smart-tagging features (requires retraining; "
                             "incompatible with --load-model)")
    parser.add_argument("--zero-feature-cols", default="",
                        help="comma-separated feature column indices to zero at inference "
                             "(e.g. '14,15'); useful for post-hoc ablation without retraining")
    args = parser.parse_args()

    if args.tagging_feats and args.load_model:
        raise SystemExit("--tagging-feats changes the feature dimension and requires retraining. "
                         "Remove --load-model or remove --tagging-feats.")

    out_name = args.name or f"{args.model}_challenge"
    out_path = os.path.join(ROOT, "evaluation", "results", f"{out_name}.csv")

    # ------------------------------------------------------------------
    # 1. Train or load model
    # ------------------------------------------------------------------
    if args.load_model:
        print(f"Loading model from {args.load_model}...")
        if args.model == "lstm":
            from src.possession_lstm import PossessionLSTM
            model = PossessionLSTM.load(args.load_model)
        else:
            from src.possession_hmm import PossessionHMM
            model = PossessionHMM.load(args.load_model)
        print("  Done.")
    else:
        train_dirs = [GAME_DIRS[g] for g in args.train]
        print(f"Training {args.model.upper()} on sample games {args.train}...")
        model = train_model(train_dirs, model_type=args.model, save_path=args.save_model)

    # ------------------------------------------------------------------
    # 2. Load BAR-ATH challenge tracking
    # ------------------------------------------------------------------
    print("Loading BAR-ATH challenge tracking...")
    data = load_challenge_tracking()

    ball_visible = ~np.isnan(data["ball_xy"][:, 0])
    n_frames = data["ball_xy"].shape[0]
    print(f"  Ball visible: {ball_visible.sum():,} / {n_frames:,} ({100 * ball_visible.mean():.1f}%)")

    if args.interp_ball:
        from src.ball_interp import interpolate_ball
        data = interpolate_ball(data)
        filled = (~np.isnan(data["ball_xy"][:, 0])).sum() - ball_visible.sum()
        print(f"  Ball after interpolation: {(~np.isnan(data['ball_xy'][:, 0])).sum():,} visible (+{filled:,} filled)")

    feats = extract_features(data)
    if args.tagging_feats:
        from src.tagging_features import tagging_features
        xml_path = asset_paths(game=GAME, source="tactical", data_dir=_CHALLENGE_DATA_DIR)["smart_tagging"]
        tag_feats = tagging_features(xml_path, n_frames=feats.shape[0])
        feats = np.hstack([feats, tag_feats])
        print(f"  Appended 4 tagging features → {feats.shape[1]} total")
    n_frames = feats.shape[0]
    print(f"  {n_frames:,} frames, {feats.shape[1]} features")

    # ------------------------------------------------------------------
    # 3. Dead-ball mask from Smart Tagging XML
    # ------------------------------------------------------------------
    print("Building dead-ball mask from Smart Tagging XML...")
    dmask = _build_dead_mask(n_frames, ball_xy=data["ball_xy"])

    # ------------------------------------------------------------------
    # 4. Predict
    # ------------------------------------------------------------------
    # A handful of frames at the end of each half have zero players tracked,
    # causing nanstd/nanmean in features.py to produce NaN. Replace with 0
    # before passing to the model; those frames land outside the graded blocks
    # and get X-masked anyway.
    nan_frames = np.isnan(feats).any(axis=1).sum()
    if nan_frames:
        print(f"  Imputing {nan_frames} NaN feature rows with 0 (outside graded blocks)")
        feats = np.where(np.isnan(feats), 0.0, feats)

    if args.zero_feature_cols:
        for col in args.zero_feature_cols.split(","):
            feats[:, int(col.strip())] = 0.0
        print(f"  Zeroed feature cols: {args.zero_feature_cols}")

    print("Predicting...")
    labels = model.predict(feats, dmask)

    if args.min_spell_frames is not None:
        from src.possession_hmm import _merge_short_spells
        labels = _merge_short_spells(labels, args.min_spell_frames)
        print(f"  Applied min_spell_frames={args.min_spell_frames}")

    # ------------------------------------------------------------------
    # 5. Mask frames outside graded half-blocks to X
    # ------------------------------------------------------------------
    blocks = half_video_blocks(GAME)
    labels = apply_graded_blocks(labels, blocks)
    graded = sum(last - first + 1 for _, first, last in blocks)
    print(f"  Graded half-blocks: {[(h, f, l) for h,f,l in blocks]}")
    print(f"  Graded frames: {graded:,}  |  X frames: {(labels == 'X').sum():,}")

    # ------------------------------------------------------------------
    # 6. Write CSV
    # ------------------------------------------------------------------
    _write_csv(out_path, labels)
    counts = {l: int((labels == l).sum()) for l in ("H", "A", "D", "X")}
    print(f"\nWritten: {out_path}")
    print(f"  H={counts['H']:,}  A={counts['A']:,}  D={counts['D']:,}  X={counts['X']:,}")
    print(f"\nTo score:")
    print(f"  cd evaluation && python3.11 run_evaluation.py")


if __name__ == "__main__":
    main()
