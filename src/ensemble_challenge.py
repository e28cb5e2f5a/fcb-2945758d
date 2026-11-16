"""Vote-ensemble across multiple prediction CSVs, or probability-average across models.

Mode 1 (default): majority-vote per frame across an arbitrary number of prediction
CSVs produced by sweep_challenge.py. Ties are broken by the first (best)
input file's label. An optional min-spell-frames filter is applied after
voting to merge short H/A spells into their neighbours.

Mode 2 (--prob-avg): load each .pt model, run predict_proba() on the target game's
features, average the (N, 3) softmax probability matrices across models, then
argmax → dead-ball mask → msf filter → write CSV. Sidesteps the H/A→D tie-break
collapse that afflicts hard majority vote near H/A boundaries.

Usage (mode 1)
--------------
python3.11 src/ensemble_challenge.py \
    --inputs evaluation/results/config_A_seed0.csv \
             evaluation/results/config_A_seed1.csv \
             evaluation/results/config_A_seed2.csv \
    --output evaluation/results/ensemble_top3_v3.csv \
    --min-spell-frames 125

Usage (mode 2)
--------------
python3.11 src/ensemble_challenge.py \
    --prob-avg \
    --models models/lstm_bw3_v5.pt models/lstm_bw3_v5s1.pt \
    --predict 3 \
    --output evaluation/results/ensemble_prob_bw3_g3loo_v5.csv \
    --min-spell-frames 125
"""

import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _read_labels(path: str) -> np.ndarray:
    """Read a frame,label CSV and return a (N,) array of label strings."""
    labels = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["label"])
    return np.array(labels, dtype="<U1")


def _write_csv(path: str, labels: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "label"])
        for i, label in enumerate(labels):
            writer.writerow([i, label])


def majority_vote(all_labels: list) -> np.ndarray:
    """Majority vote per frame.  Ties resolved by first array's label.

    Parameters
    ----------
    all_labels : list of (N,) arrays, each dtype='<U1'

    Returns
    -------
    (N,) array of voted labels
    """
    n_frames = len(all_labels[0])
    result = np.empty(n_frames, dtype="<U1")
    for i in range(n_frames):
        votes = [arr[i] for arr in all_labels]
        count = Counter(votes)
        # most_common() is ordered — if tie, Python preserves insertion order
        # so first-seen label wins, which corresponds to the best single model
        winner = count.most_common(1)[0][0]
        result[i] = winner
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", nargs="+", default=None,
                        help="paths to prediction CSVs (at least 2; used without --prob-avg)")
    parser.add_argument("--output", required=True,
                        help="output CSV path")
    parser.add_argument("--min-spell-frames", type=int, default=None, metavar="N",
                        help="merge H/A spells shorter than N frames after voting "
                             "(default: no post-processing)")
    parser.add_argument("--prob-avg", action="store_true",
                        help="average softmax probabilities across models "
                             "instead of majority-voting label CSVs; "
                             "requires --models instead of --inputs")
    parser.add_argument("--models", nargs="+", default=None,
                        help="paths to .pt model files (used with --prob-avg)")
    parser.add_argument("--train", nargs="+", type=int, choices=[1, 2, 3],
                        default=None,
                        help="game numbers used for LOO training (to load "
                             "the correct predict game's features); "
                             "e.g. --train 1 2 means predict game 3")
    parser.add_argument("--predict", type=int, choices=[1, 2, 3],
                        default=None,
                        help="game number to predict (LOO)")
    args = parser.parse_args()

    # Mode guards
    if args.prob_avg and (args.models is None or args.predict is None):
        parser.error("--prob-avg requires --models and --predict")
    if args.prob_avg and args.inputs is not None:
        parser.error("--prob-avg is incompatible with --inputs; "
                     "use --models instead")
    if not args.prob_avg and args.inputs is None:
        parser.error("--inputs is required unless --prob-avg is set")

    # ------------------------------------------------------------------ #
    #  Mode 2: probability averaging                                       #
    # ------------------------------------------------------------------ #
    if args.prob_avg:
        from src.possession_lstm import PossessionLSTM
        from src.pipeline import _load_game, GAME_DIRS

        predict_dir = GAME_DIRS[args.predict]
        print(f"Loading features for Game {args.predict}...")
        pred_feats, _, pred_dmask, _ = _load_game(predict_dir)

        prob_sum = None
        for model_path in args.models:
            print(f"  Loading model {model_path}...")
            model = PossessionLSTM.load(model_path)
            proba = model.predict_proba(pred_feats)  # (N, 3)
            if prob_sum is None:
                prob_sum = proba.copy()
            else:
                prob_sum += proba
            print(f"    {proba.shape[0]:,} frames processed")

        avg_proba = prob_sum / len(args.models)        # (N, 3)
        idx_map = {0: "H", 1: "A", 2: "D"}
        preds = avg_proba.argmax(axis=1)
        labels = np.array([idx_map[int(p)] for p in preds], dtype="<U1")

        # Apply dead-ball mask override
        labels[pred_dmask] = "D"

        # Apply min-spell filter
        if args.min_spell_frames is not None:
            from src.possession_hmm import _merge_short_spells
            labels = _merge_short_spells(labels, args.min_spell_frames)
            print(f"Applied min_spell_frames={args.min_spell_frames}")

        _write_csv(args.output, labels)
        counts = {l: int((labels == l).sum()) for l in ("H", "A", "D", "X")}
        spells = sum(
            1 for i in range(1, len(labels))
            if labels[i] in ("H", "A") and labels[i] != labels[i - 1]
        ) + (1 if labels[0] in ("H", "A") else 0)
        print(f"Written: {args.output}")
        print(f"  H={counts['H']:,}  A={counts['A']:,}  D={counts['D']:,}  "
              f"X={counts['X']:,}  spells={spells}")
        return

    # ------------------------------------------------------------------ #
    #  Mode 1: majority vote across label CSVs                            #
    # ------------------------------------------------------------------ #
    if len(args.inputs) < 2:
        parser.error("--inputs requires at least 2 files for a meaningful vote")

    print(f"Ensemble: {len(args.inputs)} inputs → {args.output}")
    all_labels = []
    for path in args.inputs:
        labels = _read_labels(path)
        all_labels.append(labels)
        print(f"  Loaded {len(labels):,} frames from {path}")

    n_frames = len(all_labels[0])
    for i, labels in enumerate(all_labels[1:], 1):
        if len(labels) != n_frames:
            raise ValueError(
                f"Frame count mismatch: {args.inputs[0]} has {n_frames} frames "
                f"but {args.inputs[i]} has {len(labels)} frames."
            )

    print(f"Voting across {len(all_labels)} models on {n_frames:,} frames...")
    voted = majority_vote(all_labels)

    if args.min_spell_frames is not None:
        from src.possession_hmm import _merge_short_spells
        voted = _merge_short_spells(voted, args.min_spell_frames)
        print(f"Applied min_spell_frames={args.min_spell_frames}")

    _write_csv(args.output, voted)

    counts = {l: int((voted == l).sum()) for l in ("H", "A", "D", "X")}
    spells = 0
    prev = None
    for lbl in voted:
        if lbl in ("H", "A") and lbl != prev:
            spells += 1
        prev = lbl
    print(f"Written: {args.output}")
    print(f"  H={counts['H']:,}  A={counts['A']:,}  D={counts['D']:,}  "
          f"X={counts['X']:,}  spells={spells}")


if __name__ == "__main__":
    main()
