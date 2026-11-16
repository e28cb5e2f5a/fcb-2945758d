"""Grid-search MIN_SPELL_FRAMES × dead-ball buffer on a labeled game directory.

By default tunes on the first PUBLIC_FRAMES frames only (5 min = 7500 frames at 25 fps),
simulating BAR-ATH competition conditions where only a 5-min GT window is public.
Use --full-gt to evaluate the winning params against the full GT.

Logs every combination to MLflow experiment "possession-hmm-tune".

Usage
-----
    # practice on Game 3 public slice (default)
    python src/tune.py

    # evaluate on full Game 3 GT
    python src/tune.py --full-gt

    # Oct 20 BAR-ATH
    python src/tune.py --game-dir data/tactical --events-path data/tactical/BAR-ATH_tactical_smart_tagging.xml

    # or via env vars
    TUNE_GAME_DIR=data/tactical TUNE_EVENTS_PATH=data/tactical/... python src/tune.py
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import mlflow
import possession_metrics as pm

from src.sample_loader import load_tracking, load_gt
from src.features import extract_features, FEATURE_NAMES
from src.dead_ball import dead_ball_mask
from src.possession_hmm import PossessionHMM, _merge_short_spells

_DEFAULT_GAME3   = os.path.join(ROOT, "data", "sample", "Sample_Game_3")
_DEFAULT_EVENTS3 = os.path.join(_DEFAULT_GAME3, "Sample_Game_3_events.json")

PUBLIC_FRAMES    = 7_500   # 5 min × 25 fps — matches BAR-ATH public GT window size

MIN_SPELL_VALUES   = [5, 8, 10, 15, 20]
BUFFER_VALUES      = [2, 3, 5]

LSTM_HIDDEN_SIZES       = [64, 128, 256]
LSTM_MAX_EPOCHS         = [10, 20, 40]
LSTM_CHUNK_SIZES        = [512, 2048]
LSTM_BOUNDARY_WEIGHTS   = [1.0, 5.0]   # 1.0 = disabled (baseline), 5.0 = upweight transitions
LSTM_USE_CRF            = [False, True]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game-dir",
                        default=os.environ.get("TUNE_GAME_DIR", _DEFAULT_GAME3),
                        help="game directory to tune on (default: Sample_Game_3)")
    parser.add_argument("--events-path",
                        default=os.environ.get("TUNE_EVENTS_PATH", _DEFAULT_EVENTS3),
                        help="events file path for dead-ball mask (default: Game 3 JSON)")
    parser.add_argument("--full-gt", action="store_true",
                        help="use full GT instead of the public 5-min slice")
    parser.add_argument("--model", choices=["hmm", "lstm", "all"], default="all",
                        help="which model to tune: hmm, lstm, or all (default)")
    args = parser.parse_args()

    game_dir    = args.game_dir
    events_path = args.events_path
    game_name   = os.path.basename(game_dir)

    print(f"Loading {game_name}...")
    data  = load_tracking(game_dir)
    feats = extract_features(data)
    gt    = load_gt(game_dir)

    if args.full_gt:
        print(f"Using full GT ({len(gt)} frames)")
        gt_slice   = gt
        feats_slice = feats
    else:
        n_public = min(PUBLIC_FRAMES, len(gt))
        print(f"Using public slice: first {n_public} frames ({n_public/25/60:.1f} min, "
              f"simulates BAR-ATH 5-min window)")
        gt_slice    = gt[:n_public]
        feats_slice = feats[:n_public]

    gt_codes = pm.label_codes(gt_slice)

    mlflow.set_experiment("possession-hmm-tune")
    gt_scope = "full" if args.full_gt else f"public_{len(gt_slice)}fr"

    results = []

    if args.model in ("hmm", "all"):
        for buffer in BUFFER_VALUES:
            dmask = dead_ball_mask(events_path, n_frames=len(gt_slice), buffer=buffer)

            # Train once per buffer value — transition/emission probs don't depend on min_spell
            print(f"\nbuffer={buffer}: training HMM...")
            model = PossessionHMM()
            model.fit(feats_slice, gt_slice, dmask)
            raw_labels = model._predict_raw(feats_slice, dmask)

            for min_spell in MIN_SPELL_VALUES:
                labels = _merge_short_spells(raw_labels.copy(), min_spell)
                pred_codes = pm.label_codes(labels)
                result = pm.evaluate(gt_codes, pred_codes)
                spell_f1 = float(result["spell_f1_mean"])
                macro_f1 = float(result["macro_f1"])

                print(f"  min_spell={min_spell:2d}  buffer={buffer}  "
                      f"spell_f1={spell_f1:.4f}  macro_f1={macro_f1:.4f}")

                with mlflow.start_run(run_name=f"msf{min_spell}_buf{buffer}_{gt_scope}"):
                    mlflow.log_params({
                        "min_spell_frames": min_spell,
                        "dead_ball_buffer": buffer,
                        "n_features":       len(FEATURE_NAMES),
                        "train_game":       game_name,
                        "gt_scope":         gt_scope,
                    })
                    mlflow.log_metrics({"spell_f1": spell_f1, "macro_f1": macro_f1})

                results.append((spell_f1, macro_f1, min_spell, buffer))

    results.sort(reverse=True)
    if results:
        print("\n--- Top 5 HMM by spellF1 ---")
        print(f"{'spell_f1':>8}  {'macro_f1':>8}  {'min_spell':>9}  {'buffer':>6}")
        for spell_f1, macro_f1, msf, buf in results[:5]:
            print(f"{spell_f1:8.4f}  {macro_f1:8.4f}  {msf:9d}  {buf:6d}")

    if args.model in ("lstm", "all"):
        print(f"\nTuning LSTM on {game_name} ({gt_scope})...")
        _dmask = dead_ball_mask(events_path, n_frames=len(gt_slice), buffer=3)
        best = tune_lstm(feats_slice, gt_slice, _dmask,
                         game_name=game_name, gt_scope=gt_scope)
        print(f"\n--- Best LSTM ---")
        print(f"  hidden_size={best['hidden_size']}  max_epochs={best['max_epochs']}  "
              f"chunk_size={best['chunk_size']}  boundary_weight={best['boundary_weight']}  "
              f"use_crf={best['use_crf']}  "
              f"spell_f1={best['spell_f1']:.4f}  macro_f1={best['macro_f1']:.4f}")


def tune_lstm(feats, gt, dmask,
              hidden_sizes=None, max_epochs_values=None, chunk_sizes=None,
              boundary_weight_values=None, use_crf_values=None,
              game_name="unknown", gt_scope="public") -> dict:
    """Grid-search LSTM hyperparameters. Returns best params dict.

    Parameters
    ----------
    feats, gt, dmask        : arrays for the evaluation window
    hidden_sizes            : list of hidden_size values to try
    max_epochs_values       : list of max_epochs values to try
    chunk_sizes             : list of chunk_size values to try
    boundary_weight_values  : list of boundary_weight values to try (1.0 = disabled)
    use_crf_values          : list of bool — whether to add a CRF decode layer
    game_name               : logged to MLflow
    gt_scope                : logged to MLflow ('public' or 'full')

    Returns
    -------
    dict with keys: hidden_size, max_epochs, chunk_size, boundary_weight, use_crf,
                    spell_f1, macro_f1
    """
    from src.possession_lstm import PossessionLSTM
    import itertools

    hidden_sizes           = hidden_sizes           or LSTM_HIDDEN_SIZES
    max_epochs_values      = max_epochs_values      or LSTM_MAX_EPOCHS
    chunk_sizes            = chunk_sizes            or LSTM_CHUNK_SIZES
    boundary_weight_values = boundary_weight_values or LSTM_BOUNDARY_WEIGHTS
    use_crf_values         = use_crf_values         or LSTM_USE_CRF

    gt_codes = pm.label_codes(gt)
    results  = []

    mlflow.set_experiment("possession-lstm-tune")

    grid = list(itertools.product(
        hidden_sizes, max_epochs_values, chunk_sizes,
        boundary_weight_values, use_crf_values,
    ))
    print(f"  {len(grid)} LSTM combinations to evaluate...")

    for hidden_size, max_epochs, chunk_size, boundary_weight, use_crf in grid:
        tag = (f"lstm_h{hidden_size}_e{max_epochs}_c{chunk_size}"
               f"_bw{boundary_weight}_crf{int(use_crf)}_{gt_scope}")
        print(f"  {tag}...")
        model = PossessionLSTM(hidden_size=hidden_size, max_epochs=max_epochs,
                               chunk_size=chunk_size,
                               boundary_weight=boundary_weight,
                               use_crf=use_crf)
        model.fit(feats, gt, dmask)
        labels     = model.predict(feats, dmask)
        pred_codes = pm.label_codes(labels)
        result     = pm.evaluate(gt_codes, pred_codes)
        spell_f1   = float(result["spell_f1_mean"])
        macro_f1   = float(result["macro_f1"])
        print(f"    spell_f1={spell_f1:.4f}  macro_f1={macro_f1:.4f}")

        with mlflow.start_run(run_name=tag):
            mlflow.log_params({
                "model":            "lstm",
                "hidden_size":      hidden_size,
                "max_epochs":       max_epochs,
                "chunk_size":       chunk_size,
                "boundary_weight":  boundary_weight,
                "use_crf":          use_crf,
                "train_game":       game_name,
                "gt_scope":         gt_scope,
            })
            mlflow.log_metrics({"spell_f1": spell_f1, "macro_f1": macro_f1})

        results.append((spell_f1, macro_f1,
                        hidden_size, max_epochs, chunk_size, boundary_weight, use_crf))

    results.sort(reverse=True)
    best = results[0]
    return {
        "spell_f1":        best[0],
        "macro_f1":        best[1],
        "hidden_size":     best[2],
        "max_epochs":      best[3],
        "chunk_size":      best[4],
        "boundary_weight": best[5],
        "use_crf":         best[6],
    }


if __name__ == "__main__":
    main()
