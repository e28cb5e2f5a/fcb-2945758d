"""Overnight grid search: boundary_weight × min_spell_frames for BAR-ATH.

Trains one LSTM per boundary_weight on all 3 sample games, then applies each
min_spell_frames post-hoc (no retraining per msf value). Optionally also tests
with ball interpolation and smart-tagging features.

Grid
----
    boundary_weight  : 5, 10, 20
    min_spell_frames : 50, 75, 100, 125

That is 3 trains × 4 msf values = 12 combos. With --interp-ball, each
combo is doubled (with/without interpolation) for 24 total rows.

Config selection
----------------
This script does NOT score against BAR-ATH ground truth — using those
labels to pick between configs violates competition RULES.md §2 ("the
5 minutes are not a tuning set").

Select the best config using a separate LOO evaluation:
    python3.11 src/pipeline.py --train 1 2 --predict 3 --model lstm

The sweep summary records H/A/D/X frame counts and spell counts per
config so label-distribution shifts are visible.

Crash resilience: if evaluation/results/lstm_bw{bw}_msf{msf}[_interp].csv
already exists, that combo is skipped — no retraining or re-prediction.

Usage
-----
    python3.11 src/sweep_challenge.py
    python3.11 src/sweep_challenge.py --interp-ball
    python3.11 src/sweep_challenge.py --bw 5 --msf 100 --dry-run
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

EVAL_DIR = os.path.join(ROOT, "evaluation")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)
MODELS_DIR = os.path.join(ROOT, "models")

BW_GRID = [5, 10, 20]
MSF_GRID = [50, 75, 100, 125]


def _write_csv(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "label"])
        for i, label in enumerate(labels):
            writer.writerow([i, label])


def _label_counts(labels):
    return {l: int((labels == l).sum()) for l in ("H", "A", "D", "X")}


def _count_spells(labels):
    spells = 0
    prev = None
    for label in labels:
        if label in ("H", "A") and label != prev:
            spells += 1
        prev = label
    return spells


def _load_sample_game_data(game_dirs: list, tagging_feats: bool = False):
    """Load tracking features (and optionally tagging features) for sample games.

    When tagging_feats=True, appends 4 zero columns to each game's features because
    the sample games do not have a Smart Tagging XML.  This ensures the model input
    dimension matches the challenge data at inference (N + 4 features).

    Returns (feats, gt, dmask) concatenated across all game_dirs.
    """
    from src.pipeline import _load_game
    all_feats, all_gt, all_dmask = [], [], []
    for d in game_dirs:
        feats, gt, dmask, _ = _load_game(d)
        if tagging_feats:
            # Sample games have no tagging XML — pad with zeros so the
            # model receives the same 18-dim input as at challenge inference.
            tag_zeros = np.zeros((feats.shape[0], 4), dtype=np.float32)
            feats = np.hstack([feats, tag_zeros])
        all_feats.append(feats)
        all_gt.append(gt)
        all_dmask.append(dmask)
    return (np.concatenate(all_feats),
            np.concatenate(all_gt),
            np.concatenate(all_dmask))


def _load_challenge_data(interp_ball: bool, tagging_feats: bool = False):
    from src.challenge_loader import load_challenge_tracking
    from src.features import extract_features
    from src.dead_ball import dead_ball_mask
    from src.challenge_loader import _CHALLENGE_DATA_DIR
    from games import asset_paths, half_video_blocks, GAME

    data = load_challenge_tracking()

    if interp_ball:
        from src.ball_interp import interpolate_ball
        data = interpolate_ball(data)

    feats = extract_features(data)

    if tagging_feats:
        from src.tagging_features import tagging_features as _tag_feats
        paths = asset_paths(game=GAME, source="tactical", data_dir=_CHALLENGE_DATA_DIR)
        tag = _tag_feats(paths["smart_tagging"], n_frames=feats.shape[0])
        feats = np.hstack([feats, tag])

    if np.isnan(feats).any():
        feats = np.where(np.isnan(feats), 0.0, feats)

    paths = asset_paths(game=GAME, source="tactical", data_dir=_CHALLENGE_DATA_DIR)
    dmask = dead_ball_mask(paths["smart_tagging"], n_frames=feats.shape[0],
                           ball_xy=data["ball_xy"])

    blocks = half_video_blocks(GAME)
    ch_frames = data["frames"]
    return feats, dmask, blocks, ch_frames, data["ball_xy"]


def _apply_x_mask(labels, blocks):
    from src.pipeline import apply_graded_blocks
    return apply_graded_blocks(labels, blocks)


def main():
    import re
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bw", type=float, nargs="+", default=None,
                        help="boundary_weight values (default: full grid)")
    parser.add_argument("--msf", type=int, nargs="+", default=None,
                        help="min_spell_frames values (default: full grid)")
    parser.add_argument("--interp-ball", action="store_true",
                        help="also score with ball interpolation (doubles combos)")
    parser.add_argument("--tagging-feats", action="store_true",
                        help="also score with smart-tagging features (requires retraining per combo)")
    parser.add_argument("--dry-run", action="store_true",
                        help="run one combo only (first bw × first msf) and exit")
    parser.add_argument("--run-id", default="",
                        help="run identifier appended to all output filenames (e.g. v2)")
    parser.add_argument("--use-crf", action="store_true",
                        help="enable CRF output layer (separate retrain)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for torch and numpy (reproducibility)")
    parser.add_argument("--use-overlap", action="store_true",
                        help="enable overlapping inference windows (§11b benchmark)")
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="LSTM hidden size (default: 256, must match LOO best)")
    parser.add_argument("--max-epochs", type=int, default=40,
                        help="LSTM max training epochs (default: 40, must match LOO best)")
    parser.add_argument("--model-path", default=None,
                        help="explicit .pt path to load/save the model, overriding the "
                             "auto-generated name; only valid with a single --bw value")
    parser.add_argument("--output-csv", default=None,
                        help="explicit output CSV path, overriding the auto-generated name; "
                             "only valid with a single --bw and single --msf value")
    # NOTE: --include-gt-tuning is intentionally absent. Appending BAR-ATH GT
    # labels to training data violates competition RULES.md §2.
    # NOTE: --use-attention is intentionally absent. Practice 2 ablations
    # showed attention costs -0.039 spellF1 on every tested split.
    args = parser.parse_args()

    if args.run_id and re.search(r'[/\\.]', args.run_id):
        parser.error("--run-id must not contain '/', '\\', or '.'")
    if args.model_path and len(args.bw or BW_GRID) > 1:
        parser.error("--model-path requires a single --bw value")
    if args.output_csv and (len(args.bw or BW_GRID) > 1 or len(args.msf or MSF_GRID) > 1):
        parser.error("--output-csv requires a single --bw and single --msf value")

    # Preflight: corrected GT must exist before any expensive work starts.
    from src.sample_loader import require_corrected_gt
    from src.pipeline import GAME_DIRS as _GAME_DIRS
    require_corrected_gt([_GAME_DIRS[g] for g in [1, 2, 3]])

    # Derived tags
    run_tag  = f"_{args.run_id}" if args.run_id else ""
    crf_tag  = "_crf" if args.use_crf else ""
    ov_tag   = "_ov" if args.use_overlap else ""

    # Override module-level SUMMARY_PATH for this run
    summary_name = f"sweep_summary{run_tag}.txt"
    summary_path = os.path.join(RESULTS_DIR, summary_name)

    # Reproducibility
    import random
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bw_grid = args.bw or BW_GRID
    msf_grid = args.msf or MSF_GRID
    if args.dry_run:
        bw_grid = [bw_grid[0]]
        msf_grid = [msf_grid[0]]

    interp_variants = [False, True] if args.interp_ball else [False]
    tagging_variants = [False, True] if args.tagging_feats else [False]

    n_combos = len(bw_grid) * len(msf_grid) * len(interp_variants) * len(tagging_variants)
    print(f"Sweep: bw={bw_grid} × msf={msf_grid} × interp={interp_variants} × tagging={tagging_variants}")
    print(f"       crf={args.use_crf}  seed={args.seed}  run={args.run_id!r}")
    print(f"Total combos: {n_combos}")
    print(f"NOTE: no BAR-ATH GT scoring — select config via LOO on Game 3.")
    print()

    from src.pipeline import train_model, GAME_DIRS, clamp_d_to_mask
    from src.possession_hmm import _merge_short_spells

    results = []
    trained_models = {}  # (bw, tagging, crf, overlap, seed) → model, shared across msf/interp

    for tagging in tagging_variants:
        tagging_tag = "_tag" if tagging else ""
        for interp in interp_variants:
            interp_tag = "_interp" if interp else ""
            print(f"--- Loading challenge data (interp={interp}, tagging={tagging}) ---")
            feats, dmask, blocks, ch_frames, ball_xy = _load_challenge_data(interp, tagging_feats=tagging)

            for bw in bw_grid:
                model_key = (bw, tagging, args.use_crf, args.use_overlap, args.seed)
                model_path = args.model_path or os.path.join(
                    MODELS_DIR,
                    f"lstm_bw{int(bw)}{tagging_tag}{crf_tag}{ov_tag}{run_tag}.pt")

                # Train once per unique key — interp and msf don't affect training
                if model_key not in trained_models and os.path.isfile(model_path):
                    print(f"\nLoading saved model {model_path} ...")
                    from src.possession_lstm import PossessionLSTM
                    trained_models[model_key] = PossessionLSTM.load(model_path)
                if model_key not in trained_models:
                    print(f"\nTraining LSTM boundary_weight={bw} tagging={tagging} "
                          f"crf={args.use_crf} overlap={args.use_overlap} seed={args.seed}...")
                    train_dirs = [GAME_DIRS[g] for g in [1, 2, 3]]
                    if tagging:
                        # Load sample-game data with zero-padded tagging cols so
                        # the model input dimension matches challenge inference (N+4).
                        tr_feats, tr_gt, tr_dmask = _load_sample_game_data(
                            train_dirs, tagging_feats=True)
                        model = train_model(
                            [],  # empty — pre-loaded data passed via append_*
                            model_type="lstm",
                            boundary_weight=bw, save_path=model_path,
                            use_crf=args.use_crf,
                            use_overlap=args.use_overlap,
                            seed=args.seed,
                            hidden_size=args.hidden_size,
                            max_epochs=args.max_epochs,
                            append_feats=tr_feats,
                            append_gt=tr_gt,
                            append_dmask=tr_dmask,
                        )
                    else:
                        model = train_model(train_dirs, model_type="lstm",
                                            boundary_weight=bw, save_path=model_path,
                                            use_crf=args.use_crf,
                                            use_overlap=args.use_overlap,
                                            seed=args.seed,
                                            hidden_size=args.hidden_size,
                                            max_epochs=args.max_epochs)
                    trained_models[model_key] = model
                else:
                    model = trained_models[model_key]

                # Raw prediction (before msf)
                print(f"  Predicting bw={bw}...")
                raw_labels = model.predict(feats, dmask)
                before_d = int((raw_labels == "D").sum())
                raw_labels = clamp_d_to_mask(raw_labels, dmask, ball_xy=ball_xy)
                after_d = int((raw_labels == "D").sum())
                print(f"  Clamp: D {before_d:,} → {after_d:,} "
                      f"({before_d - after_d:,} converted to H/A)")

                for msf in msf_grid:
                    csv_name = (f"lstm_bw{int(bw)}_msf{msf}"
                                f"{interp_tag}{tagging_tag}{crf_tag}{ov_tag}{run_tag}")
                    csv_path = args.output_csv or os.path.join(
                        RESULTS_DIR, f"{csv_name}.csv")

                    if os.path.isfile(csv_path):
                        print(f"  SKIP {csv_name} — CSV exists")
                        import possession_metrics as pm_mod
                        table = pm_mod.read_label_csv(csv_path)
                        labels = np.array([pm_mod.LABELS[c] for c in pm_mod.label_codes(table)],
                                          dtype="<U1")
                    else:
                        labels = _merge_short_spells(raw_labels.copy(), msf)
                        labels = _apply_x_mask(labels, blocks)
                        _write_csv(csv_path, labels)

                    counts = _label_counts(labels)
                    n_spells = _count_spells(labels)
                    print(f"  bw={bw:4.0f} msf={msf:3d} interp={str(interp):5} "
                          f"tagging={str(tagging):5} "
                          f"H={counts['H']:,} A={counts['A']:,} "
                          f"D={counts['D']:,} X={counts['X']:,} spells={n_spells}")
                    results.append({
                        "name": csv_name, "bw": bw, "msf": msf,
                        "interp": interp, "tagging": tagging,
                        "n_H": counts["H"], "n_A": counts["A"],
                        "n_D": counts["D"], "n_X": counts["X"],
                        "n_spells": n_spells,
                    })

    # Write summary — label counts only, no BAR-ATH spellF1
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(f"{'name':<45} {'bw':>4} {'msf':>5} {'interp':>6}  "
                f"{'n_H':>7} {'n_A':>7} {'n_D':>7} {'n_X':>7} {'spells':>6}\n")
        f.write("-" * 100 + "\n")
        for r in results:
            f.write(f"{r['name']:<45} {r['bw']:>4.0f} {r['msf']:>5d} {str(r['interp']):>6}  "
                    f"{r['n_H']:>7,} {r['n_A']:>7,} {r['n_D']:>7,} {r['n_X']:>7,} "
                    f"{r['n_spells']:>6}\n")
        f.write("\n")
        f.write("Config selection: run LOO evaluation on Game 3 to compare configs.\n")
        f.write("  python3.11 src/pipeline.py --train 1 2 --predict 3 --model lstm\n")

    print(f"\nSummary written to {summary_path}")
    print(f"Select config via Game 3 LOO — do not use BAR-ATH public GT for selection.")


if __name__ == "__main__":
    main()
