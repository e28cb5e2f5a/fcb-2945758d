"""End-to-end possession pipeline: load → features → dead ball → HMM → CSV.

Train on one or more sample games, predict on another.

Usage
-----
    python src/pipeline.py --train 3 --predict 3 --out evaluation/results/hmm_game3.csv
"""

import argparse
import csv
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.sample_loader import load_tracking, load_gt, require_corrected_gt
from src.features import extract_features, FEATURE_NAMES
from src.dead_ball import dead_ball_mask
from src.ball_interp import interpolate_ball
from src.possession_hmm import PossessionHMM, MIN_SPELL_FRAMES

try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False

SAMPLE_DIR = os.path.join(ROOT, "data", "sample")

GAME_DIRS = {
    1: os.path.join(SAMPLE_DIR, "Sample_Game_1"),
    2: os.path.join(SAMPLE_DIR, "Sample_Game_2"),
    3: os.path.join(SAMPLE_DIR, "Sample_Game_3"),
}


def _events_path(game_dir: str) -> str:
    """Return the events file path for a game directory (JSON or CSV)."""
    game_name = os.path.basename(game_dir)
    json_path = os.path.join(game_dir, f"{game_name}_events.json")
    if os.path.isfile(json_path):
        return json_path
    return os.path.join(game_dir, f"{game_name}_RawEventsData.csv")


def _load_game(game_dir: str, interp_max_gap: int = 0):
    """Load tracking, features, gt and dead-ball mask for one game directory.

    GT is optional: returns None when the GT file is absent (e.g. BAR-ATH before release).
    interp_max_gap: passed to interpolate_ball(); 0 disables interpolation.
    """
    data = load_tracking(game_dir)
    if interp_max_gap > 0:
        data = interpolate_ball(data, max_gap=interp_max_gap)
    feats = extract_features(data)
    try:
        gt = load_gt(game_dir)
    except FileNotFoundError:
        gt = None
    n_frames = feats.shape[0]
    if gt is not None and len(gt) != n_frames:
        raise ValueError(
            f"GT length {len(gt)} != tracking frames {n_frames} in {game_dir}; "
            "they must be aligned frame-for-frame."
        )
    dmask = dead_ball_mask(_events_path(game_dir), n_frames=n_frames,
                           ball_xy=data["ball_xy"])
    return feats, gt, dmask, data


def train_model(train_dirs: list, model_type: str = "hmm", save_path: str = None,
                boundary_weight: float = 5.0, boundary_radius: int = 8,
                use_crf: bool = False, hidden_size: int = 256, max_epochs: int = 40,
                lr: float = None, min_spell_frames: int = None,
                append_feats=None, append_gt=None, append_dmask=None,
                seed: int = 42, use_overlap: bool = False,
                interp_max_gap: int = 0):
    """Train model on train_dirs and optionally save to disk. Returns the fitted model."""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Loading {len(train_dirs)} training game(s)...")
    all_feats, all_gt, all_dmask = [], [], []
    for d in train_dirs:
        feats, gt, dmask, _ = _load_game(d, interp_max_gap=interp_max_gap)
        all_feats.append(feats)
        all_gt.append(gt)
        all_dmask.append(dmask)

    if all_feats:
        train_feats = np.concatenate(all_feats)
        train_gt    = np.concatenate(all_gt)
        train_dmask = np.concatenate(all_dmask)
    else:
        # No game dirs — must have append_feats; initialise empty arrays
        # that will be replaced immediately below.
        train_feats = np.empty((0,), dtype=np.float32)
        train_gt    = np.empty((0,), dtype="<U1")
        train_dmask = np.empty((0,), dtype=bool)

    if append_feats is not None:
        if train_feats.ndim == 1 and train_feats.size == 0:
            # No game-dir data loaded — use append arrays directly
            train_feats  = append_feats
            train_gt     = append_gt
            train_dmask  = append_dmask
        else:
            train_feats  = np.concatenate([train_feats,  append_feats],  axis=0)
            train_gt     = np.concatenate([train_gt,     append_gt],     axis=0)
            train_dmask  = np.concatenate([train_dmask,  append_dmask],  axis=0)
        print(f"  Appended {len(append_feats)} extra training frames")

    if model_type == "lstm":
        from src.possession_lstm import PossessionLSTM
        import src.possession_hmm as _hmm
        if min_spell_frames is not None:
            _hmm.MIN_SPELL_FRAMES = min_spell_frames
        kwargs = dict(hidden_size=hidden_size, max_epochs=max_epochs,
                      boundary_weight=boundary_weight, boundary_radius=boundary_radius,
                      use_crf=use_crf, use_overlap=use_overlap, seed=seed)
        if lr is not None:
            kwargs["lr"] = lr
        model = PossessionLSTM(**kwargs)
        print("Training LSTM...")
    else:
        model = PossessionHMM()
        print("Training HMM...")
    model.fit(train_feats, train_gt, train_dmask)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"Model saved: {save_path}")

    return model


def apply_graded_blocks(labels: np.ndarray, blocks: list) -> np.ndarray:
    """Set frames outside graded half-blocks to 'X'.

    Parameters
    ----------
    labels : (N,) label array, modified in-place copy
    blocks : list of (half, first_frame, last_frame) — 0-indexed, inclusive

    Returns
    -------
    np.ndarray (N,) with frames outside all blocks set to 'X'
    """
    out = np.full(len(labels), "X", dtype="<U1")
    for _, first, last in blocks:
        out[first:last + 1] = labels[first:last + 1]
    return out


def clamp_d_to_mask(labels, dmask, ball_xy=None):
    """Threshold-free: ball visible → game live → D wrong → carry-forward H/A.

    last_ha starts as None; the clamp is suppressed until the first genuine
    H or A frame is observed, so no arbitrary team is assigned before
    possession is established.
    """
    if ball_xy is not None:
        ball_visible = ~np.isnan(ball_xy[:, 0])
    else:
        ball_visible = np.ones(len(labels), dtype=bool)
    live = ~dmask & ball_visible
    out = labels.copy()
    last_ha = None
    for i in range(len(out)):
        if dmask[i]:
            if out[i] in ("H", "A"):
                last_ha = out[i]
        elif live[i] and out[i] == "D" and last_ha is not None:
            out[i] = last_ha
        elif out[i] in ("H", "A"):
            last_ha = out[i]
    return out


def predict_with_model(model, predict_dir: str, output_csv: str,
                       train_ids: list = None, model_type: str = "hmm",
                       clamp: bool = False, interp_max_gap: int = 0) -> str:
    """Run predict_dir through a fitted model and write submission CSV."""
    predict_id = os.path.basename(predict_dir)
    print(f"Predicting on {predict_id}...")
    pred_feats, pred_gt, pred_dmask, pred_data = _load_game(predict_dir,
                                                             interp_max_gap=interp_max_gap)
    labels = model.predict(pred_feats, pred_dmask)
    if clamp:
        before_d = int((labels == "D").sum())
        labels = clamp_d_to_mask(labels, pred_dmask, ball_xy=pred_data["ball_xy"])
        after_d = int((labels == "D").sum())
        print(f"  Clamp: D {before_d:,} → {after_d:,} ({before_d - after_d:,} converted to H/A)")

    # Apply graded half-blocks if available (BAR-ATH: marks frames outside halves as X)
    try:
        from games import half_video_blocks
        blocks = half_video_blocks(predict_id)
        labels = apply_graded_blocks(labels, blocks)
        print(f"  Applied graded blocks: {[(h, f, l) for h, f, l in blocks]}")
    except Exception:
        pass  # not available for sample games — skip silently

    _write_csv(output_csv, labels)

    counts = {l: int((labels == l).sum()) for l in ("H", "A", "D", "X")}
    n_spells = _count_spells(labels)
    print(f"Written: {output_csv}")
    print(f"  H={counts['H']:,}  A={counts['A']:,}  D={counts['D']:,}  X={counts['X']:,}")
    print(f"  Possession spells: {n_spells}")

    spell_f1, macro_f1, scored = _score(labels, pred_gt)

    if _MLFLOW:
        mlflow.set_experiment(f"possession-{model_type}")
        with mlflow.start_run(run_name=f"train{train_ids}_pred{predict_id}"):
            mlflow.log_params({
                "train_games":      train_ids,
                "predict_game":     predict_id,
                "n_features":       len(FEATURE_NAMES),
                "min_spell_frames": MIN_SPELL_FRAMES,
                "model_type":       model_type,
                "scored":           scored,
            })
            if scored:
                mlflow.log_metrics({"spell_f1": spell_f1, "macro_f1": macro_f1})

    return output_csv


def run_pipeline(train_dirs: list, predict_dir: str, output_csv: str,
                 model_type: str = "hmm") -> str:
    """Train model on train_dirs, predict on predict_dir, write submission CSV.

    Parameters
    ----------
    model_type : "hmm" (default) or "lstm"

    Returns path to written CSV.
    """
    train_ids = [os.path.basename(d) for d in train_dirs]
    model = train_model(train_dirs, model_type=model_type)
    return predict_with_model(model, predict_dir, output_csv,
                              train_ids=train_ids, model_type=model_type)


def _score(labels: np.ndarray, gt: np.ndarray):
    """Compute spell_f1 and macro_f1 against GT. Returns (spell_f1, macro_f1, scored).

    scored=False when GT is absent (e.g. BAR-ATH before Oct 20) or metrics unavailable.
    Callers must check scored before using the metric values.
    """
    if gt is None:
        return 0.0, 0.0, False
    import sys as _sys
    eval_dir = os.path.join(ROOT, "evaluation")
    if eval_dir not in _sys.path:
        _sys.path.insert(0, eval_dir)
    import possession_metrics as pm  # ImportError propagates — broken scorer must not go undetected
    try:
        gt_codes   = pm.label_codes(gt)
        pred_codes = pm.label_codes(labels)
        result = pm.evaluate(gt_codes, pred_codes)
        spell_f1 = float(result["spell_f1_mean"])
        macro_f1 = float(result["macro_f1"])
        print(f"  spell_f1={spell_f1:.4f}  macro_f1={macro_f1:.4f}")
        return spell_f1, macro_f1, True
    except Exception as e:
        print(f"  [warn] scoring skipped: {e}")
        return 0.0, 0.0, False


def _write_csv(path: str, labels: np.ndarray) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "label"])
        for i, label in enumerate(labels):
            writer.writerow([i, label])


def _count_spells(labels: np.ndarray) -> int:
    spells = 0
    prev = None
    for label in labels:
        if label in ("H", "A") and label != prev:
            spells += 1
        prev = label
    return spells


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", nargs="+", type=int, choices=[1, 2, 3],
                        default=[3], help="game numbers to train on (default: 3)")
    parser.add_argument("--predict", type=int, choices=[1, 2, 3],
                        default=None, help="game number to predict (default: 3)")
    parser.add_argument("--predict-dir",
                        default=None,
                        help="raw directory path to predict on (overrides --predict; use for BAR-ATH on Oct 20)")
    parser.add_argument("--model", choices=["hmm", "lstm"], default="hmm",
                        help="model to use: hmm (default) or lstm")
    parser.add_argument("--save-model", default=None, metavar="PATH",
                        help="train and save model to PATH, then exit (no prediction)")
    parser.add_argument("--load-model", default=None, metavar="PATH",
                        help="load saved model from PATH and skip training")
    # LSTM-specific training options (ignored for --model hmm)
    parser.add_argument("--bw", type=float, default=None, metavar="W",
                        help="alias for --boundary-weight (overrides if both supplied)")
    parser.add_argument("--boundary-weight", type=float, default=5.0, metavar="W",
                        help="loss upweight for frames near spell boundaries (default: 5.0; 1.0=disabled)")
    parser.add_argument("--boundary-radius", type=int, default=8, metavar="R",
                        help="frames either side of a transition to upweight (default: 8)")
    parser.add_argument("--use-crf", action="store_true", default=False,
                        help="add a CRF decode layer on top of the LSTM")
    parser.add_argument("--hidden-size", type=int, default=256, metavar="H",
                        help="LSTM hidden size (default: 128)")
    parser.add_argument("--epochs", type=int, default=40, metavar="E",
                        help="max training epochs (default: 40)")
    parser.add_argument("--lr", type=float, default=None, metavar="LR",
                        help="Adam learning rate (default: 1e-3 hardcoded in model)")
    parser.add_argument("--msf", type=int, default=None, metavar="N",
                        help="alias for --min-spell-frames (overrides if both supplied)")
    parser.add_argument("--min-spell-frames", type=int, default=None, metavar="N",
                        help="minimum spell length in frames; shorter spells merged (default: 10)")
    parser.add_argument("--seed", type=int, default=42, metavar="S",
                        help="random seed for reproducibility (default: 42)")
    parser.add_argument("--clamp", action="store_true", default=False,
                        help="apply threshold-free visible-ball D→H/A clamp at inference")
    parser.add_argument("--interp-max-gap", type=int, default=0, metavar="G",
                        help="max consecutive missing-ball frames to interpolate (default: 0;"
                             " 0 disables; locked config is 0)")
    parser.add_argument("--out",
                        default=os.path.join(ROOT, "evaluation", "results", "hmm_game3.csv"),
                        help="output CSV path")
    args = parser.parse_args()

    # Preflight: corrected GT must exist before any expensive work starts.
    require_corrected_gt([GAME_DIRS[g] for g in args.train])

    # --bw and --msf are aliases; they override the long-form if supplied
    effective_bw  = args.bw if args.bw is not None else args.boundary_weight
    effective_msf = args.msf if args.msf is not None else args.min_spell_frames

    if args.load_model:
        if effective_msf is not None:
            import src.possession_hmm as _hmm
            _hmm.MIN_SPELL_FRAMES = effective_msf
        if args.model == "lstm":
            from src.possession_lstm import PossessionLSTM
            model = PossessionLSTM.load(args.load_model)
        else:
            model = PossessionHMM.load(args.load_model)
        print(f"Model loaded: {args.load_model}")
    else:
        train_dirs = [GAME_DIRS[g] for g in args.train]
        model = train_model(train_dirs, model_type=args.model,
                            save_path=args.save_model,
                            boundary_weight=effective_bw,
                            boundary_radius=args.boundary_radius,
                            use_crf=args.use_crf,
                            hidden_size=args.hidden_size,
                            max_epochs=args.epochs,
                            lr=args.lr,
                            min_spell_frames=effective_msf,
                            seed=args.seed,
                            interp_max_gap=args.interp_max_gap)
        if args.save_model and args.predict is None and args.predict_dir is None:
            raise SystemExit(0)

    if args.predict_dir is not None:
        predict_dir = args.predict_dir
    else:
        predict_dir = GAME_DIRS[args.predict if args.predict is not None else 3]

    train_ids = args.train if not args.load_model else None
    predict_with_model(model, predict_dir, args.out,
                       train_ids=train_ids, model_type=args.model,
                       clamp=args.clamp, interp_max_gap=args.interp_max_gap)
