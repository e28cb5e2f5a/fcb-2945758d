"""Leave-one-out cross-validation grid search for PossessionLSTM.

G3-only LOO (train G1+G2, eval G3):
  G3 is the only organizer-provided practice game.
  G1 and G2 are user-found games, used for training only.

Grid: LSTM hyperparams × msf (min_spell_frames post-processing).
msf does not require retraining — one model per LSTM combo is trained,
then evaluated at every msf value.

Evaluation is always on the public 5-min slice (first 7,500 frames).
Logs to MLflow experiment "possession-lstm-loo".
Prints top-5 summary at the end.

Usage
-----
    python3.11 src/tune_loo.py
"""

import datetime
import hashlib
import itertools
import json
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
from src.ball_interp import interpolate_ball
from src.possession_lstm import PossessionLSTM

# ---------------------------------------------------------------------------
# IMPORTANT — G3 result alone is NOT a valid config decision.
# After tune_loo.py completes, the winning config MUST be validated via a
# G1/G2 cross-check script (crosscheck_vNN.py) using the weighted gate:
#   weighted_delta = (3×ΔG3 + 1×ΔG1 + 1×ΔG2) / 5
# Accepting a config based on G3 spellF1 alone is a §9 violation.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Grid: LSTM params locked to Round 12 best (h=128, e=20, bw=5.0).
# Only msf is swept — no retraining needed per msf value.
# 1 LSTM combo × 5 msf values = 1 training run, 5 evaluations.
# Dropped: hidden=512 (too slow), chunk=1024 (redundant), crf=True (3× overhead)
# ---------------------------------------------------------------------------
LSTM_HIDDEN_SIZES       = [128]
LSTM_MAX_EPOCHS         = [20]
LSTM_CHUNK_SIZES        = [2048]
LSTM_BOUNDARY_WEIGHTS   = [5.0]
LSTM_USE_CRF            = [False]

# msf swept via LOO — no retraining, evaluated post-hoc on each fold
LSTM_MSF_VALUES         = [50, 100, 125, 175, 250]

# Ball interpolation: locked to 0 (no interpolation) — matches the v8 submission
# config selected by LOO (no-interp 0.4450 > interp 0.4356 on G3 LOO, 2026-11-10).
# Set to 10 only if re-running the intermediate interp-on probe for reference.
_INTERP_MAX_GAP = 0

PUBLIC_FRAMES = 7_500

_DEFAULT_DATA_ROOT = os.path.join(ROOT, "data", "sample")


def _build_games(data_root: str) -> list:
    """Return GAMES list for data_root.

    Expects subdirectories Sample_Game_1, Sample_Game_2, Sample_Game_3.
    Events file is auto-detected: .csv preferred, falls back to .json.
    Override via --data-root or PRACTICE_DATA_ROOT env var.
    """
    games = []
    for i, name in enumerate(["Sample_Game_1", "Sample_Game_2", "Sample_Game_3"], 1):
        d = os.path.join(data_root, name)
        csv_path  = os.path.join(d, f"{name}_RawEventsData.csv")
        json_path = os.path.join(d, f"{name}_events.json")
        events = csv_path if os.path.exists(csv_path) else json_path
        games.append({"name": f"Game{i}", "dir": d, "events": events})
    return games


# ---------------------------------------------------------------------------
# Live log (append-only so progress is visible without waiting for completion)
# ---------------------------------------------------------------------------
LOG_PATH   = os.path.join(ROOT, "docs", "overnight_log.md")
CACHE_DIR  = os.path.join(ROOT, "data", "loo_cache")

# ---------------------------------------------------------------------------
# Cache version — invalidates stale results when source files change.
# Hash the two files that directly affect LOO scores.
# ---------------------------------------------------------------------------
def _source_hash() -> str:
    h = hashlib.md5()
    for rel in ("src/features.py", "src/possession_lstm.py", "src/ball_interp.py"):
        path = os.path.join(ROOT, rel)
        try:
            with open(path, "rb") as fh:
                h.update(fh.read())
        except FileNotFoundError:
            pass
    return h.hexdigest()[:8]

CACHE_VERSION = _source_hash()


def _log(msg: str, *, echo: bool = True):
    """Append msg + newline to overnight_log.md and optionally print it."""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_PATH, "a") as fh:
        fh.write(line + "\n")
    if echo:
        print(line, flush=True)


def _cache_load() -> dict:
    """Return {tag: tuple} for cached combos whose source hash matches CACHE_VERSION.

    Entries written by a different version of features.py / possession_lstm.py
    are silently skipped — they will be re-run and overwritten.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = {}
    stale = 0
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(CACHE_DIR, fname)
        try:
            with open(path) as fh:
                entry = json.load(fh)
            if entry.get("cache_version") != CACHE_VERSION:
                stale += 1
                continue
            tag = entry["tag"]
            cache[tag] = (
                entry["mean_sf"], entry["mean_mf"],
                entry["g1"], entry["g2"], entry["g3"],
                entry["hidden"], entry["epochs"],
                entry["chunk"], entry["bw"], entry["crf"],
                entry.get("msf"),  # None for older entries without msf
            )
        except Exception:
            pass  # corrupt file — will be overwritten if that combo reruns
    if stale:
        print(f"[cache] skipped {stale} stale entry/entries (source changed, hash={CACHE_VERSION})")
    return cache


def _cache_save_one(tag: str, mean_sf: float, mean_mf: float,
                    fold_spell: list, fold_macro: list,
                    hidden: int, epochs: int, chunk: int,
                    bw: float, crf: bool, msf: int = None):
    """Write a single combo result to its own JSON file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = tag.replace("/", "_")
    path = os.path.join(CACHE_DIR, f"{safe}.json")
    entry = {
        "tag": tag,
        "cache_version": CACHE_VERSION,
        "mean_sf": mean_sf, "mean_mf": mean_mf,
        "g1": fold_spell[0], "g2": fold_spell[1], "g3": fold_spell[2],
        "hidden": hidden, "epochs": epochs, "chunk": chunk,
        "bw": bw, "crf": crf,
    }
    if msf is not None:
        entry["msf"] = msf
    with open(path, "w") as fh:
        json.dump(entry, fh, indent=2)


# ---------------------------------------------------------------------------
# Data loading (cached per-game — all 24 runs share the same arrays)
# ---------------------------------------------------------------------------

def _load_game(game: dict):
    """Return (feats, gt, dmask) for a game dict."""
    print(f"  Loading {game['name']}...")
    data  = load_tracking(game["dir"])
    if _INTERP_MAX_GAP > 0:
        data = interpolate_ball(data, max_gap=_INTERP_MAX_GAP)
    feats = extract_features(data)
    gt    = load_gt(game["dir"])
    n     = feats.shape[0]
    dmask = dead_ball_mask(game["events"], n_frames=n)
    return feats, gt, dmask


def _load_all():
    return [_load_game(g) for g in GAMES]


# ---------------------------------------------------------------------------
# LOO evaluation for one param combo
# ---------------------------------------------------------------------------

def _evaluate_fold(train_data: list, val_data: tuple,
                   hidden_size: int, max_epochs: int,
                   chunk_size: int, boundary_weight: float,
                   use_crf: bool) -> tuple:
    """Train on concatenated train_data; return (raw_labels, gt_codes, dmask_pub).

    msf is applied externally so multiple msf values can be evaluated
    from a single trained model without retraining.
    """
    from src.possession_hmm import _merge_short_spells

    all_feats = np.concatenate([d[0] for d in train_data])
    all_gt    = np.concatenate([d[1] for d in train_data])
    all_dmask = np.concatenate([d[2] for d in train_data])

    model = PossessionLSTM(
        hidden_size=hidden_size,
        max_epochs=max_epochs,
        chunk_size=chunk_size,
        boundary_weight=boundary_weight,
        use_crf=use_crf,
    )
    model.fit(all_feats, all_gt, all_dmask)

    val_feats, val_gt, val_dmask = val_data
    n_pub = min(PUBLIC_FRAMES, len(val_gt))
    feats_pub = val_feats[:n_pub]
    gt_pub    = val_gt[:n_pub]
    dmask_pub = val_dmask[:n_pub]

    raw_labels = model.predict(feats_pub, dmask_pub)
    gt_codes   = pm.label_codes(gt_pub)
    return raw_labels, gt_codes


def _score_with_msf(raw_labels: np.ndarray, gt_codes, msf: int) -> tuple:
    """Apply msf filter and return (spell_f1, macro_f1)."""
    from src.possession_hmm import _merge_short_spells
    labels     = _merge_short_spells(raw_labels.copy(), msf)
    pred_codes = pm.label_codes(labels)
    result     = pm.evaluate(gt_codes, pred_codes)
    return float(result["spell_f1_mean"]), float(result["macro_f1"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None,
                        help="root dir containing Sample_Game_1/2/3 "
                             "(default: $PRACTICE_DATA_ROOT or data/sample)")
    args = parser.parse_args()
    data_root = (args.data_root
                 or os.environ.get("PRACTICE_DATA_ROOT")
                 or _DEFAULT_DATA_ROOT)
    global GAMES
    GAMES = _build_games(data_root)

    try:
        import mlflow
        mlflow.set_experiment("possession-lstm-loo")
        _mlflow = True
    except Exception:
        _mlflow = False
        print("[warn] MLflow unavailable — continuing without logging")

    cache = _cache_load()
    n_cached = len(cache)
    _log("=== Phase 1: LOO grid search started ===")
    _log(f"data_root: {data_root}")
    if n_cached:
        _log(f"Resuming — {n_cached} combo(s) already in cache, skipping them.")
    print("Loading all three games...")
    game_data = _load_all()   # list of (feats, gt, dmask) indexed 0/1/2
    _log("All three games loaded.")

    # G3-only validation: train on G1+G2, eval on G3.
    # G3 is the only organizer-provided practice game.
    VAL_IDX   = 2
    TRAIN_IDXS = [0, 1]

    lstm_grid = list(itertools.product(
        LSTM_HIDDEN_SIZES,
        LSTM_MAX_EPOCHS,
        LSTM_CHUNK_SIZES,
        LSTM_BOUNDARY_WEIGHTS,
        LSTM_USE_CRF,
    ))
    n_total = len(lstm_grid)  # 1 training run per LSTM combo; msf is post-hoc
    _log(f"{len(lstm_grid)} LSTM combo(s) × {len(LSTM_MSF_VALUES)} msf values "
         f"= {len(lstm_grid)} training run(s), "
         f"{len(lstm_grid) * len(LSTM_MSF_VALUES)} evaluations")

    # all_results tuples: (g3_sf, g3_mf, hidden, epochs, chunk, bw, crf, msf)
    all_results = []

    # Restore cached results
    for tag, v in cache.items():
        vals = v[:11]
        if len(vals) == 11:
            _, _, _, _, g3, h, e, c, bw, crf, msf = vals
        else:
            # older cache entries without msf field
            _, _, _, _, g3, h, e, c, bw, crf = vals[:10]
            msf = None
        if g3 is not None:
            all_results.append((g3, v[1], h, e, c, bw, crf, msf))

    train_data = [game_data[i] for i in TRAIN_IDXS]
    val_data   = game_data[VAL_IDX]

    for combo_idx, (hidden, epochs, chunk, bw, crf) in enumerate(lstm_grid, 1):
        lstm_tag = (f"lstm_h{hidden}_e{epochs}_c{chunk}"
                    f"_bw{bw}_crf{int(crf)}")

        # Check if all msf values for this LSTM combo are cached
        all_cached = all(
            f"{lstm_tag}_msf{msf}_loo" in cache for msf in LSTM_MSF_VALUES
        )
        if all_cached:
            _log(f"--- Combo [{combo_idx}/{len(lstm_grid)}] {lstm_tag}  [ALL MSF CACHED]")
            continue

        _log(f"--- Combo [{combo_idx}/{len(lstm_grid)}] {lstm_tag}  training G3 fold...")
        raw_labels, gt_codes = _evaluate_fold(
            train_data, val_data,
            hidden_size=hidden, max_epochs=epochs,
            chunk_size=chunk, boundary_weight=bw,
            use_crf=crf,
        )
        _log(f"  training done — evaluating {len(LSTM_MSF_VALUES)} msf values")

        for msf in LSTM_MSF_VALUES:
            tag = f"{lstm_tag}_msf{msf}_loo"
            if tag in cache:
                _log(f"  msf={msf}  [CACHED]")
                continue

            sf, mf = _score_with_msf(raw_labels, gt_codes, msf)
            _log(f"  msf={msf:4d}  G3_spellF1={sf:.4f}  G3_macroF1={mf:.4f}")

            cache[tag] = (sf, mf, None, None, sf, hidden, epochs, chunk, bw, crf, msf)
            _cache_save_one(tag, sf, mf,
                            [None, None, sf], [None, None, mf],
                            hidden, epochs, chunk, bw, crf, msf=msf)

            if _mlflow:
                try:
                    with mlflow.start_run(run_name=tag):
                        mlflow.log_params({
                            "model":           "lstm",
                            "hidden_size":     hidden,
                            "max_epochs":      epochs,
                            "chunk_size":      chunk,
                            "boundary_weight": bw,
                            "use_crf":         crf,
                            "msf":             msf,
                            "cv":              "g3_only",
                        })
                        mlflow.log_metrics({
                            "spell_f1_g3": sf,
                            "macro_f1_g3": mf,
                        })
                except Exception as exc:
                    print(f"  [warn] mlflow log failed: {exc}")

            all_results.append((sf, mf, hidden, epochs, chunk, bw, crf, msf))

    all_results.sort(reverse=True)

    header = (f"{'G3_sf':>8}  {'h':>4}  {'e':>3}  {'c':>5}  "
              f"{'bw':>4}  {'crf':>5}  {'msf':>5}")
    sep = "-" * 52
    rows = []
    for r in all_results[:5]:
        g3sf, g3mf, h, e, c, bw, crf, msf = r
        rows.append(f"{g3sf:8.4f}  {h:4d}  {e:3d}  {c:5d}  "
                    f"{bw:4.1f}  {str(crf):>5}  {msf:5d}")

    _log("=" * 52)
    _log("## Phase 1 complete — G3-only msf grid search")
    _log(f"TOP-{min(5,len(all_results))} by G3 spellF1 (train G1+G2, eval G3):")
    _log(header)
    _log(sep)
    for row in rows:
        _log(row)
    _log("=" * 52)

    best = all_results[0]
    g3sf, g3mf, h, e, c, bw, crf, msf = best
    _log(f"Best config: hidden={h} epochs={e} chunk={c} bw={bw} crf={crf} msf={msf}")
    _log("WARNING: G3-only result is NOT sufficient for a config decision.")
    _log("  Mandatory next step: run G1/G2 cross-check (crosscheck_vNN.py) and")
    _log("  apply the weighted gate (G3×3, G1×1, G2×1) before accepting any config.")
    _log("  Skipping the cross-check is a §9 violation.")

    return all_results


if __name__ == "__main__":
    main()
