"""Round 12 overnight pipeline — fully automated.

Steps
-----
1. Run tune_loo.py grid search (G3-only, msf sweep).
2. Read best combo from LOO cache.
3. Cross-check winning config on G1 and G2 folds.
4. Run sweep_challenge.py with best params.
5. Score the resulting CSV against BAR-ATH public GT.
6. Apply weighted decision gate (§9).
7. Write docs/CHALLENGE12.md with full reasoning.

Usage
-----
    python3.11 src/run_round12.py

Decision gate (§9 — weighted LOO)
-----------------------------------
G3 is the only organizer-provided practice game → weight 3.
G1 and G2 are user-found games → weight 1 each.

Weighted delta = (3×ΔG3 + 1×ΔG1 + 1×ΔG2) / 5
If weighted delta > 0 → lock v10
If weighted delta ≤ 0 → revert to v8

Weights fixed before running any experiment — not tuned on outcomes.
BAR-ATH score is recorded for transparency only, never used for decisions.

v8 baselines (14-feat, corrected GT, §C11):
  G1=0.4999  G2=0.6028  G3=0.5325
"""

import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVAL_DIR  = os.path.join(ROOT, "evaluation")
CACHE_DIR = os.path.join(ROOT, "data", "loo_cache")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
C12_PATH  = os.path.join(ROOT, "docs", "CHALLENGE12.md")

# v8 baselines — 14-feat, corrected GT (§C11)
V8_G1 = 0.4999
V8_G2 = 0.6028
V8_G3 = 0.5325

# Weighted decision: G3 (organizer GT) counts 3×, G1/G2 count 1× each
FOLD_WEIGHTS = {"G1": 1, "G2": 1, "G3": 3}
WEIGHT_TOTAL = sum(FOLD_WEIGHTS.values())  # 5

MSF_V10 = 125   # fixed; same as v8 (LOO did not search msf)
RUN_ID  = "v10"


# ---------------------------------------------------------------------------
# Step 1 — Run tune_loo.py
# ---------------------------------------------------------------------------

def run_tune_loo():
    print("=" * 60)
    print("Step 1: Running tune_loo.py (8 combos × 3 folds = 24 runs)")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, os.path.join(HERE, "tune_loo.py")],
        cwd=ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tune_loo.py failed with exit code {result.returncode}")
    print("tune_loo.py complete.")


# ---------------------------------------------------------------------------
# Step 2 — Read best combo from cache
# ---------------------------------------------------------------------------

def read_best_combo() -> dict:
    """Return the best (mean_sf, hidden, epochs, chunk, bw, crf) from cache."""
    if not os.path.isdir(CACHE_DIR):
        raise FileNotFoundError(f"LOO cache not found: {CACHE_DIR}")

    entries = []
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(CACHE_DIR, fname)) as fh:
            e = json.load(fh)
        entries.append(e)

    if not entries:
        raise RuntimeError("LOO cache is empty — tune_loo.py may not have written results.")

    # Sort by G3 spellF1; tiebreak: prefer higher msf (more conservative)
    entries = [e for e in entries if e.get("g3") is not None]
    entries.sort(key=lambda e: (e["g3"], e.get("msf", 0)), reverse=True)
    best = entries[0]
    msf_best = best.get("msf", MSF_V10)
    print(f"\nStep 2: Best LOO combo — G3_spellF1={best['g3']:.4f}")
    print(f"  hidden={best['hidden']}  epochs={best['epochs']}  "
          f"chunk={best['chunk']}  bw={best['bw']}  crf={best['crf']}  msf={msf_best}")
    best["msf_best"] = msf_best
    return best


# ---------------------------------------------------------------------------
# Step 3 — Sweep v10 with best params
# ---------------------------------------------------------------------------

def run_sweep(best: dict):
    print("\n" + "=" * 60)
    print("Step 3: Running sweep_challenge.py with LOO-best params")
    print("=" * 60)
    msf = best.get("msf_best", MSF_V10)
    cmd = [
        sys.executable, os.path.join(HERE, "sweep_challenge.py"),
        "--bw",          str(best["bw"]),
        "--msf",         str(msf),
        "--hidden-size", str(best["hidden"]),
        "--max-epochs",  str(best["epochs"]),
        "--run-id",      RUN_ID,
    ]
    if best["crf"]:
        cmd.append("--use-crf")
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"sweep_challenge.py failed with exit code {result.returncode}")
    print("sweep_challenge.py complete.")


# ---------------------------------------------------------------------------
# Step 4 — Score the CSV
# ---------------------------------------------------------------------------

def score_v10(best: dict) -> float:
    """Score the v10 CSV against BAR-ATH public GT. Returns spellF1."""
    print("\n" + "=" * 60)
    print("Step 4: Scoring v10 against BAR-ATH public GT")
    print("=" * 60)

    sys.path.insert(0, EVAL_DIR)
    import possession_metrics as pm
    from run_evaluation import find_gt, load_meta, score_submission

    gt_path  = find_gt(os.path.join(EVAL_DIR, "gt", "BAR-ATH_possession_gt_public.csv"))
    gt_codes = pm.label_codes(pm.read_label_csv(gt_path))

    # Find the v10 CSV — name is lstm_bwN_msfM_v10.csv
    bw_int   = int(best["bw"])
    msf_used = best.get("msf_best", MSF_V10)
    csv_name = f"lstm_bw{bw_int}_msf{msf_used}_{RUN_ID}.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_name)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Expected sweep output not found: {csv_path}\n"
            "Check sweep_challenge.py output for the actual filename."
        )

    result, reason = score_submission(csv_path, gt_codes, pm.TAUS, pm.BOOTSTRAP_N)
    if result is None:
        raise RuntimeError(f"Scoring failed: {reason}")

    spell_f1 = float(result["spell_f1_mean"])
    macro_f1 = float(result["macro_f1"])
    print(f"v10 spellF1={spell_f1:.4f}  macroF1={macro_f1:.4f}")
    return spell_f1, macro_f1


# ---------------------------------------------------------------------------
# Step 4b — Cross-check winning config on G1 and G2
# ---------------------------------------------------------------------------

def run_crosscheck(best: dict) -> tuple:
    """Train winning config on G1/G2 folds, return (sf_g1, sf_g2)."""
    print("\n" + "=" * 60)
    print("Step 4b: Cross-checking winning config on G1 and G2 folds")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, os.path.join(HERE, "crosscheck_v10.py")],
        cwd=ROOT, capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"crosscheck_v10.py failed:\n{result.stderr}")

    # Parse G1/G2 spellF1 from output
    sf_g1 = sf_g2 = None
    for line in result.stdout.splitlines():
        if line.strip().startswith("G1:"):
            sf_g1 = float(line.split("spellF1=")[1].split()[0])
        elif line.strip().startswith("G2:"):
            sf_g2 = float(line.split("spellF1=")[1].split()[0])
    if sf_g1 is None or sf_g2 is None:
        raise RuntimeError("Could not parse G1/G2 spellF1 from crosscheck output")
    return sf_g1, sf_g2


# ---------------------------------------------------------------------------
# Step 5 + 6 — Decision gate and write CHALLENGE12.md
# ---------------------------------------------------------------------------

def write_challenge12(best: dict, sf_g1: float, sf_g2: float,
                      spell_f1: float, macro_f1: float):
    top5_entries = []
    for fname in sorted(os.listdir(CACHE_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(CACHE_DIR, fname)) as fh:
            top5_entries.append(json.load(fh))
    top5_entries = [e for e in top5_entries if e.get("g3") is not None]
    top5_entries.sort(key=lambda e: (e["g3"], e.get("msf", 0)), reverse=True)
    top5 = top5_entries[:5]

    sf_g3 = best["g3"]

    # Weighted delta: G3×3, G1×1, G2×1
    d_g3 = sf_g3 - V8_G3
    d_g1 = sf_g1 - V8_G1
    d_g2 = sf_g2 - V8_G2
    weighted_delta = (FOLD_WEIGHTS["G3"] * d_g3 +
                      FOLD_WEIGHTS["G1"] * d_g1 +
                      FOLD_WEIGHTS["G2"] * d_g2) / WEIGHT_TOTAL
    gate_pass = weighted_delta > 0
    wins = sum([d_g3 > 0, d_g1 > 0, d_g2 > 0])

    decision = ("v10 LOCKED — weighted LOO delta > 0" if gate_pass
                else "REVERT to v8 — weighted LOO delta ≤ 0")

    date_str = datetime.date.today().isoformat()

    lines = [
        f"# CHALLENGE12 — Round 12 results ({date_str})",
        "",
        "## §C12 — 15-feat tune_loo.py re-optimisation",
        "",
        "### Setup",
        "",
        "15-feat config (ball_visible restored at col 2) swept over the",
        "hyperparameter grid using `tune_loo.py` (G3-only selection fold).",
        "Config selection is based on weighted LOO (§9).",
        "",
        "**Fold weights:** G3 (organizer-provided GT) = 3×, G1/G2",
        "(user-found games) = 1× each. Weights fixed before running",
        "any experiment.",
        "",
        "### G3-only LOO grid top-5",
        "",
        "| G3_sf | h | e | bw | crf | msf |",
        "|---|---|---|---|---|---|",
    ]
    for e in top5:
        lines.append(
            f"| {e['g3']:.4f} | {e['hidden']} | {e['epochs']} | "
            f"{e['bw']} | {e['crf']} | {e.get('msf', '?')} |"
        )

    lines += [
        "",
        "### Winning config (selected by G3 LOO)",
        "",
        f"hidden={best['hidden']}  epochs={best['epochs']}  "
        f"bw={best['bw']}  crf={best['crf']}  msf={best.get('msf_best', '?')}",
        f"G3 spellF1 = **{sf_g3:.4f}**",
        "",
        "### Cross-check on G1 and G2",
        "",
        "| fold | v10 | v8 | delta | weight |",
        "|---|---|---|---|---|",
        f"| G3 (selection) | {sf_g3:.4f} | {V8_G3:.4f} | "
        f"{d_g3:+.4f} | 3× |",
        f"| G1 (cross-chk) | {sf_g1:.4f} | {V8_G1:.4f} | "
        f"{d_g1:+.4f} | 1× |",
        f"| G2 (cross-chk) | {sf_g2:.4f} | {V8_G2:.4f} | "
        f"{d_g2:+.4f} | 1× |",
        "",
        "### Weighted decision gate (§9)",
        "",
        "Weighted delta = (3×ΔG3 + 1×ΔG1 + 1×ΔG2) / 5",
        f"= (3×{d_g3:+.4f} + 1×{d_g1:+.4f} + 1×{d_g2:+.4f}) / 5",
        f"= **{weighted_delta:+.4f}**",
        "",
        f"v10 wins {wins}/3 folds. Weighted delta {'> 0 → accept' if gate_pass else '≤ 0 → reject'}.",
        "",
        f"**{decision}**",
        "",
        "### BAR-ATH result (reporting only — not used for decision)",
        "",
        "| config | spellF1 | macroF1 |",
        "|---|---|---|",
        f"| v8 (14-feat) | 0.4000 | — |",
        f"| v10 (15-feat, msf=100) | {spell_f1:.4f} | {macro_f1:.4f} |",
        "",
    ]

    with open(C12_PATH, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\n§C12 written to {C12_PATH}")
    print(f"\n{'='*60}")
    print(f"DECISION: {decision}")
    print(f"Weighted delta: {weighted_delta:+.4f}  "
          f"(ΔG3={d_g3:+.4f}×3  ΔG1={d_g1:+.4f}×1  ΔG2={d_g2:+.4f}×1)")
    print(f"BAR-ATH v10={spell_f1:.4f} (reporting only)")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    run_tune_loo()
    best = read_best_combo()
    sf_g1, sf_g2 = run_crosscheck(best)
    run_sweep(best)
    spell_f1, macro_f1 = score_v10(best)
    write_challenge12(best, sf_g1, sf_g2, spell_f1, macro_f1)


if __name__ == "__main__":
    main()
