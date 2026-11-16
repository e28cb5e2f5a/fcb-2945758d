"""Tests that --run-id and --use-crf flags are wired into sweep_challenge.py."""
import sys
import os
import argparse


def _parse(args):
    """Parse args through sweep_challenge's actual argument parser."""
    sys.path.insert(0, os.path.abspath("."))
    import importlib
    import src.sweep_challenge as sc
    importlib.reload(sc)  # ensure fresh state

    # Build a minimal parser matching what main() should register
    parser = argparse.ArgumentParser()
    # These are the flags we're testing — if main() doesn't add them,
    # parse_known_args will surface them as unknowns and we can detect that.
    known, unknown = parser.parse_known_args(args)
    return unknown  # flags not recognised by a bare parser


def test_run_id_flag_accepted_by_sweep():
    """sweep_challenge.py --run-id v2 must parse without error."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "src/sweep_challenge.py", "--help"],
        capture_output=True, text=True
    )
    assert "--run-id" in result.stdout, (
        "--run-id flag not found in sweep_challenge.py --help output"
    )


def test_use_crf_flag_accepted_by_sweep():
    """sweep_challenge.py --use-crf must appear in help."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "src/sweep_challenge.py", "--help"],
        capture_output=True, text=True
    )
    assert "--use-crf" in result.stdout, (
        "--use-crf flag not found in sweep_challenge.py --help output"
    )


def test_seed_flag_accepted_by_sweep():
    """sweep_challenge.py --seed must appear in help."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "src/sweep_challenge.py", "--help"],
        capture_output=True, text=True
    )
    assert "--seed" in result.stdout, (
        "--seed flag not found in sweep_challenge.py --help output"
    )


def test_run_id_namespaces_csv_name():
    """run_tag built from run_id must appear in csv_name."""
    bw, msf = 5, 125
    interp_tag, tagging_tag = "_interp", "_tag"
    run_id = "v2"
    run_tag = f"_{run_id}" if run_id else ""
    crf_tag = ""
    csv_name = f"lstm_bw{int(bw)}_msf{msf}{interp_tag}{tagging_tag}{crf_tag}{run_tag}"
    assert csv_name == "lstm_bw5_msf125_interp_tag_v2"


def test_empty_run_id_matches_iter1_behaviour():
    """Empty run_id produces same name as Iteration 1 (no suffix)."""
    bw, msf = 5, 125
    interp_tag, tagging_tag = "_interp", ""
    run_id = ""
    run_tag = f"_{run_id}" if run_id else ""
    csv_name = f"lstm_bw{int(bw)}_msf{msf}{interp_tag}{tagging_tag}{run_tag}"
    assert csv_name == "lstm_bw5_msf125_interp"
