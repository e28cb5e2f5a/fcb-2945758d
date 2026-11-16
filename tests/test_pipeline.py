import os
import csv
import numpy as np
import pytest

N_FRAMES = 2_000  # matches conftest._N (synthetic fixture size)


def test_pipeline_produces_valid_csv(pipeline_csv):
    assert os.path.isfile(pipeline_csv)
    with open(pipeline_csv) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["frame", "label"]
    assert len(rows) == N_FRAMES + 1  # frames + header


def test_pipeline_labels_valid(pipeline_csv):
    with open(pipeline_csv) as f:
        next(f)
        labels = {row.strip().split(",")[1] for row in f}
    assert labels.issubset({"H", "A", "D", "X"})


def test_pipeline_frame_column_sequential(pipeline_csv):
    with open(pipeline_csv) as f:
        reader = csv.reader(f)
        next(reader)
        frames = [int(row[0]) for row in reader]
    assert frames == list(range(len(frames)))


def test_apply_graded_blocks_marks_outside_as_x():
    """Frames outside graded half-blocks must be X; inside must keep their label."""
    from src.pipeline import apply_graded_blocks
    labels = np.array(["H"] * 20, dtype="<U1")
    # two half-blocks: frames 5-9 and 14-17 (0-indexed, inclusive)
    blocks = [(1, 5, 9), (2, 14, 17)]
    result = apply_graded_blocks(labels, blocks)
    assert (result[:5]   == "X").all(),  "before first half must be X"
    assert (result[5:10] == "H").all(),  "inside first half must keep label"
    assert (result[10:14] == "X").all(), "between halves must be X"
    assert (result[14:18] == "H").all(), "inside second half must keep label"
    assert (result[18:]  == "X").all(),  "after second half must be X"
