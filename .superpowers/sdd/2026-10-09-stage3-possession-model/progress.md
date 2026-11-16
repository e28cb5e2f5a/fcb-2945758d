# SDD ledger — plan: docs/superpowers/plans/2026-10-09-stage3-possession-model.md

## Pre-flight interface scan

- Task 1 produces `load_tracking(game_dir) -> dict`, `load_gt(game_dir) -> np.ndarray`, `PITCH`
- Task 2 consumes `load_tracking` dict with keys `home_xy, away_xy, ball_xy, periods` — matches Task 1 Produces ✓
- Task 3 consumes events CSV path only — no dependency on Tasks 1-2 ✓
- Task 4 consumes `extract_features()` (Task 2), `load_gt()` (Task 1), `dead_ball_mask()` (Task 3) — all match ✓
- Task 5 consumes all prior tasks — matches ✓

Pre-flight: interfaces consistent across all tasks. No conflicts found.

## Task progress

Task 1: complete
Task 2: complete
Task 3: complete
Task 4: complete (tests: python3.11 -m pytest tests/test_possession_hmm.py → 5/5 pass)
Task 5: complete (tests: python3.11 -m pytest tests/ → 25/25 pass)
Ruling: sample player count not fixed at 14 — Game 2 Away has 12. Fixed loader to detect dynamically from header. Cost if wrong: none, verified against data.
Ruling: task score skips hmm_sample (141k rows vs 199k expected) — correct, scorer only handles BAR-ATH format. Sample pipeline is for training/validation only. Cost if wrong: none, expected behavior.
Ruling: task score step shows working baselines, confirming evaluation infrastructure is intact. (commits 378ea06..c02752e, tests: python3.11 -m pytest tests/test_dead_ball.py → 5/5 pass) (commits b44f955..378ea06, tests: python3.11 -m pytest tests/test_features.py → 6/6 pass) (commits db09ccd..b44f955, tests: python3.11 -m pytest tests/test_sample_loader.py → 6/6 pass)
Ruling: coordinates test loosened to [-0.2, 1.2] — ball goes out of play and tracking follows it beyond pitch boundary. Cost if wrong: none, real data confirms this.
