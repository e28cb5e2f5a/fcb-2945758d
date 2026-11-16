# Project rules for Claude

## §9 — Evaluation integrity (HARD RULE)

BAR-ATH public window scores are **reporting-only**. Never use them to
justify a config change, feature decision, hyperparameter choice, or
revert. That is a §9 violation — equivalent to tuning on the test set.

All config selection must go through LOO cross-validation on the three
practice games. BAR-ATH numbers may appear in tables marked
"reporting only — not used for selection." They must never appear as
the stated reason for any decision.
