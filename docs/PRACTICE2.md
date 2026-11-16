# Practice 2 — Conclusion (2026-10-12)

Ran: **9:02 → 16:35** (~7.5 hrs). Phase 3 did not complete — agent stopped after Phase 2.

---

## Conclusion — What We Learned

Actual runtime: **9:02 → 16:35** (~7.5 hrs, vs 2-3 hr estimate).
The CPU cost of h=256/e=40 folds was ~23 min each, not ~10 min.
Phase 3 did not complete — the autonomous agent stopped after Phase 2.

---

### Phase 1 — LOO Grid Search Results

Full top-5 (mean spellF1 across 3 LOO folds):

| hidden | epochs | bw | G1 | G2 | G3 | mean |
|--------|--------|----|----|----|----|------|
| 256 | 40 | 2.5 | 0.7149 | 0.7619 | 0.5000 | **0.6589** |
| 256 | 40 | 5.0 | 0.7000 | 0.7628 | 0.4889 | 0.6506 |
| 128 | 40 | 2.5 | 0.7628 | 0.7220 | 0.4333 | 0.6394 |
| 128 | 40 | 5.0 | 0.6449 | 0.7600 | 0.4444 | 0.6164 |
| 256 | 20 | 5.0 | 0.5680 | 0.7070 | 0.3707 | 0.5486 |

**Hyperparam findings:**
- `epochs=40` is essential — all top-4 use it; dropping to 20 costs ~0.13 mean spellF1
- `hidden=256` beats `hidden=128` on the Game 3 LOO fold (+0.07)
- `bw=2.5` marginally beats `bw=5.0` — high boundary weight over-penalises and hurts generalisation
- CRF was dropped from the grid (3× training cost, known to underperform on this data)

**Game 3 distribution shift:** G3 is consistently 0.22-0.28 below G1/G2 in the same combo.
Games 1+2 use `RawEventsData.csv`; Game 3 uses a JSON events file — different source pipeline,
likely different teams/competition. The model struggles to generalise to Game 3 when it hasn't
seen it in training. This directly mirrors the BAR-ATH challenge: training on all 3 known games
before predicting BAR-ATH is critical.

---

### Phase 2 — Architecture Ablations (h=256, e=40, bw=2.5, train 1+2 → eval G3)

| Variant | spellF1 | vs baseline | vs original 0.649 |
|---------|---------|-------------|-------------------|
| baseline (plain BiLSTM) | 0.6054 | — | -0.044 |
| **B: cosine LR only** | **0.6811** | **+0.076** | **+0.032** |
| A: overlap inference only | 0.5730 | -0.032 | -0.076 |
| C: self-attention only | 0.5667 | -0.039 | -0.082 |
| Combined A+B+C | 0.5000 | -0.105 | -0.149 |

**Architecture findings:**
- **Cosine LR is the only beneficial change** — +7.6% over fixed-LR baseline, beats published 0.649
- **Overlap inference hurts** — BiLSTM already has bidirectional context; averaging over overlapping
  windows smears sharp spell boundaries, directly hurting spellF1
- **Self-attention hurts** — adds ~130k parameters that can't be learned reliably from 2-game training
  sets (~30k frames); the attention layer fits noise rather than signal
- **Combinations are destructive** — attention overfits noise, cosine LR fine-tunes into that noise,
  overlap then smears the result; Combined A+B+C = worst outcome (0.5000)

**Critical mistake in the run:** the autonomous agent baked all three arch changes (A+B+C) into
`possession_lstm.py` before Phase 1, so all 24 LOO runs were penalised by overlap and attention.
Phase 1 LOO scores are valid for **relative hyperparam comparison** but their absolute values are
suppressed vs what plain BiLSTM + cosine LR would have achieved.

---

### What Needs To Happen Next (Phase 3)

1. **Revert `possession_lstm.py`** — strip out overlap inference and self-attention; keep only cosine LR
2. **Train on all 3 games** with `h=256, e=40, bw=2.5, cosine_lr=True`
3. **Save to `models/lstm_v2.pt`**
4. **Expected leaderboard score**: B_cosine_lr scored 0.6811 on an unseen game (train 1+2 → G3).
   Training on all 3 games before BAR-ATH should close the distribution gap further.

---

### Process Lessons

- **Estimate actual CPU cost before committing to a grid** — h=256/e=40 folds took 23 min each,
  not 10 min. Always time one fold first with `time python3.11 ...`
- **Cache per-combo results from the start** — two restarts lost early combos before caching was added
- **Ablate before permanently modifying the model** — baking A+B+C into `possession_lstm.py` before
  LOO confounded the hyperparam search with the arch changes
- **Phase 2 baseline ≠ Phase 1 winner** — combo 7's G3 fold (0.5000) used the broken all-3-combined
  model; the true baseline (plain BiLSTM) on the same split scored 0.6054, revealing the damage
