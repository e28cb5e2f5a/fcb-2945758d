# CHALLENGE13 — Round 13 results (2026-11-12)

## §C13 — Tracking-derived Smart Tagging proxies (18-feat)

### Motivation

BAR-ATH GT has ~60% missing-ball frames vs 32–41% in practice games.
The 14-feat model (v8) cannot learn possession from tracking shape
alone and defaults toward D when the ball is absent. Smart Tagging
XML (BAR-ATH only) encodes tactical context (transitions, set pieces,
zone) that could break this bias. However, the XML is unavailable for
practice games, blocking LOO validation.

Round 13 instead derives four proxy features from tracking data
only — LOO-valid because they require no XML:

| col | feature | derivation |
|-----|---------|------------|
| 14 | `ball_in_final_third` | 1.0 if `ball_x < 0.33` or `> 0.67` |
| 15 | `ball_vx` | signed x-velocity m/s (inherits period/reappear zeroing) |
| 16 | `set_piece_proxy` | 1.0 if ball slow (< 1.5 m/s, visible) ≥ 20/25 frames |
| 17 | `h_centroid_adv_x` | home centroid x − away centroid x (metres) |

Base feature vector: **18-dim** (up from 14-dim in v8).

### Setup

**Fold weights:** G3 (organizer-provided GT) = 3×, G1/G2
(user-found games) = 1× each. Weights fixed before running
any experiment.

LSTM grid locked to previous best (h=128, e=20, bw=5.0, crf=False).
Only msf swept via LOO — one training run, 5 evaluations.

### G3-only LOO grid top-5

| G3_sf | h | e | bw | crf | msf |
|---|---|---|---|---|---|
| **0.6000** | 128 | 20 | 5.0 | False | 50 |
| 0.4857 | 128 | 20 | 5.0 | False | 100 |
| 0.4444 | 128 | 20 | 5.0 | False | 125 |
| 0.4444 | 128 | 20 | 5.0 | False | 175 |
| 0.4444 | 128 | 20 | 5.0 | False | 250 |

The msf cliff reversed versus v8: msf=50 now wins (vs msf=125 in v8),
suggesting the new features produce cleaner spell boundaries and need
less post-filtering.

### Winning config (selected by G3 LOO)

hidden=128  epochs=20  bw=5.0  crf=False  msf=50
G3 spellF1 = **0.6000**  (+0.0675 vs v8)

### Cross-check on G1 and G2

| fold | v13 | v8 | delta | weight |
|---|---|---|---|---|
| G3 (selection) | 0.6000 | 0.5325 | +0.0675 | 3× |
| G1 (cross-chk) | 0.6043 | 0.4999 | +0.1044 | 1× |
| G2 (cross-chk) | 0.6824 | 0.6028 | +0.0796 | 1× |

### Weighted decision gate (§9)

Weighted delta = (3×ΔG3 + 1×ΔG1 + 1×ΔG2) / 5
= (3×+0.0675 + 1×+0.1044 + 1×+0.0796) / 5
= **+0.0773**

v13 wins 3/3 folds. Weighted delta > 0 → accept.

**ACCEPT v13 — 18-feat tracking proxies adopted as new submission.**

The new features generalise strongly across all three folds. G1
(+0.1044) and G2 (+0.0796) gains exceed the G3 selection gain
(+0.0675), confirming the signal is not fold-specific noise.

### Feature ablation — one-out probes (§10)

Each of the four proxy columns dropped individually; model retrained
from scratch on G1+G2, evaluated on G3 at msf=50.

| col | feature | G3_sf | Δ vs v13 |
|-----|---------|-------|----------|
| 15 | `ball_vx` | 0.5379 | −0.0621 |
| 14 | `ball_in_final_third` | 0.5467 | −0.0533 |
| 16 | `set_piece_proxy` | 0.5867 | −0.0133 |
| 17 | `h_centroid_adv_x` | 0.6000 | 0.0000 |
| — | v13 baseline (all 4) | 0.6000 | — |

`ball_vx` is the dominant driver (−0.062 when absent): signed
x-velocity captures transition direction that the existing speed
scalar (col 6) cannot encode. `ball_in_final_third` is the second
driver (−0.053). `set_piece_proxy` is marginal (−0.013).
`h_centroid_adv_x` contributes nothing individually (0.000 delta) —
likely redundant with the existing centroid distance features (cols
11–12); candidate for removal in a future round if model complexity
becomes a concern.

### Permutation importance — D-proxy check for cols 14–17

5-rep permutation importance on the full-game v13 model
(`diagonalmar.pt`, trained on all 3 sample games, evaluated on G3
public 7,500-frame slice, msf=50). Baseline: spellF1=0.6667,
D-rate=0.455. Note: baseline spellF1 is higher than the LOO score
(0.6000) because the model saw G3 during training; the D-rate
analysis is unaffected by this.

All 18 features, sorted by spellF1 drop:

| col | feature | sf_drop | ΔD-rate | note |
|-----|---------|---------|---------|------|
| 13 | `dist_ratio` | +0.4595 | −0.006 | |
| 2 | `dist_nearest_H` | +0.4454 | **+0.187** | top D driver (sentinel=999) |
| 3 | `dist_nearest_A` | +0.2983 | −0.015 | |
| 11 | `centroid_dist_H` | +0.1733 | +0.011 | |
| 12 | `centroid_dist_A` | +0.1248 | +0.009 | |
| 8 | `ball_accel` | +0.1151 | −0.007 | |
| 6 | `ball_speed` | +0.0989 | +0.006 | |
| **17** | **`h_centroid_adv_x`** | **+0.0925** | **−0.029** | **NEW R13 — suppresses D** |
| **15** | **`ball_vx`** | **+0.0736** | **+0.0001** | **NEW R13 — neutral on D** |
| 10 | `spread_A` | +0.0408 | −0.004 | |
| 9 | `spread_H` | +0.0240 | −0.000 | |
| 4 | `delta_dist_H` | +0.0122 | +0.002 | |
| **14** | **`ball_in_final_third`** | **0.0000** | **0.0000** | **NEW R13 — zero perm effect** |
| **16** | **`set_piece_proxy`** | **0.0000** | **0.0000** | **NEW R13 — zero perm effect** |
| 0 | `ball_x` | 0.0000 | 0.0000 | |
| 1 | `ball_y` | 0.0000 | 0.0000 | |
| 7 | `n_H_within_3m` | 0.0000 | 0.0000 | |
| 5 | `delta_dist_A` | −0.0430 | −0.009 | |

**None of the four new features are D-proxies.** `h_centroid_adv_x`
(col 17) actively suppresses D predictions (ΔD-rate = −0.029 when
shuffled), consistent with team advancement providing genuine
possession signal. `ball_vx` (col 15) has negligible D-rate effect
(+0.0001). `ball_in_final_third` and `set_piece_proxy` show zero
permutation effect — likely redundant with existing features at this
model scale.

**Note on col 17 (h_centroid_adv_x) — intentional retention (§10):**
The one-out ablation shows Δ=0.000 on G3; the permutation importance
shows +0.0925 sf_drop. These are not contradictory: ablation measures
whether the model *can* compensate by relearning without the feature
(it can); permutation importance measures whether the *trained* model
*actually uses* it (it does, and to suppress D). A Δ=0 ablation means
other features are sufficient substitutes — not that the feature adds
noise. Dropping col 17 would require a positive G3 gate signal first
(none exists) before a G1/G2 cross-check is warranted. Col 17 is
retained in the locked 18-feat submission.

The dominant D driver remains `dist_nearest_H` (+18.7pp), unchanged
from the C8 analysis on v8 — confirming the missing-ball sentinel
pattern persists and is not amplified by the new features.

### §11a probe — y-spread (20-feat)

Probe: add `spread_H_y` and `spread_A_y` (std of player y-positions)
as cols 18–19. G3 LOO at msf=50, same LSTM config.

| config | G3_sf | Δ vs v13 |
|--------|-------|----------|
| v13 (18-feat, x-spread only) | 0.6000 | — |
| 20-feat (+ y-spread) | 0.5161 | −0.0839 |

**Rejected.** Adding y-spread hurts significantly (−0.084). x-spread
alone is the right representation; y-axis team shape does not carry
additional possession signal at this feature scale.

### §10 probe — 19-feat (18-feat + ball_visible)

§C11 showed 15-feat > 14-feat on all three corrected-GT LOO folds
(weighted delta +0.046). Round 13 built the 4 proxy features on top
of the 14-feat base, never testing whether restoring `ball_visible`
on top of the 18-feat config would improve further. G3 LOO probe
at msf=50 with same LSTM config (h=128, e=20, bw=5.0).

| config | G3_sf | macroF1 | D-rate | Δ vs v13 |
|--------|-------|---------|--------|----------|
| v13 (18-feat, no ball_visible) | 0.6000 | — | — | — |
| 19-feat (18-feat + ball_visible) | 0.5517 | 0.8821 | 0.418 | −0.0483 |

**G3 gate fails — keep v13.**

Restoring `ball_visible` on top of the 4 proxy features hurts
(−0.048). The D-rate (0.418) confirms the `ball_visible=0 → D`
shortcut (§C3) persists: the model still exploits the binary
missing-ball indicator as a D signal even when 4 proxy features are
present. The proxy features are not strong enough to suppress this
shortcut. v13 (18-feat, ball_visible absent) remains locked.

### BAR-ATH result (reporting only — not used for decision)

| config | spellF1 | macroF1 |
|---|---|---|
| v8 (14-feat, msf=125) | 0.4000 | — |
| v13 (18-feat, msf=50) | 0.4000 | 0.6310 |

v13 matches v8 on BAR-ATH (no regression, no gain). The LOO gains
(+0.0773 weighted) do not translate to BAR-ATH. Root cause is
distribution shift: BAR-ATH has ~60% missing-ball frames vs 32–41%
in practice games. The tracking-proxy features (derived from ball
position and speed) are mostly undefined or uninformative during
missing-ball periods, limiting their impact under BAR-ATH conditions.
v13 is adopted as submission (no regression; LOO-validated).
