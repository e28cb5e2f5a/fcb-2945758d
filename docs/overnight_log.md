## Phase 6 complete
Date: 2026-10-12 / 2026-10-13
Baseline (train 1+2 -> eval G3): spellF1=0.6811

Experiment results (train 1+2 -> G3, baseline=0.6811):
  E1 e=60:          spellF1=0.6005  (WORSE -- overfitting)
  E1 e=80:          spellF1=0.6226  (WORSE -- more overfitting)
  E2 bw=1.0:        spellF1=0.5879  (WORSE -- bw=2.5 stays)
  E3 +2 features:   spellF1=0.6055  (WORSE -- reverted to 14 features)
  E4 lr=5e-4:       spellF1=0.5921  (WORSE)
  E4 lr=3e-4:       spellF1=0.5746  (WORSE)
  E5 h=512:         spellF1=0.6248  (WORSE)
  E6 msf=5:         spellF1=0.6889
  E6 msf=10:        spellF1=0.6903  <- best msf, default already optimal
  E6 msf=15:        spellF1=0.6863
  E6 msf=25:        spellF1=0.6842

Winner: baseline config -- h=256, e=40, bw=2.5, lr=1e-3(default), msf=10
  No cross-game experiment beat 0.6811.
  E6 msf=10 on all-3-games model: spellF1=0.6903 (+0.0092 vs cross-game baseline)

All-3-games final retrain: spellF1=0.6903  macroF1=0.9805
Model saved: models/lstm_v2.pt
  config: h=256, e=40, bw=2.5, use_attention=False, use_overlap=False, n_features=14
[20:37:42] === Phase 1: LOO grid search started ===
[20:37:42] data_root: /Users/andrew/Documents/www/fcbarcelona-hack-2027/data/sample
[20:37:55] === Phase 1: LOO grid search started ===
[20:37:55] data_root: /Users/andrew/Documents/www/fcbarcelona-hack-2027/data/sample
[20:38:18] All three games loaded.
[20:38:18] 8 combos × 3 folds = 24 training runs
[20:38:18] --- Combo [1/8] lstm_h128_e20_c2048_bw2.5_crf0_loo
[20:38:18]   fold val=Game1 training...
[20:38:30] All three games loaded.
[20:38:30] 8 combos × 3 folds = 24 training runs
[20:38:30] --- Combo [1/8] lstm_h128_e20_c2048_bw2.5_crf0_loo
[20:38:30]   fold val=Game1 training...
[20:39:44] === Phase 1: LOO grid search started ===
[20:39:44] data_root: /Users/andrew/Documents/www/fcbarcelona-hack-2027/data/sample
[20:40:21] All three games loaded.
[20:40:21] 8 combos × 3 folds = 24 training runs
[20:40:21] --- Combo [1/8] lstm_h128_e20_c2048_bw2.5_crf0_loo
[20:40:21]   fold val=Game1 training...
[20:55:31] === Phase 1: LOO grid search started ===
[20:55:31] data_root: /Users/andrew/Documents/www/fcbarcelona-hack-2027/data/sample
[20:56:07] All three games loaded.
[20:56:07] 8 combos × 3 folds = 24 training runs
[20:56:07] --- Combo [1/8] lstm_h128_e20_c2048_bw2.5_crf0_loo
[20:56:07]   fold val=Game1 training...
[20:59:51]   fold val=Game1 done  spellF1=0.5565  macroF1=0.6464
[20:59:51]   fold val=Game2 training...
[21:03:23]   fold val=Game2 done  spellF1=0.6667  macroF1=0.9410
[21:03:23]   fold val=Game3 training...
[21:06:59]   fold val=Game3 done  spellF1=0.6471  macroF1=0.8888
[21:06:59]   COMBO DONE  mean_spellF1=0.6234  mean_macroF1=0.8254  (G1=0.5565 G2=0.6667 G3=0.6471)
[21:07:00] --- Combo [2/8] lstm_h128_e20_c2048_bw5.0_crf0_loo
[21:07:00]   fold val=Game1 training...
[21:10:36]   fold val=Game1 done  spellF1=0.6000  macroF1=0.6662
[21:10:36]   fold val=Game2 training...
[21:14:13]   fold val=Game2 done  spellF1=0.6556  macroF1=0.9434
[21:14:13]   fold val=Game3 training...
[21:17:50]   fold val=Game3 done  spellF1=0.6625  macroF1=0.8918
[21:17:50]   COMBO DONE  mean_spellF1=0.6394  mean_macroF1=0.8338  (G1=0.6000 G2=0.6556 G3=0.6625)
[21:17:50] --- Combo [3/8] lstm_h128_e40_c2048_bw2.5_crf0_loo
[21:17:50]   fold val=Game1 training...
[21:25:05]   fold val=Game1 done  spellF1=0.6870  macroF1=0.8731
[21:25:05]   fold val=Game2 training...
[21:32:31]   fold val=Game2 done  spellF1=0.6703  macroF1=0.8663
[21:32:31]   fold val=Game3 training...
[21:39:43]   fold val=Game3 done  spellF1=0.5486  macroF1=0.8889
[21:39:43]   COMBO DONE  mean_spellF1=0.6353  mean_macroF1=0.8761  (G1=0.6870 G2=0.6703 G3=0.5486)
[21:39:43] --- Combo [4/8] lstm_h128_e40_c2048_bw5.0_crf0_loo
[21:39:43]   fold val=Game1 training...
[21:46:56]   fold val=Game1 done  spellF1=0.5308  macroF1=0.8230
[21:46:56]   fold val=Game2 training...
[21:54:15]   fold val=Game2 done  spellF1=0.6667  macroF1=0.9422
[21:54:15]   fold val=Game3 training...
[22:01:30]   fold val=Game3 done  spellF1=0.6222  macroF1=0.8894
[22:01:30]   COMBO DONE  mean_spellF1=0.6066  mean_macroF1=0.8849  (G1=0.5308 G2=0.6667 G3=0.6222)
[22:01:30] --- Combo [5/8] lstm_h256_e20_c2048_bw2.5_crf0_loo
[22:01:30]   fold val=Game1 training...
[22:07:31]   fold val=Game1 done  spellF1=0.5333  macroF1=0.6721
[22:07:31]   fold val=Game2 training...
[22:13:30]   fold val=Game2 done  spellF1=0.6706  macroF1=0.9346
[22:13:30]   fold val=Game3 training...
[22:19:31]   fold val=Game3 done  spellF1=0.6471  macroF1=0.8910
[22:19:31]   COMBO DONE  mean_spellF1=0.6170  mean_macroF1=0.8326  (G1=0.5333 G2=0.6706 G3=0.6471)
[22:19:31] --- Combo [6/8] lstm_h256_e20_c2048_bw5.0_crf0_loo
[22:19:31]   fold val=Game1 training...
[22:25:28]   fold val=Game1 done  spellF1=0.6275  macroF1=0.8428
[22:25:28]   fold val=Game2 training...
[22:31:30]   fold val=Game2 done  spellF1=0.6105  macroF1=0.9447
[22:31:30]   fold val=Game3 training...
[22:37:24]   fold val=Game3 done  spellF1=0.6171  macroF1=0.8847
[22:37:24]   COMBO DONE  mean_spellF1=0.6184  mean_macroF1=0.8907  (G1=0.6275 G2=0.6105 G3=0.6171)
[22:37:24] --- Combo [7/8] lstm_h256_e40_c2048_bw2.5_crf0_loo
[22:37:24]   fold val=Game1 training...
[22:50:25]   fold val=Game1 done  spellF1=0.6044  macroF1=0.8599
[22:50:25]   fold val=Game2 training...
[23:04:10]   fold val=Game2 done  spellF1=0.6300  macroF1=0.8561
[23:04:10]   fold val=Game3 training...
[23:16:49]   fold val=Game3 done  spellF1=0.6118  macroF1=0.8821
[23:16:49]   COMBO DONE  mean_spellF1=0.6154  mean_macroF1=0.8660  (G1=0.6044 G2=0.6300 G3=0.6118)
[23:16:49] --- Combo [8/8] lstm_h256_e40_c2048_bw5.0_crf0_loo
[23:16:49]   fold val=Game1 training...
[23:28:42]   fold val=Game1 done  spellF1=0.6041  macroF1=0.8692
[23:28:42]   fold val=Game2 training...
[23:40:12]   fold val=Game2 done  spellF1=0.6333  macroF1=0.8787
[23:40:12]   fold val=Game3 training...
[23:51:31]   fold val=Game3 done  spellF1=0.5600  macroF1=0.8743
[23:51:31]   COMBO DONE  mean_spellF1=0.5991  mean_macroF1=0.8741  (G1=0.6041 G2=0.6333 G3=0.5600)
[23:51:31] ========================================================================
[23:51:31] ## Phase 1 complete — LOO grid search
[23:51:31] TOP-5 by mean spellF1 (LOO, 3 folds):
[23:51:31]  mean_sf    G1_sf    G2_sf    G3_sf     h    e      c    bw    crf
[23:51:31] ------------------------------------------------------------------------
[23:51:31]   0.6394   0.6000   0.6556   0.6625   128   20   2048   5.0  False
[23:51:31]   0.6353   0.6870   0.6703   0.5486   128   40   2048   2.5  False
[23:51:31]   0.6234   0.5565   0.6667   0.6471   128   20   2048   2.5  False
[23:51:31]   0.6184   0.6275   0.6105   0.6171   256   20   2048   5.0  False
[23:51:31]   0.6170   0.5333   0.6706   0.6471   256   20   2048   2.5  False
[23:51:31] ========================================================================
[23:51:31] Phase 2 base config: hidden=128 epochs=20 chunk=2048 bw=5.0 crf=False
[00:17:03] === Phase 1: LOO grid search started ===
[00:17:03] data_root: /Users/andrew/Documents/www/fcbarcelona-hack-2027/data/sample
[00:17:03] Resuming — 8 combo(s) already in cache, skipping them.
[00:17:41] All three games loaded.
[00:17:41] 8 combos × 3 folds = 24 training runs
[00:17:41] --- Combo [1/8] lstm_h128_e20_c2048_bw2.5_crf0_loo  [CACHED mean_spellF1=0.6234]
[00:17:41] --- Combo [2/8] lstm_h128_e20_c2048_bw5.0_crf0_loo  [CACHED mean_spellF1=0.6394]
[00:17:41] --- Combo [3/8] lstm_h128_e40_c2048_bw2.5_crf0_loo  [CACHED mean_spellF1=0.6353]
[00:17:41] --- Combo [4/8] lstm_h128_e40_c2048_bw5.0_crf0_loo  [CACHED mean_spellF1=0.6066]
[00:17:41] --- Combo [5/8] lstm_h256_e20_c2048_bw2.5_crf0_loo  [CACHED mean_spellF1=0.6170]
[00:17:41] --- Combo [6/8] lstm_h256_e20_c2048_bw5.0_crf0_loo  [CACHED mean_spellF1=0.6184]
[00:17:41] --- Combo [7/8] lstm_h256_e40_c2048_bw2.5_crf0_loo  [CACHED mean_spellF1=0.6154]
[00:17:41] --- Combo [8/8] lstm_h256_e40_c2048_bw5.0_crf0_loo  [CACHED mean_spellF1=0.5991]
[00:17:41] ========================================================================
[00:17:41] ## Phase 1 complete — LOO grid search
[00:17:41] TOP-5 by mean spellF1 (LOO, 3 folds):
[00:17:41]  mean_sf    G1_sf    G2_sf    G3_sf     h    e      c    bw    crf
[00:17:41] ------------------------------------------------------------------------
[00:17:41]   0.6394   0.6000   0.6556   0.6625   128   20   2048   5.0  False
[00:17:41]   0.6353   0.6870   0.6703   0.5486   128   40   2048   2.5  False
[00:17:41]   0.6234   0.5565   0.6667   0.6471   128   20   2048   2.5  False
[00:17:41]   0.6184   0.6275   0.6105   0.6171   256   20   2048   5.0  False
[00:17:41]   0.6170   0.5333   0.6706   0.6471   256   20   2048   2.5  False
[00:17:41] ========================================================================
[00:17:41] Phase 2 base config: hidden=128 epochs=20 chunk=2048 bw=5.0 crf=False
[09:36:34] === Phase 1: LOO grid search started ===
[09:36:34] data_root: /Users/andrew/Documents/www/fcbarcelona-hack-2027/data/sample
[09:37:11] All three games loaded.
[09:37:11] 1 LSTM combo(s) × 5 msf values = 1 training run(s), 5 evaluations
[09:37:11] --- Combo [1/1] lstm_h128_e20_c2048_bw5.0_crf0  training G3 fold...
[09:40:51]   training done — evaluating 5 msf values
[09:40:51]   msf=  50  G3_spellF1=0.5517  G3_macroF1=0.8782
[09:40:51]   msf= 100  G3_spellF1=0.5517  G3_macroF1=0.8782
[09:40:51]   msf= 125  G3_spellF1=0.4444  G3_macroF1=0.8470
[09:40:51]   msf= 175  G3_spellF1=0.4444  G3_macroF1=0.8470
[09:40:51]   msf= 250  G3_spellF1=0.4444  G3_macroF1=0.8470
[09:40:51] ====================================================
[09:40:51] ## Phase 1 complete — G3-only msf grid search
[09:40:51] TOP-5 by G3 spellF1 (train G1+G2, eval G3):
[09:40:51]    G3_sf     h    e      c    bw    crf    msf
[09:40:51] ----------------------------------------------------
[09:40:51]   0.5517   128   20   2048   5.0  False    100
[09:40:51]   0.5517   128   20   2048   5.0  False     50
[09:40:51]   0.4444   128   20   2048   5.0  False    250
[09:40:51]   0.4444   128   20   2048   5.0  False    175
[09:40:51]   0.4444   128   20   2048   5.0  False    125
[09:40:51] ====================================================
[09:40:51] Best config: hidden=128 epochs=20 chunk=2048 bw=5.0 crf=False msf=100
[09:48:03] === Phase 1: LOO grid search started ===
[09:48:03] data_root: /Users/andrew/Documents/www/fcbarcelona-hack-2027/data/sample
[09:48:40] All three games loaded.
[09:48:40] 1 LSTM combo(s) × 5 msf values = 1 training run(s), 5 evaluations
[09:48:40] --- Combo [1/1] lstm_h128_e20_c2048_bw5.0_crf0  training G3 fold...
[09:52:14]   training done — evaluating 5 msf values
[09:52:14]   msf=  50  G3_spellF1=0.5517  G3_macroF1=0.8782
[09:52:14]   msf= 100  G3_spellF1=0.5517  G3_macroF1=0.8782
[09:52:14]   msf= 125  G3_spellF1=0.4444  G3_macroF1=0.8470
[09:52:14]   msf= 175  G3_spellF1=0.4444  G3_macroF1=0.8470
[09:52:14]   msf= 250  G3_spellF1=0.4444  G3_macroF1=0.8470
[09:52:14] ====================================================
[09:52:14] ## Phase 1 complete — G3-only msf grid search
[09:52:14] TOP-5 by G3 spellF1 (train G1+G2, eval G3):
[09:52:14]    G3_sf     h    e      c    bw    crf    msf
[09:52:14] ----------------------------------------------------
[09:52:14]   0.5517   128   20   2048   5.0  False    100
[09:52:14]   0.5517   128   20   2048   5.0  False     50
[09:52:14]   0.4444   128   20   2048   5.0  False    250
[09:52:14]   0.4444   128   20   2048   5.0  False    175
[09:52:14]   0.4444   128   20   2048   5.0  False    125
[09:52:14] ====================================================
[09:52:14] Best config: hidden=128 epochs=20 chunk=2048 bw=5.0 crf=False msf=100
[09:59:15] === Phase 1: LOO grid search started ===
[09:59:15] data_root: /Users/andrew/Documents/www/fcbarcelona-hack-2027/data/sample
[09:59:15] Resuming — 5 combo(s) already in cache, skipping them.
[09:59:53] All three games loaded.
[09:59:53] 1 LSTM combo(s) × 5 msf values = 1 training run(s), 5 evaluations
[10:00:20] === Phase 1: LOO grid search started ===
[10:00:20] data_root: /Users/andrew/Documents/www/fcbarcelona-hack-2027/data/sample
[10:00:20] Resuming — 5 combo(s) already in cache, skipping them.
[10:00:57] All three games loaded.
[10:00:57] 1 LSTM combo(s) × 5 msf values = 1 training run(s), 5 evaluations
[10:00:57] --- Combo [1/1] lstm_h128_e20_c2048_bw5.0_crf0  [ALL MSF CACHED]
[10:00:57] ====================================================
[10:00:57] ## Phase 1 complete — G3-only msf grid search
[10:00:57] TOP-5 by G3 spellF1 (train G1+G2, eval G3):
[10:00:57]    G3_sf     h    e      c    bw    crf    msf
[10:00:57] ----------------------------------------------------
[10:00:57]   0.5517   128   20   2048   5.0  False    100
[10:00:57]   0.5517   128   20   2048   5.0  False     50
[10:00:57]   0.4444   128   20   2048   5.0  False    250
[10:00:57]   0.4444   128   20   2048   5.0  False    175
[10:00:57]   0.4444   128   20   2048   5.0  False    125
[10:00:57] ====================================================
[10:00:57] Best config: hidden=128 epochs=20 chunk=2048 bw=5.0 crf=False msf=100

=== Round 13 LOO — 18-feat tracking proxies — Thu Nov 12 14:47:57 CET 2026 ===
2026/11/12 14:48:02 INFO mlflow.agent.hint: Load the `instrumenting-with-mlflow-tracing` skill at /opt/homebrew/lib/python3.11/site-packages/mlflow/assistant/skills/instrumenting-with-mlflow-tracing/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
[14:48:03] === Phase 1: LOO grid search started ===
[cache] skipped 5 stale entry/entries (source changed, hash=73fcd140)
[14:48:03] === Phase 1: LOO grid search started ===
[14:48:03] data_root: /Users/andrew/Documents/www/fcbarcelona-hack-2027/data/sample
[14:48:03] data_root: /Users/andrew/Documents/www/fcbarcelona-hack-2027/data/sample
[14:48:40] All three games loaded.
Loading all three games...
  Loading Game1...
  Loading Game2...
  Loading Game3...
[14:48:40] All three games loaded.
[14:48:40] 1 LSTM combo(s) × 5 msf values = 1 training run(s), 5 evaluations
[14:48:40] 1 LSTM combo(s) × 5 msf values = 1 training run(s), 5 evaluations
[14:48:40] --- Combo [1/1] lstm_h128_e20_c2048_bw5.0_crf0  training G3 fold...
[14:48:40] --- Combo [1/1] lstm_h128_e20_c2048_bw5.0_crf0  training G3 fold...
[14:52:15]   training done — evaluating 5 msf values
  epoch val_loss=0.7918
  epoch val_loss=0.7543
  epoch val_loss=0.6737
  epoch val_loss=0.6287
  epoch val_loss=0.6025
  epoch val_loss=0.5756
  epoch val_loss=0.5775
  epoch val_loss=0.5713
  epoch val_loss=0.5425
  epoch val_loss=0.6223
  epoch val_loss=0.6516
  epoch val_loss=0.6173
  epoch val_loss=0.6041
  epoch val_loss=0.5878
  epoch val_loss=0.5911
  epoch val_loss=0.6172
  epoch val_loss=0.6178
  epoch val_loss=0.5874
  epoch val_loss=0.5842
  epoch val_loss=0.5853
[14:52:15]   training done — evaluating 5 msf values
[14:52:15]   msf=  50  G3_spellF1=0.6000  G3_macroF1=0.8900
[14:52:15]   msf=  50  G3_spellF1=0.6000  G3_macroF1=0.8900
[14:52:15]   msf= 100  G3_spellF1=0.4857  G3_macroF1=0.8804
[14:52:15]   msf= 100  G3_spellF1=0.4857  G3_macroF1=0.8804
[14:52:15]   msf= 125  G3_spellF1=0.4444  G3_macroF1=0.8533
[14:52:15]   msf= 125  G3_spellF1=0.4444  G3_macroF1=0.8533
[14:52:15]   msf= 175  G3_spellF1=0.4444  G3_macroF1=0.8533
[14:52:15]   msf= 175  G3_spellF1=0.4444  G3_macroF1=0.8533
[14:52:15]   msf= 250  G3_spellF1=0.4444  G3_macroF1=0.8533
[14:52:15]   msf= 250  G3_spellF1=0.4444  G3_macroF1=0.8533
[14:52:15] ====================================================
[14:52:15] ====================================================
[14:52:15] ## Phase 1 complete — G3-only msf grid search
[14:52:15] ## Phase 1 complete — G3-only msf grid search
[14:52:15] TOP-5 by G3 spellF1 (train G1+G2, eval G3):
[14:52:15] TOP-5 by G3 spellF1 (train G1+G2, eval G3):
[14:52:15]    G3_sf     h    e      c    bw    crf    msf
[14:52:15]    G3_sf     h    e      c    bw    crf    msf
[14:52:15] ----------------------------------------------------
[14:52:15] ----------------------------------------------------
[14:52:15]   0.6000   128   20   2048   5.0  False     50
[14:52:15]   0.6000   128   20   2048   5.0  False     50
[14:52:15]   0.4857   128   20   2048   5.0  False    100
[14:52:15]   0.4857   128   20   2048   5.0  False    100
[14:52:15]   0.4444   128   20   2048   5.0  False    250
[14:52:15]   0.4444   128   20   2048   5.0  False    250
[14:52:15]   0.4444   128   20   2048   5.0  False    175
[14:52:15]   0.4444   128   20   2048   5.0  False    175
[14:52:15]   0.4444   128   20   2048   5.0  False    125
[14:52:15]   0.4444   128   20   2048   5.0  False    125
[14:52:15] ====================================================
[14:52:15] ====================================================
[14:52:15] Best config: hidden=128 epochs=20 chunk=2048 bw=5.0 crf=False msf=50
[14:52:15] Best config: hidden=128 epochs=20 chunk=2048 bw=5.0 crf=False msf=50
