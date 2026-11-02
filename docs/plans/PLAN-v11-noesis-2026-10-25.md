# PLAN v11: Noesis fear_v7, v10 put on the user's own scale (frozen 25 Oct 2026, before any computing)

## Why
The user rated 100 dev headlines by hand (`my-ratings/MY_100_final.csv`, after 21 agreed corrections for missing context). Findings:
- v10 ranks headlines as well as its teachers do against the user (ρ .74 vs .73), so more model tweaks have a low ceiling.
- Every rater, and the model, is about **+0.44 too fearful** on the user's scale (rater 3 → user ≈ 2.2; 2 → 1.1; 1 → 0.6).
- Cursor's part labels agree least with the user (ρ .58).
Noesis today runs fear_v6 (MiniLM, fear ρ .70, disaster-only training). It should carry v10's gains, on the user's scale, at phone size. No downloads.

## Steps
1. **Codebook U1** (`docs/codebook.md`): the user's rules, written from the 100 verified ratings: rescued / safe / over = 0; figurative = 0; investigations and aftermath = 1; animals only = 0; a single accidental death = 2; mass or violent deaths = 3; missing people with live danger = 2–3; local job losses in the hundreds = 2. This becomes the reference for future labels.
2. **Calibration to the user:** isotonic map from the v10 5-seed average to the user's 100 ratings. Report the 5-fold cross-validated MAE (fitting all 100 then applying is optimistic, so CV is the honest number).
3. **Distillation:** teacher = calibrated v10 average. Student = MiniLM-L6 (local cache), fine-tuned by MSE on the teacher's scores for ~300k random ABC headlines plus every labelled text whose labels are not in the dev sets. 1 epoch, seed 0. Refit v6's focus head (logistic on the disaster focus labels, train rows only) on the student's embeddings, so the interface is unchanged.
4. **Tests (dev sets only; no sealed set exists now, so this is development, not a final claim):**
   - everyday 400 ρ, disaster 11 events ρ, weak-spot dev 128 MAE, synthetic dev direction, focus accuracy;
   - the user's 100: ρ (the calibration is monotone, so rank is fair) and MAE (optimistic, labelled as such);
   - compared: v6 (now in Noesis) vs v10 teacher vs v7 student.
   - **Gate to offer v7 as the Noesis default:** v7 ρ ≥ v6 on everyday AND disaster AND the user's 100, and focus accuracy ≥ v6 − 2 points. Otherwise v7 ships behind a flag only, and that is reported.
5. **Ship:** `Memory-Engine/classifier/fear_v7.py` with the same `appraise()` output as v6 (fear, level, fear_range, focus, focus tiers), flag `NOESIS_FEAR_V7=1`. v6 stays the default until the user approves. Not committed (Noesis classifier is uncommitted by the user's choice).

## Parked (not in v11)
T2 / T5 / T7 synthetic redo with write-then-check; bigger encoder; relabelling the training data with codebook U1. A fresh sealed set by the user is needed before any new final claim.

## Amendment 1 (25 Oct, after the one-time sealed scary test; frozen before the fix runs)
- **Result (sealed scary set, 300 NYT, U1):** AUC v6 .936 / v10 raw .968 / v10 calibrated .964 / v7 .965. Recall of 3s: .40 / .65 / .28 / .28. False alarms on tricky calm items: .34 / .17 / .13 / .12. Outlet 600: v7 caught 0 of 9 threes. The focus gate also failed (v7 .682 vs v6 .756).
- **Cause:** the isotonic calibration fit on the user's 100 maps every raw score from 1.95 to 2.99 to 2.0, so the top end is flat. The 100 had few 3s and predate the "violent death = 3" rule.
- **Fix:**
  1. Recalibrate on about 600 U1 labels: the user's 100 plus the 500 labelled NYT pool items NOT in the sealed 300. Use a smooth monotone map: isotonic on 12 quantile-bin means, linear between bin centres, ends anchored at 0→0 and 3→3. Report 5-fold CV (MAE, ρ, recall of 3s at ≥ 2.5) for raw, old and new calibration.
  2. Rebuild v7: same corpus and the cached teacher scores, re-targeted with the new map. Train focus jointly (a cross-entropy focus head on the disaster train rows during fine-tuning) so the encoder learns story type the way v6's did.
  3. **Fresh sealed test:** about 300 new NYT headlines, labelled blind by Claude on U1 (150 at 2–3, 150 at 0–1, half of the low side tricky calm), drawn excluding all earlier pools.
- **Gate:** recall of 3s ≥ v10 raw − .05; AUC ≥ .95; false alarms on tricky calm ≤ v6's; focus accuracy ≥ v6 − 2 points (dev). Pass → v7 is offered as the Noesis default.
- **Amendment 1b (same day, before the rebuild):** the smooth calibration still flattened the top end (CV recall of 3s .53 vs raw .71): mean-error calibration pulls extremes toward the average by design. Replaced by **level cut-points** on the raw v10 score, chosen by quadratic kappa on the 600 U1 labels: **0.95 / 1.9 / 2.65**, the same in all 5 folds. CV: kappa .657 (raw rounding .557), 3s recall .65 / precision .50, 2–3 recall .72 / precision .51. v7b's target = a piecewise-linear map that sends these cuts to the level boundaries.

## Amendment 2 (25 Oct, after sealed set #2; frozen before tuning)
- **Sealed #2 (300 fresh NYT):** v7b κ .773, AUC .925, recall of 2–3s .647 (precision .933), recall of 3s .573, false alarms on tricky calm .080. v6: .594 / .879 / .793 / .773 / .387 / .373. Gate FAIL (3s .573 vs bar .577; AUC bar .95 not met by any model). Main error: real 2s called 1 (33 of 75).
- **Fix, agreed with the user ("catch more 2s"):** re-tune the level cut-points on v7b's OUTPUT, keeping the model. Tuning data: 1,200 U1 labels (sealed #1 + sealed #2, both spent, plus the 600 calibration labels). Rule: pick the 2-cut with the highest recall of 2–3s subject to false alarms on tricky calm ≤ .15; then the 3-cut by quadratic κ. Report 5-fold CV. Ship as a remap in fear_v7.py.
- **Confirm on fresh sealed set #3** (300 NYT, same design, labelled blind by Claude). Gate: recall of 2–3s ≥ .78, false alarms on tricky calm ≤ .15, recall of 3s ≥ .55, κ ≥ v6 + .10. Pass → v7b becomes the Noesis default, with the user's OK.

## Amendment 3 (25 Oct): result of sealed #3, v7b made the default, and the 2s pack
- **Sealed #3 (300 fresh NYT):** v7b κ .829 (v10 .797, v6 .609), AUC .951, 3s caught .693, false alarms on tricky calm .107, 2–3s caught .747 (bar .78: **FAIL**, the only miss). Main error: real 2s called 1 (30 of 75).
- **The user approved v7b as the Noesis default.** `classifier.process_text` now uses fear_v7 (`fear_score` = fear/3, plus level, label, range and focus). The grammar score is kept as `fear_grammar` and used as the fallback. `NOESIS_FEAR_V7=0` reverts.
- **2s pack (training only):** about 600 real ABC headlines NOT in any eval set, picked by v7b's own score: 450 near the 1/2 boundary (remapped 0.9–2.1) and 150 above 2.1. Labelled blind by Claude on U1. Added to the v7b student training as direct labels (weight ×3) alongside the teacher targets. Retrain, refit heads, re-tune cuts by the same rule on the 900 spent NYT sealed labels plus the 600 calibration labels (CV).
- **Gate:** CV on those labels: 2–3s caught ≥ .78, false alarms ≤ .15, 3s caught ≥ .65, κ ≥ .80. A final claim needs fresh sealed set #4.

## Amendment 4 (25 Oct): v7c is live; fix the two leans (accidents scored 3, escapes scored 2)
- **v7c on the 1,500 U1 tuning labels** (cuts fit on the same set, so slightly flattering): exact level 68% (v6 37%), within one level 95%, scary-vs-calm 87%. Per true level: 0 → 80%, 1 → 53%, **2 → 52%**, 3 → 71%. Of the true 2s, 48 are called 3 and 62 are called 1.
- **Boundary pack B (training only):** about 300 real ABC headlines not in any eval set or the 2s pack: about 150 accidental deaths and injuries (crash / drown / fall / electrocuted / hit by, with no violence word) and about 150 escape / rescue / found-safe / survive lines. Picked by keyword (sampling only, not a model feature) and labelled blind by Claude on U1. Added like the 2s pack (×3). Retrain → v7d, re-tune the cuts by the same rule, 5-fold CV on the 1,500.
- **Gate:** v7d κ ≥ v7c; 2–3s caught ≥ .78; false alarms ≤ .15; 3s caught ≥ .65; true-2 exact ≥ v7c + .03; focus ≥ v6 − .02. Pass → replace v7c (back it up as fear_v7c).
- **Amendment 4 result:** v7d FAIL (κ .808 vs .812; true-2 exact .475 vs the .526 needed). Better on 3s (.737), the escape slice (.11 vs .22), the accident slice (.44 vs .50; slices tiny) and the outlet set (≥2 recall .74 vs .65, 4 of 9 threes). Worse on true 1s (.456 vs .528). v7c stays live; v7d is saved in data/processed/events/v11/student_v7d/.

## Amendment 5 (25 Oct): consistency check, then a U1 teacher
- **Step 2, averaging v7c and v7d: null.** κ .812 vs .813; true-2 exact .508 vs .496; false alarms .156 > .15. Not kept.
- **Step 1, consistency:** a fresh labeller with codebook U1 re-rated 150 of Claude's labels blind: exact 86%, quadratic κ .926 (98% exact when it felt sure). v7c on the same items: exact 64%, κ .82. **The labels are not the ceiling; the model is.** Cause: the v10 teacher was trained on AI-rater G1 labels, not U1.
- **Step 3:** 3 rules added to U1 (charges or verdicts over killings = 1; a fugitive on the run = 2; an accident inquiry = 1). 34 label lines changed accordingly.
- **Plan:**
  - **(a) U1 corpus:** every Claude/user U1 label (NYT pools 1–3, 2s pack, boundary B, outlet 600, the user's 100): about 4,700 unique items. Split by seed: 80% train / 20% dev, stratified by source and level.
  - **(b) Teacher v12:** MPNet with the v10 F2F3 recipe (CORAL ordinal head, pair loss, synthetic pairs T1 T3 T4 T6 T7 T8). Two arms, picked by dev κ: (i) U1 train only, ×2; (ii) U1 train ×2 plus the old G1/v2 rater rows ×0.5. Then 5 seeds of the winner.
  - **(c) Student v7e:** MiniLM distilled from the v12 average (the v7c recipe: 300k ABC teacher targets, U1 train rows as direct labels ×3, joint focus). Cuts by the same rule on dev, with CV inside dev.
  - **(d) Dev gate:** v7e κ ≥ v7c + .02 and true-2 exact ≥ v7c + .05 on dev; 2–3s caught ≥ .78; false alarms ≤ .15; 3s caught ≥ .65; focus ≥ v6 − .02.
  - **(e) Sealed set #4** (fresh NYT, 300, labelled blind by Claude on the updated U1, drawn excluding everything) is opened once on v7c vs v7e. v7e replaces v7c only if it beats v7c on κ and true-2 exact there.
- **Amendment 5 result (26 Oct):**
  - U1 corpus: 3,900 items (3,118 train / 782 dev).
  - v12 teacher: arm ii (U1 ×2 plus the old rater rows ×0.5) won. 5-seed average on dev: κ .843, true-2 exact .789.
  - v7e student: passed all 6 dev gates (κ .832 vs v7c .812; true-2 .728 vs .526).
  - **Sealed #4 (opened once):** v7e κ .927, exact .807, true-2 .773, true-3 .827, 2–3s caught .940 (precision .916), false alarms .173, AUC .977. v7c: .850 / .693 / .533 / .773 / .873 / .200. **v7e installed as Noesis fear_v7** (v7c backed up as models/fear_v7c). Spot checks match U1 (accident 2, escaped 0, fugitive 2, court 1).
