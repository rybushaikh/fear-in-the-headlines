# PLAN: key transitions and pre-publication checks (frozen 3 Nov 2026, before computing)
Follow-up to the COVID analysis. Exploratory: the COVID-period results were seen first, so this is reported as exploratory.
Data: `all_scored.parquet` (v7e + subject head). NYT news, generics excluded; Guardian random-day for yearly checks.

## Transitions
- T1. NYT news yearly 2003–2026: scary share, alarm share, scary per day. Compare 2024–26 with the highest earlier years.
- T2. Change points on monthly NYT scary share, 2003-01 to 2026-10: least-squares optimal partition, minimum segment 6 months, penalty 3·ln(n)·σ² where σ² is estimated from month-to-month differences. Report segments and their means.
- T3. Subject shares and scary share within subject for each segment.
- T4. The 10 scariest months, with their top subject and their most distinctive words.

## Threats to the findings
- V1. Denominator: headlines per day and scary per day by period; scary share with sport excluded, and with sport and culture excluded.
- V2. Subject-model bias: the fear and subject heads share an encoder, so scary headlines could be pulled into threat subjects, turning framing into apparent agenda. Check: among NYT rows with an outlet tag, repeat the decomposition by tag, and report scary share by tag within each period.
- V3. Natural-mix accuracy: on the random (not keyword-picked) items of sealed pools #4 and #5, which are not in training: κ, exact level, recall and precision of scary, true vs model scary share, with bootstrap intervals.
- V4. Era drift: model vs true scary share in the random items, by pool, as a check on whether the rise could be a model artefact.

## Amendment 1 (3 Nov, after seeing T1–T4 and V1–V4; frozen before computing)
- **Correction found:** "random" items in the sealed pools are drawn only from NON-cue headlines (`N[~N.cue]`). So the 6.2% vs 6.2% check does not validate the overall scary share. The report's calibration statement must be corrected.
- **V5, stratified label-based estimate:** use the blind-labelled pools #1–#5 (cue + non-cue samples). For 2017–19 and 2024–26 separately, true scary share = cue share of NYT articles in that period × scary rate in labelled cue items + (1 − cue share) × scary rate in labelled non-cue items. Same weights for the model's scary rate on the same items. Population: NYT articles of at least 4 words, non-generic, all material types (the pools' population). Bootstrap within strata (1,000, seed 0) for each share and for the rise. Pools #1–#3 use labels as they stand (labelled before some rules were added), and are in training, so model figures from them are optimistic; label-based figures are unaffected. Also report weighted recall and precision of the model on the unseen pools #4–#5.
- **V6, long-run decomposition:** shift-share (model subject) from 2003–06 and from 2011–14 to 2024–26; scary-per-day versus calm-per-day split of the share rise for 2017–19 → 2024–26 and 2007–09 → 2024–26.
