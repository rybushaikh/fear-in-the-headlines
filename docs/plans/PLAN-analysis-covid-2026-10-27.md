# PLAN: fear in the news before, during and after COVID (frozen 27 Oct 2026, before any numbers are looked at)

Data: `Media-Fear-Research/data/processed/final/all_scored.parquet`, scored by live Noesis fear_v7 (v7e) and the subject head.

## Questions
1. **How much news is scary, by year?** Share of headlines at level ≥ 2 ("scary") and at level 3 ("alarm"; fear ≥ 2.5, the user's round-up rule), 2015–2026.
2. **Which subjects?** For each of the 10 subjects: its share of all headlines (agenda) and its scary share (framing), by year.
3. **Why did the scary share change?** Shift-share decomposition of the change in scary share between periods into agenda (subject mix), framing (within-subject scariness) and interaction.
4. **Outlets and types:** NYT news, NYT opinion, Guardian (random-day sample).

## Periods (fixed)
pre-COVID 2017–2019 · COVID 2020–2021 · after 2022–2023 · recent 2024–2026 (to Oct). Headline comparison: **recent vs pre-COVID**.

## Samples
- **NYT:** full archive, material news and opinion separately ("other" excluded). Jan 2026 is under-collected (~700 articles); kept, noted.
- **Guardian:** the random-day design only (one random day per month, representative); the threat-tag archives are NOT used for shares (they over-sample threat topics).
- **ABC:** 2003–2021 only; used for the long view (2003–2021 yearly scary share), not for after-COVID claims.
- Generic repeated headlines (≥ 5 times in a source) excluded.

## Uncertainty
95% intervals by a day-cluster bootstrap (1,000 resamples, seed 0) for period shares and differences. A difference is reported as real only if its interval excludes 0.

## Robustness (reported next to the main result)
(a) confident-subject rows only; (b) the outlet's own tag subject (NYT/Guardian) instead of the model's subject; (c) mean fear (continuous) instead of the ≥ 2 share; (d) including generic headlines.

## Model check
On the 80 random (model-free) 2022–26 NYT headlines labelled blind for sealed #5, compare the true ≥ 2 share with the model's. Known lean: v7e calls some true 1s "2", so absolute scary shares run high; trends assume the lean is stable across years (sealed #4, 2003–21, and sealed #5, 2022–26, show similar error patterns).

## Outputs
`docs/results-covid.md` (plain-language summary first, then the tables) and `data/processed/final/analysis_covid.json`. Script: `pipeline/analysis_covid.py`.
