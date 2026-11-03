# RESULTS: fear in the news before, during and after COVID (27 Oct 2026)
Plan (frozen before computing): `PLAN-analysis-covid-2026-10-27.md`. Script: `pipeline/analysis_covid.py` → `data/processed/final/analysis_covid.json`. Model: Noesis fear_v7 (v7e) + subject head. "Scary" = level 2–3 on the user's scale (U1).

## In plain words
- **News got scarier and stayed scarier.** In NYT news, scary headlines rose from **7.1%** (2017–19) to **10.1%** (2024–26), about 40% more (+3.0 points, 95% CI 2.7–3.3). The Guardian rose the same amount (5.9% → 8.9%, CI +1.9 to +3.9).
- **COVID was a spike, not the cause of the lasting rise.** Health news jumped in 2020–21 (NYT: from 6% to 16% of headlines, and much scarier), then went back to normal by 2024. The Guardian peaked in 2020 (11.4% scary).
- **What kept it high is war and disasters.** In NYT news, about **90% of the rise is agenda** (which subjects get covered), not tone: war/terror went from 7.6% to 12% of headlines (Ukraine 2022, Gaza 2023, Iran 2026), and disasters/accidents from 4.3% to 7.0%. Within each subject, scariness barely changed. The Guardian is similar (about two-thirds agenda).
- **Opinion pages did not get scarier:** NYT opinion bumped up during COVID, then returned to the pre-COVID level (1.9%).
- **Alarm-level (3) headlines** rose a little: NYT 1.6% → 2.1%.
- **Correction (3 Nov):** an earlier version cited 80 "random" headlines as proof the overall level was right. Those items excluded keyword matches, so they don't show that. See section 5.4 of the [final report](final-report.md).

## Main tables
**Scary share by period** (generic repeats excluded)

| | pre-COVID 2017–19 | COVID 2020–21 | after 2022–23 | recent 2024–26 | recent − pre (95% CI) |
|---|---|---|---|---|---|
| NYT news | 7.1% | 8.7% | 9.9% | 10.1% | +3.0 (2.7 to 3.3) |
| Guardian (random days) | 5.9% | 10.3% | 8.7% | 8.9% | +3.0 (1.9 to 3.9) |
| NYT opinion | 2.1% | 2.9% | 2.5% | 1.9% | −0.2 (−0.6 to +0.1) |

**NYT news by year:** 2017 6.8% · 2018 6.9% · 2019 7.7% · 2020 8.4% · 2021 9.1% · 2022 9.5% · 2023 10.2% · 2024 10.1% · 2025 9.9% · 2026 10.3% (to Oct).

**Where the NYT rise comes from** (shift-share, pre-COVID → recent, +2.97 points): agenda +2.73, framing +0.13, interaction +0.11. By subject: war/terror +1.48, disaster/accident +1.47, health +0.10, crime −0.10, the rest ~0.

**NYT news, share of headlines (scary share within the subject):**

| subject | pre-COVID | COVID | after | recent |
|---|---|---|---|---|
| war/terror | 7.6% (31%) | 6.1% (30%) | 12.0% (32%) | 12.0% (32%) |
| disaster/accident | 4.3% (45%) | 4.7% (46%) | 5.9% (47%) | 7.0% (49%) |
| crime/justice | 8.1% (26%) | 7.3% (25%) | 8.6% (25%) | 8.7% (23%) |
| health | 6.0% (6%) | 16.0% (15%) | 7.4% (10%) | 5.7% (8%) |
| politics | 15.1% | 14.3% | 12.2% | 17.9% |
| sport | 11.3% | 9.7% | 9.6% | 4.8% |

## Robustness (all point the same way)
- Using the outlets' own topic tags instead of the model's subjects: NYT +3.4 points, agenda-driven, led by war/terror and disasters. Guardian +5.4 points (tags are mostly from the threat-tag design there, so this check is weaker).
- Confident-subject rows only: NYT 11.6% → 17.8%; Guardian 7.6% → 12.0%.
- Including generic repeated headlines: unchanged (NYT 6.9% → 9.8%).

## Long view (ABC, 2003–2021)
Scary share was 12–14% in 2003–09, fell to about 10.5% in 2012–14, then rose to 14% in 2019, 16% in 2020 and 18% in 2021. ABC data ends in 2021.

## Caveats
- One person's scale (U1). The model is somewhat weaker on 2022–26 news (sealed #5: κ .88) than on older news (.93), and tends to call some mild (1) headlines scary (2). See section 5.4 of the [final report](final-report.md) for accuracy on ordinary news.
- Subjects are about 2 in 3 right (85–91% when confident); the subject conclusions match the outlet-tag check.
- NYT sport share fell partly because the NYT closed its sports desk in 2023 (agenda artefact, not fear). Jan 2026 NYT is under-collected; 2026 runs to October.
- Headlines only, not article text; NYT and Guardian only after 2021.
