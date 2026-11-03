# PLAN: data audit after the NYT blog error (frozen 3 Nov 2026, before any audit numbers are looked at)

Trigger: blog posts were 38% of 2008 "news" headlines and moved the trend. Question: what else in the archive changed over time and could change the study's output?

## Checks (all reported, including those that find nothing)
**A. What counts as NYT "news", by year:** document type, type of material, section, desk, URL pattern, print vs web, byline/wire share.
**B. Non-journalism or automated items:** death notices, obituaries, corrections, lists, lottery, briefings, automated or templated pages (earthquake, storm, wildfire, flood, election maps/trackers), live-update pages, interactive features, video, podcasts. Near-duplicate templates the exact-repeat "generic" filter misses.
**C. Coverage:** days and items per month, gaps, under-collection, duplicates by URL and by headline, per year.
**D. Headline style drift:** length, sentence style, web vs print headline. Does scary share depend on length, and does the rise survive holding length fixed? Rescore print headlines for the same articles in 2017-19 and 2024-26.
**E. Calm-section changes:** sections that were added, closed or moved (sport, games, cooking, etc.), and the rise with each removed.
**F. Tag vocabulary:** whether NYT tags changed in a way that affects the outlet-tag check (8.8% to 20.5%).
**G. Guardian and ABC:** design and section mix by year, liveblogs, whether the sampling design changes between periods, ABC item types and regional/audio items.
**H. Scoring integrity:** rescore a random sample per year with the live model and compare; missing values; all rows on one model version.
**I. Model drift:** existing labelled data by era and error rates by year band.

## Decision rule
An issue is material if removing it moves a headline result by at least 0.3 points or changes its direction, or removes/moves a turning point. Material issues are fixed in the analysis; the corrected base becomes the main result; the old number is shown beside it. Non-material issues are listed with their size.

## Fixes that are already decided
Blog posts leave the NYT news base. Turning points rerun on the corrected base with the original method. Report, README, figures and results are updated. All audit results stay exploratory, since they are post hoc.

## Outcome
Results are in section 3.5 and sections 6.5 to 6.8 of the final report, and `results/analysis_corrected.json`.
