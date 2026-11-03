# Fear in the Headlines

**Is the news getting scarier, and why?** This project scores 3.35 million headlines (NYT, Guardian, ABC Australia, 2003–2026) for fear with a small model trained on one reader's written fear scale.

> **Status: ongoing.** This is research code and a report on the work so far, not a finished product.

![Share of scary headlines by year](docs/figures/figure1-scary-share-by-year.png)

## What we found

> **Corrected on 3 November 2026.** A year-by-year data audit found blog posts, repeats and closed sections in the NYT "news" data, and it changed several earlier results. [See what changed](docs/final-report.md#35-data-audit-and-corrected-nyt-base). NYT results use the corrected data and are exploratory.

- **War and disaster went from about 1 in 7 NYT news headlines to about 1 in 5** (13.7% in 2017–19 to 19.9% in 2024–26). That's the biggest shift, and the NYT's own tags show it more strongly (10.3% to 21.3%).
- **Scary headlines rose and stayed up.** NYT news went from 8.2% to 10.7%, +2.4 points (95% CI 2.1 to 2.7). The Guardian rose 3.0 points, from 5.9% to 8.9% (CI 1.9 to 3.9).
- **It's a wave, not a straight climb.** War and disaster were already 18% of NYT headlines in 2003–06, so we don't claim a 24-year high at the NYT.
- **The size depends on what counts as news.** The rise is +1.0 in print only and +1.5 without the NYT's quick-news and weather desks. The direction holds in every version.
- **It really is daily.** In 2024–26, NYT news carried scary headlines on 100% of days with data, about 10 a day, yet about 89% of its headlines weren't scary.
- **COVID was a spike.** Health news surged in 2020–21, then returned to normal by 2024.
- **The rise since 2017 is about what gets covered.** About 93% of the NYT rise is agenda, not tone. In the Guardian it's about two-thirds.
- **Opinion pages had no lasting rise.**
- **What this can't tell you:** it measures coverage, not events, so it can't say whether the world got more dangerous or newsrooms paid danger more attention. "Scary" means danger to life and safety, so a financial panic like 2008 barely registers.

![What drove the rise in scary NYT headlines](docs/figures/figure2-nyt-rise-bridge.png)

"Scary" means level 2 or 3 on the [fear scale](docs/codebook.md). Read the full **[final report](docs/final-report.md)** ([PDF](docs/final-report.pdf)): method, results, discussion, limitations, references and appendices.

## How well the model works

Each sealed test has 300 unseen NYT headlines, half of them scary, labelled blind and opened once.

| Test | Agreement (weighted κ) | Exact level | Scary caught | False alarms on tricky calm news |
|---|---|---|---|---|
| NYT 2003–21 | .93 | 81% | 94% | 17% |
| NYT 2022–26 | .88 | 77% | 87% | 23% |

Two separate AI labelling sessions agreed at κ .93, and the model scored κ .81 on 20 held-out hand ratings by the reader. The weakest level is mild news (level 1), where only about half are right. **Use it for overall trends:** reweighted to ordinary news, the overall scary share is close to the labels for recent news, but only about half of individual flagged headlines are truly scary.

## Repository map

| Path | Contents |
|---|---|
| [`docs/final-report.md`](docs/final-report.md) | The full final report |
| [`docs/results-covid.md`](docs/results-covid.md) | Detailed results tables for the COVID-era analysis |
| [`docs/codebook.md`](docs/codebook.md) | The fear scale and its rules |
| [`docs/plans/`](docs/plans) | Plans written before results were computed, with their outcomes |
| [`pipeline/`](pipeline) | The scripts behind the final model and analysis (see below) |
| [`noesis_fear/`](noesis_fear) | Scoring code and small model head files |
| [`results/`](results) | Aggregate results as JSON |

### Pipeline

| Stage | Scripts |
|---|---|
| 1. Collect | `outlets_fetch`, `guardian_alltopics`, `guardian_targeted`, `recent_fetch` |
| 2. Build labels and sealed tests | `build_u1_corpus`, `make_synthetic`, `prep_scary_test`, `prep_scary_test4`, `prep_scary_test5`, `prep_v7f_data` |
| 3. Train | `v10_model`, `train_v12` (teacher), `build_fear_v7e` (student), `build_subject`, `build_subject_rules` |
| 4. Evaluate | `u1_eval`, `eval_scary_sealed4`, `eval_v7f_sealed5` |
| 5. Score and analyse | `score_all_v7e`, `analysis_covid`, `analysis_transitions`, `analysis_transitions2`, `analysis_corrected` |

## What's not included, and what that means

- **Headline text, labels and per-headline scores** are left out. Some of this data is available on request: see [DATA_ACCESS.md](DATA_ACCESS.md).
- **The encoder weights (88 MB)** aren't published yet, so `noesis_fear` can't score text on its own.

So you can read the method, inspect the code and check the aggregate results, but you can't re-run training or scoring from this repo alone. Scripts expect local data under `data/`. To collect headlines, copy `.env.example` to `.env` and add your own NYT and Guardian API keys.

## Limits

One reader's scale. Headlines only. Most labels came from an AI labeller following the written codebook. Only the NYT and the Guardian run past 2021. About 1 subject in 3 is wrong (the conclusions also hold with the outlets' own tags). It measures coverage, not world events. NYT results use a corrected base chosen after the first results were seen.

## Data sources and licence

NYT Archive API, Guardian Content API, [A Million News Headlines](https://doi.org/10.7910/DVN/SYBGZL) (ABC, CC0) and [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) (HuffPost, CC BY 4.0). Code is under the [MIT licence](LICENSE), and the report, docs and figures are under [CC BY 4.0](LICENSE-DOCS.md), so you can reuse them with credit. Neither covers third-party data, and the cleaned data needs [permission](DATA_ACCESS.md). See [SECURITY.md](SECURITY.md) to report a security issue.
