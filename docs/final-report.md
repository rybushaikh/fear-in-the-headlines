# Fear in the Headlines: Final Report

Rahheb Shaikh · Updated 2 November 2026 · Project ongoing

We built a small, offline model that rates news headlines for fear on one reader's written scale, and used it to score 3.35 million headlines. Scary headlines rose by about 3 points after 2019 at both the NYT and the Guardian and have stayed up, mostly because more war and disaster news is being covered, not because news is told more fearfully.

For the full method and references, see the [working paper](paper.md).

## What we accomplished

| Area | Result |
|---|---|
| **Fear scale** | A written four-level codebook, started from 100 hand ratings by the reader |
| **Labelled data** | About 3,900 headlines in the final training corpus, with a further 600-headline test pool for 2022–26 news |
| **Fear model** | An 88 MB model that runs offline: agreement (κ) with the reader's scale up from .60 to .93 on the same unseen test |
| **Subject model** | 10 subjects from the same encoder: 67% right on a hand check, 85% when confident (about half of headlines) |
| **Data** | 3.35 million headlines from the NYT, ABC Australia and the Guardian, de-duplicated, spanning 2003 to October 2026 |
| **Scale of scoring** | All 3.35 million headlines scored in about 16 minutes on a laptop |
| **Analysis** | A before, during and after COVID study with a plan frozen before any numbers were computed |
| **Publication** | A working paper, this report and a security-checked code repository |

## What we found

![Share of scary headlines by year](figures/figure1-scary-share-by-year.png)

"Scary" means level 2 or 3 on the [fear scale](codebook.md).

| Sample | 2017–19 | 2020–21 | 2022–23 | 2024–26 | Change, 2017–19 to 2024–26 (95% CI) |
|---|---|---|---|---|---|
| NYT news | 7.1% | 8.7% | 9.9% | 10.1% | +3.0 points (2.7 to 3.3) |
| Guardian | 5.9% | 10.3% | 8.7% | 8.9% | +3.0 points (1.9 to 3.9) |
| NYT opinion | 2.1% | 2.9% | 2.5% | 1.9% | −0.2 points (−0.6 to 0.1) |

- **News got scarier and stayed that way.** NYT scary headlines rose about 40% in relative terms.
- **COVID was a spike.** Health news rose from 6% to 16% of NYT headlines in 2020–21 and became scarier, then returned to normal by 2024.
- **Agenda drove the lasting rise.** About 90% of the NYT rise comes from what gets covered. In the Guardian it's about two-thirds.
- **War and disasters account for it.** War and terror coverage rose from 7.6% to 12.0% of NYT headlines, and disasters and accidents from 4.3% to 7.0%.
- **Opinion pages had no lasting rise.** They rose during COVID, then returned to their earlier level.

![What drove the rise in scary NYT headlines](figures/figure2-nyt-rise-bridge.png)

## How much the model improved

All three models were tested on the same 300 unseen NYT headlines from 2003–21.

| Measure | First model (v6) | Previous (v7c) | Final (v7e) |
|---|---|---|---|
| Agreement with the reader (κ) | .60 | .85 | **.93** |
| Exact level right | 40% | 69% | **81%** |
| True 2s right | 60% | 53% | **77%** |
| True 3s right | 44% | 77% | **83%** |
| Scary headlines caught | 84% | 87% | **94%** |
| False alarms on tricky calm news | 41% | 20% | **17%** |

On a newer test of 300 headlines from 2022–26, the final model scored κ .88, got 77% of exact levels right and caught 87% of scary headlines. Two independent labellers agreed with each other at κ .93, so the model is close to the human ceiling on older news and a little below it on recent news.

## What worked, and what didn't

- **Worked: training directly on the reader's scale.** This was the biggest single gain, taking κ from .85 to .93.
- **Worked: level cut-points instead of smooth calibration.** Calibration had flattened the top of the scale, leaving only 28% of 3s caught.
- **Worked: sealed tests and frozen plans.** Each test was opened once, so failed ideas couldn't be quietly kept.
- **Didn't work: adding 300 accident and rescue labels (v7d).** It failed its development gate.
- **Didn't work: 197,000 recent headlines plus 600 new labels (v7f).** It caught more scary news (91%), but true 1s fell to 38%, so it was rejected.
- **Didn't work: moving the 1/2 cut-point.** True 1s rose to 66%, but scary news caught fell to 81%.

## Limits

- The scale is one reader's judgement.
- The model reads headlines only, not article text.
- Most labels came from an AI labeller following the written codebook.
- Only the NYT and the Guardian run past 2021, so post-COVID claims rest on those two outlets.
- Mild news is the weakest level: only about half of true 1s are right.
- About 1 subject in 3 is wrong, although the main conclusion also holds with the outlets' own topic tags.
- The paper's references were compiled from memory and are still being checked.

## What's next

1. Check the paper's references.
2. Decide whether to publish the model weights, so others can run it.
3. Add more outlets past 2021, for example through Media Cloud.
4. Build more labelled recent news, because accuracy is lower on 2022–26 headlines.
5. Take the method into Differoo.
