"""Scary-news test set (25 Oct 2026, plan agreed with the user): 300 NYT headlines, 150 rated 2-3 and 150 rated 0-1 on codebook U1, labelled blind by Claude.
Pool (model-free selection): NYT full archive 2003-21, articles, no Paid Death Notice / Correction, 4+ words, not in the outlet-check set.
450 drawn among headlines containing a harm cue word (sampling only, not a model feature) + 150 simple random; seed 1026.
After labelling: 150 from the 2-3 labels (as close to half 2s / half 3s as available), 75 hard negatives (cue pool, 0-1) + 75 random (0-1)."""
import json, gzip, glob, re, numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1]; P = R / "data/eval/scary_test_private"
CUE = re.compile(r"\b(kill|killed|kills|dead|dies|died|death|deaths|murder\w*|shot|shoot\w*|gunman|bomb\w*|blast|explosion|attack\w*|missing|fire|blaze|flood\w*|quake|earthquake|storm|hurricane|crash\w*|collapse\w*|victim\w*|injur\w*|wounded|massacre|hostage\w*|kidnap\w*|outbreak|virus|epidemic|evacuat\w*|terror\w*|rape\w*|stabb\w*|abuse\w*|threat\w*|fear\w*|disaster|toll|layoffs?|jobs)\b", re.I)
out = set(pd.read_parquet(R / "data/eval/outlets_private/key.parquet").text.str.lower())
rows = []
for f in sorted(glob.glob(str(R / "data/raw/outlets/nyt/20*.json.gz"))):
    for x in json.load(gzip.open(f))["response"]["docs"]:
        if x.get("document_type") != "article" or x.get("type_of_material") in ("Paid Death Notice", "Correction"): continue
        h = ((x.get("headline") or {}).get("main") or "").strip()
        if len(h.split()) >= 4 and h.lower() not in out: rows.append(dict(date=x["pub_date"][:10], text=h, material=x.get("type_of_material") or ""))
N = pd.DataFrame(rows).drop_duplicates("text"); N = N[N.date.between("2003-01-01", "2021-12-31")]; N["cue"] = N.text.str.contains(CUE)
print(f"NYT pool {len(N):,}, with cue {N.cue.sum():,}")
S = pd.concat([N[N.cue].sample(450, random_state=1026).assign(pool="cue"), N[~N.text.isin(set())].sample(150, random_state=1027).assign(pool="random")])
S = S.drop_duplicates("text").sample(frac=1, random_state=1028).reset_index(drop=True); S["id"] = [f"n{i:04d}" for i in range(len(S))]
S.to_parquet(P / "pool_key.parquet"); print(len(S), S.pool.value_counts().to_dict())
# Top-up (25 Oct): the first 600 gave only 123 items at 2-3, so 200 more cue-pool headlines are drawn (seed 1029), excluding the first pool.
S2 = N[N.cue & ~N.text.isin(set(S.text))].sample(200, random_state=1029).assign(pool="cue")
S2["id"] = [f"n{i:04d}" for i in range(600, 800)]; pd.concat([S, S2], ignore_index=True).to_parquet(P / "pool_key.parquet"); print("pool now", len(S) + len(S2))
