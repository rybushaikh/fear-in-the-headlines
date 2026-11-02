"""Fresh sealed scary-news set #4 (25 Oct 2026, PLAN-v11 amendment 1): same design as prep_scary_test.py (model-free cue pool + random, NYT 3003-21),
excluding the first scary pool (800), the outlet set and the 600 calibration texts. 650 cue + 100 random drawn, seed 3001/3002; labelled blind by Claude on U1."""
import json, gzip, glob, re, numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1]; P = R / "data/eval/scary_test4_private"
src = open(R / "pipeline/prep_scary_test.py").read(); CUE = re.compile(re.search(r'CUE = re.compile\(r"(.+?)", re.I\)', src).group(1), re.I)
seen = set(pd.read_parquet(R / "data/eval/outlets_private/key.parquet").text.str.lower()) | set(pd.read_parquet(R / "data/eval/scary_test_private/pool_key.parquet").text.str.lower()) | set(pd.read_parquet(R / "data/eval/scary_test2_private/pool_key.parquet").text.str.lower()) | set(pd.read_parquet(R / "data/eval/scary_test3_private/pool_key.parquet").text.str.lower())
rows = []
for f in sorted(glob.glob(str(R / "data/raw/outlets/nyt/20*.json.gz"))):
    for x in json.load(gzip.open(f))["response"]["docs"]:
        if x.get("document_type") != "article" or x.get("type_of_material") in ("Paid Death Notice", "Correction"): continue
        h = ((x.get("headline") or {}).get("main") or "").strip()
        if len(h.split()) >= 4 and h.lower() not in seen: rows.append(dict(date=x["pub_date"][:10], text=h))
N = pd.DataFrame(rows).drop_duplicates("text"); N = N[N.date.between("2003-01-01", "2021-12-31")]; N["cue"] = N.text.str.contains(CUE)
S = pd.concat([N[N.cue].sample(650, random_state=6001).assign(pool="cue"), N[~N.cue].sample(100, random_state=6002).assign(pool="random")]).sample(frac=1, random_state=6003).reset_index(drop=True)
S["id"] = [f"q{i:04d}" for i in range(len(S))]; S.to_parquet(P / "pool_key.parquet"); print(len(S), S.pool.value_counts().to_dict())
