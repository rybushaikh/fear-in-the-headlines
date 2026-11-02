"""Fresh sealed scary-news set #5 (27 Oct 2026, PLAN-v12 step 1): sealed #4 design (model-free cue pool + random) on NYT 2022-01..2026-10,
excluding every earlier pool and the U1 corpus. 520 cue + 80 random, seed 7001/7002; labelled blind by Claude on U1."""
import json, gzip, glob, re, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1]; E = R / "data/eval"; P = E / "scary_test5_private"; P.mkdir(exist_ok=True)
src = open(R / "pipeline/prep_scary_test.py").read(); CUE = re.compile(re.search(r'CUE = re.compile\(r"(.+?)", re.I\)', src).group(1), re.I)
seen = set(pd.read_parquet(E / "outlets_private/key.parquet").text.str.lower()) | set(pd.read_parquet(R / "data/processed/events/v12/u1_corpus.parquet").text.str.lower())
for k in "", "2", "3", "4": seen |= set(pd.read_parquet(E / f"scary_test{k}_private/pool_key.parquet").text.str.lower())
rows = []
for f in sorted(glob.glob(str(R / "data/raw/outlets/nyt/202[2-6]-*.json.gz"))):
    for x in json.load(gzip.open(f))["response"]["docs"]:
        if (x.get("document_type") or "").lower() != "article" or x.get("type_of_material") in ("Paid Death Notice", "Correction"): continue
        h = ((x.get("headline") or {}).get("main") or "").strip()
        if len(h.split()) >= 4 and h.lower() not in seen: rows.append(dict(date=x["pub_date"][:10], text=h))
N = pd.DataFrame(rows); N["k"] = N.text.str.lower(); N = N[N.groupby("k").text.transform("size") < 5].drop_duplicates("k").drop(columns="k")
N = N[N.date.between("2022-01-01", "2026-10-31")]; N["cue"] = N.text.str.contains(CUE)
S = pd.concat([N[N.cue].sample(520, random_state=7001).assign(pool="cue"), N[~N.cue].sample(80, random_state=7002).assign(pool="random")]).sample(frac=1, random_state=7003).reset_index(drop=True)
S["id"] = [f"r{i:04d}" for i in range(len(S))]; S.to_parquet(P / "pool_key.parquet")
S[["id", "text"]].to_csv(P / "to_label.tsv", sep="\t", index=False)
print(len(N), "candidates;", len(S), S.pool.value_counts().to_dict(), S.date.str[:4].value_counts().sort_index().to_dict())
