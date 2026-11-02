"""PLAN-v12 steps 1-2 (27 Oct 2026).
1. Seal set #5: from the 600 blind-labelled 2022-26 NYT pool, 150 at 2-3 + 75 tricky calm (cue, <=1) + 75 other calm (random, <=1), seed 7101. The whole pool is barred from training.
2. Recent corpus: NYT 2022-01..2026-10 articles (no death notices / corrections), headlines >= 4 words, deduped (lower-cased; repeats >= 5 dropped), minus every eval/U1 text and the pool.
   Scored with installed v7e (remapped 0-3). Training pack P5 (600, seed 7201): 250 with v7e 1.5-2.5, 150 above 2.5, 200 with v7e 1.0-2.0 AND a harm/court/aftermath cue.
Outputs: scary_test5_private/SEALED_scary5_300.parquet; data/processed/events/v13/recent_corpus.parquet (text, date, v7e); v13/p5_key.parquet + p5_to_label.tsv"""
import sys, json, gzip, glob, re, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "noesis_fear"))
R = Path(__file__).resolve().parents[1]; E = R / "data/eval"; P = E / "scary_test5_private"; O = R / "data/processed/events/v13"; O.mkdir(parents=True, exist_ok=True)
# 1. sealed #5
K = pd.read_parquet(P / "labelled.parquet")
S = pd.concat([K[K.fear >= 2].sample(150, random_state=7101), K[(K.pool == "cue") & (K.fear <= 1)].sample(75, random_state=7102), K[(K.pool == "random") & (K.fear <= 1)].sample(75, random_state=7103)])
S.sample(frac=1, random_state=7104).reset_index(drop=True).to_parquet(P / "SEALED_scary5_300.parquet"); print("sealed #5", len(S), S.fear.value_counts().sort_index().to_dict())
# 2. recent corpus
src = open(R / "pipeline/prep_scary_test.py").read(); CUE = re.compile(re.search(r'CUE = re.compile\(r"(.+?)", re.I\)', src).group(1), re.I)
seen = set(pd.read_parquet(E / "outlets_private/key.parquet").text.str.lower()) | set(pd.read_parquet(R / "data/processed/events/v12/u1_corpus.parquet").text.str.lower())
for k in "", "2", "3", "4", "5": seen |= set(pd.read_parquet(E / f"scary_test{k}_private/pool_key.parquet").text.str.lower())
rows = []
for f in sorted(glob.glob(str(R / "data/raw/outlets/nyt/202[2-6]-*.json.gz"))):
    for x in json.load(gzip.open(f))["response"]["docs"]:
        if (x.get("document_type") or "").lower() != "article" or x.get("type_of_material") in ("Paid Death Notice", "Correction"): continue
        h = ((x.get("headline") or {}).get("main") or "").strip()
        if len(h.split()) >= 4: rows.append(dict(date=x["pub_date"][:10], text=h))
N = pd.DataFrame(rows); N["k"] = N.text.str.lower(); N = N[N.groupby("k").text.transform("size") < 5].drop_duplicates("k")
N = N[N.date.between("2022-01-01", "2026-10-31") & ~N.k.isin(seen)].drop(columns="k").reset_index(drop=True)
import fear_v7
m = fear_v7._load(); Em = m["enc"].encode(N.text.tolist(), batch_size=256, normalize_embeddings=True, show_progress_bar=False)
N["v7e"] = fear_v7._remap((Em @ m["fw"] + m["fb"]).clip(0, 3)); N.to_parquet(O / "recent_corpus.parquet")
print("recent corpus", len(N), "v7e level counts", np.round(N.v7e).value_counts().sort_index().to_dict())
HARM = re.compile(r"\b(kill|dead|death|die[sd]?|murder|shoot|shot|attack|crash|fire|flood|storm|charged|trial|sentenc|convict|guilty|court|inquiry|probe|investigat|after|victim|hostage|bomb|war|strike|threat|injur|wound|missing)", re.I)
a = N[N.v7e.between(1.5, 2.5)].sample(250, random_state=7201); b = N[N.v7e > 2.5].sample(150, random_state=7202)
rest = N.drop(a.index).drop(b.index); c = rest[rest.v7e.between(1.0, 2.0) & rest.text.str.contains(HARM)].sample(200, random_state=7203)
Pk = pd.concat([a.assign(arm="edge2"), b.assign(arm="high3"), c.assign(arm="ones")]).sample(frac=1, random_state=7204).reset_index(drop=True)
Pk["id"] = [f"p{i:04d}" for i in range(len(Pk))]; Pk.to_parquet(O / "p5_key.parquet"); Pk[["id", "text"]].to_csv(O / "p5_to_label.tsv", sep="\t", index=False)
print("P5", len(Pk), Pk.arm.value_counts().to_dict())
