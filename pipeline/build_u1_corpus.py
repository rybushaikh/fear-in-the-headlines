"""U1 corpus (PLAN-v11 Amendment 5a): every Claude/user U1 label -> data/processed/events/v12/u1_corpus.parquet (text, fear, source, pool, split).
Dedupe on lowercase text (latest source in SRC_ORDER wins; conflicts logged). Split 80/20 stratified by source x fear, seed 1200. Never opens scary_test4_private."""
import json, glob, numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1]; E = R / "data/eval"; OUT = R / "data/processed/events/v12"; OUT.mkdir(parents=True, exist_ok=True)
MY = Path(__file__).resolve().parents[1] / "data/private/my-ratings/MY_100_final.csv"
def labs(files):
    L = pd.concat([pd.read_json(f, lines=True) for f in sorted(files)]); n = len(L); L = L.drop_duplicates("id", keep="last"); return L[["id", "fear"]], n - len(L)
parts, dups = [], {}
for nm, d in (("scary1", "scary_test_private"), ("scary2", "scary_test2_private"), ("scary3", "scary_test3_private")):
    K = pd.read_parquet(E / d / "pool_key.parquet"); L, dp = labs(glob.glob(str(E / d / "claude_u1/labels_*.jsonl"))); dups[nm] = dp
    parts.append(K.merge(L, on="id")[["text", "fear", "pool"]].assign(source=nm))
for nm, d, pl in (("twos_pack", "twos_pack_private", "pack"), ("boundaryB", "boundaryB_private", "boundary")):
    K = pd.read_parquet(E / d / "pool_key.parquet"); L, dp = labs(glob.glob(str(E / d / "claude_u1/labels_*.jsonl"))); dups[nm] = dp
    parts.append(K.merge(L, on="id")[["text", "fear"]].assign(pool=pl, source=nm))
K = pd.read_parquet(E / "outlets_private/key.parquet"); L, dp = labs([E / "outlets/claude_u1/labels_final.jsonl"]); dups["outlets"] = dp
parts.append(K.merge(L, on="id")[["text", "fear"]].assign(pool="outlet", source="outlets"))
my = pd.read_csv(MY); parts.append(pd.DataFrame(dict(text=my.headline, fear=my.fear, pool="user", source="user100")))
SRC_ORDER = ["scary1", "scary2", "scary3", "twos_pack", "boundaryB", "outlets", "user100"]
A = pd.concat(parts, ignore_index=True); A["fear"] = A.fear.astype(int); A["text"] = A.text.astype(str).str.strip(); A["key"] = A.text.str.lower(); A["o"] = A.source.map(SRC_ORDER.index)
print("raw rows per source", A.source.value_counts().reindex(SRC_ORDER).to_dict(), "| duplicate label lines dropped (kept last)", dups)
A = A.sort_values("o", kind="stable"); g = A.groupby("key"); nsrc = g.source.nunique(); nfear = g.fear.nunique()
conf = nfear[nfear > 1].index; print(f"texts in >1 source {int((nsrc > 1).sum())}; within-text label conflicts {len(conf)}; same-source duplicates {int(len(A) - A.key.nunique() - (g.size() > 0).sum() + A.key.nunique())}")
if len(conf):
    c = A[A.key.isin(conf)].groupby("key").apply(lambda x: " -> ".join(f"{s}:{f}" for s, f in zip(x.source, x.fear))); print("conflicts (older -> latest kept):"); print(c.head(40).to_string()); print("final resolution counts by source kept:", A[A.key.isin(conf)].drop_duplicates("key", keep="last").source.value_counts().to_dict())
C = A.drop_duplicates("key", keep="last").drop(columns=["key", "o"]).reset_index(drop=True)
rng = np.random.default_rng(1200); C["split"] = "train"; C["g"] = C.source + "|" + C.fear.astype(str)
for _, ix in C.groupby("g").groups.items():
    ix = np.array(list(ix)); n_dev = int(round(0.2 * len(ix)));
    if len(ix) >= 2: C.loc[rng.choice(ix, n_dev, replace=False), "split"] = "dev"
C = C.drop(columns="g"); C.to_parquet(OUT / "u1_corpus.parquet", index=False)
print("unique items", len(C), "| train", (C.split == "train").sum(), "dev", (C.split == "dev").sum()); print(pd.crosstab(C.source, [C.split, C.fear]).to_string())
tr = ((C.pool == "cue") & (C.fear <= 1)); print("tricky-calm (cue & fear<=1):", int(tr.sum()), "dev", int((tr & (C.split == "dev")).sum())); print("pools", C.pool.value_counts().to_dict())
