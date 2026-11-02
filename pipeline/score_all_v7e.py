"""Score every collected headline with the live Noesis model (fear_v7 = v7e + subject head), 27 Oct 2026.
One row per article: source, design (how it was sampled), date, year, headline, material (news/opinion/other), outlet_subject (tag rules; NYT/Guardian only),
fear (0-3, remapped; >= 2.5 counts as 3), level, subject, subject_conf, subject_tier, generic (headline repeated >= 5 times in its source).
Dedup: NYT by id; Guardian by id (random-day 2003-21 has no id: date+headline); ABC by date+headline. Identical headline text is encoded once.
Output: data/processed/final/all_scored.parquet + log data/processed/final/score_all.log"""
import sys, json, gzip, glob, time, numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1]; A = R / "data/processed/abc_all"; O = R / "data/processed/final"; O.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "noesis_fear")); sys.path.insert(0, str(R / "pipeline"))
import fear_v7, build_subject_rules as BR
LOG = open(O / "score_all.log", "a")
def log(s): s = time.strftime("%H:%M:%S ") + s; print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
T0 = time.time(); parts = []
a = pd.read_csv(R / "data/raw/abc_million_headlines/abcnews-date-text.csv"); d = pd.to_datetime(a.publish_date.astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
parts.append(pd.DataFrame(dict(source="abc", design="full", id=None, date=d, headline=a.headline_text, material="news", outlet_subject=None)).drop_duplicates(["date", "headline"])); log(f"abc {len(parts[-1])}")
rows = []
for f in sorted(glob.glob(str(R / "data/raw/outlets/nyt/20*.json.gz"))):
    for x in json.load(gzip.open(f))["response"]["docs"]:
        if (x.get("document_type") or "").lower() != "article" or x.get("type_of_material") in ("Paid Death Notice", "Correction"): continue
        h = ((x.get("headline") or {}).get("main") or "").strip()
        if not h: continue
        tom = x.get("type_of_material") or ""; mat = "news" if tom == "News" else ("opinion" if tom in ("Op-Ed", "Editorial", "Letter") else "other")
        rows.append(dict(source="nyt", design="full", id=x.get("_id") or x.get("uri"), date=(x.get("pub_date") or "")[:10], headline=h, material=mat,
                         outlet_subject=BR.label(x.get("section_name"), BR.nyt_kw(x))))
parts.append(pd.DataFrame(rows).drop_duplicates("id")); del rows; log(f"nyt {len(parts[-1])}")
g0 = pd.read_parquet(A / "guardian_alltopics.parquet"); g0 = pd.DataFrame(dict(source="guardian", design="random_day", id=None, date=g0.date, headline=g0.headline, section=g0.section, keywords="", tone=""))
g1 = pd.read_parquet(A / "guardian_targeted.parquet"); g1 = pd.DataFrame(dict(source="guardian", design="tag:" + g1.query_tag, id=g1.id, date=g1.date, headline=g1.headline, section=g1.section, keywords=g1.keywords.fillna("") + ";" + g1.query_tag, tone=g1.tone))
g2 = pd.read_parquet(A / "recent_guardian.parquet"); g2 = pd.DataFrame(dict(source="guardian", design=np.where(g2.source == "random_day", "random_day", "tag:" + g2.source), id=g2.id, date=g2.date, headline=g2.headline, section=g2.section, keywords=g2.keywords, tone=g2.tone))
G = pd.concat([g0, g1, g2], ignore_index=True); G = G[G.headline.notna()].copy()
G["key"] = G.id.fillna(G.date + "|" + G.headline); G["design"] = G.groupby("key").design.transform(lambda s: ";".join(sorted(set(s)))); G = G.drop_duplicates("key")
G["material"] = np.where(G.tone.fillna("").str.contains("comment|editorial|letters"), "opinion", "news")
G["outlet_subject"] = [BR.label(s, k) for s, k in zip(G.section, G.keywords.fillna(""))]
parts.append(G[["source", "design", "id", "date", "headline", "material", "outlet_subject"]]); log(f"guardian {len(G)}")
D = pd.concat(parts, ignore_index=True); D["headline"] = D.headline.astype(str).str.strip(); D = D[D.headline.str.len() > 0].reset_index(drop=True)
D["year"] = D.date.str[:4].astype(int); D["generic"] = D.groupby(["source", D.headline.str.lower()]).headline.transform("size") >= 5
U = pd.Series(D.headline.unique()); log(f"rows {len(D):,}; unique headline texts {len(U):,}")
o = np.argsort(U.str.len().values, kind="stable"); m = fear_v7._load()
F = np.zeros(len(U)); SI = np.zeros(len(U), int); SC = np.zeros(len(U)); CS = 100_000
for b in range(0, len(U), CS):
    ix = o[b:b + CS]; E = m["enc"].encode(U.values[ix].tolist(), batch_size=512, normalize_embeddings=True, show_progress_bar=False)
    F[ix] = fear_v7._remap((E @ m["fw"] + m["fb"]).clip(0, 3)); z = E @ m["sw"].T + m["sb"]; p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    SI[ix] = p.argmax(1); SC[ix] = p.max(1); log(f"  encoded {min(b + CS, len(U)):,}/{len(U):,} {time.time() - T0:.0f}s")
mp = pd.Series(np.arange(len(U)), index=U.values); j = mp.loc[D.headline.values].values
D["fear"] = F[j].round(3); D["level"] = np.clip(np.floor(D.fear + .5), 0, 3).astype(int); D["subject"] = np.array(m["scls"])[SI[j]]; D["subject_conf"] = SC[j].round(3)
D["subject_tier"] = np.where(D.subject_conf >= .6, "confident", "unsure")
D.to_parquet(O / "all_scored.parquet"); log(f"saved {len(D):,} rows")
log("rows by source [first year, last year, n]: " + json.dumps({s: [int(g.year.min()), int(g.year.max()), len(g)] for s, g in D.groupby("source")}))
log("mean fear by source: " + json.dumps(D.groupby("source").fear.mean().round(3).to_dict()))
log(f"done {time.time() - T0:.0f}s")
