"""Frozen plan: Media-Fear/PLAN-transitions-2026-11-03.md (exploratory). Output: data/processed/final/analysis_transitions.json"""
import sys, re, json, numpy as np, pandas as pd
from collections import Counter
from pathlib import Path
R = Path(__file__).resolve().parents[1]; O = R / "data/processed/final"; sys.path.insert(0, str(R / "pipeline")); import u1_eval as UE
D = pd.read_parquet(O / "all_scored.parquet", columns=["source", "design", "date", "year", "headline", "material", "outlet_subject", "fear", "level", "subject", "generic"])
D = D[D.year.between(2003, 2026)]; D["scary"] = D.level >= 2; D["alarm"] = D.fear >= 2.5
N = D[(D.source == "nyt") & (D.material == "news") & ~D.generic].copy(); N["month"] = N.date.str[:7]; OUT = {}; pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 40)
y = N.groupby("year").agg(n=("scary", "size"), days=("date", "nunique"), scary=("scary", "mean"), alarm=("alarm", "mean"), sc=("scary", "sum")); y["per_day"] = y.sc / y.days; y["n_day"] = y.n / y.days
print("T1\n" + y.round(4).to_string()); OUT["T1"] = y.round(4).reset_index().to_dict("records")
m = N.groupby("month").agg(n=("scary", "size"), s=("scary", "mean")); m = m[m.n >= 300]; x = m.s.values; n = len(x); sig2 = np.var(np.diff(x)) / 2; pen = 3 * np.log(n) * sig2; MS = 6
cs, cs2 = np.r_[0, np.cumsum(x)], np.r_[0, np.cumsum(x ** 2)]; cost = lambda a, b: cs2[b] - cs2[a] - (cs[b] - cs[a]) ** 2 / (b - a)
F = np.full(n + 1, np.inf); F[0] = -pen; prev = np.zeros(n + 1, int)
for b in range(MS, n + 1):
    for a in range(0, b - MS + 1):
        if np.isfinite(F[a]) and F[a] + cost(a, b) + pen < F[b]: F[b] = F[a] + cost(a, b) + pen; prev[b] = a
cp = []; b = n
while b > 0: cp.append((prev[b], b)); b = prev[b]
cp = cp[::-1]; SEG = [(m.index[a], m.index[b - 1]) for a, b in cp]; print("months", n, "sigma", round(float(np.sqrt(sig2)), 4), "segments", len(SEG)); rows = []; subs = sorted(N.subject.unique())
for a, b in SEG:
    g = N[N.month.between(a, b)]; r = dict(start=a, end=b, scary=round(g.scary.mean(), 4), alarm=round(g.alarm.mean(), 4), per_day=round(g.scary.sum() / g.date.nunique(), 2), n_day=round(len(g) / g.date.nunique(), 1))
    for s in subs: r["sh_" + s[:7]] = round((g.subject == s).mean(), 3)
    for s in ("war_terror", "disaster_accident", "crime_justice", "health"): r["sc_" + s[:7]] = round(g.scary[g.subject == s].mean(), 3)
    rows.append(r)
T = pd.DataFrame(rows); print("T2/T3\n" + T.T.to_string()); OUT["segments"] = rows
base = Counter(w for h in N.headline.str.lower() for w in set(re.findall(r"[a-z][a-z'-]{3,}", h))); tot = len(N); top = []
for mo, r in m.sort_values("s", ascending=False).head(10).iterrows():
    g = N[N.month == mo]; sc = g[g.scary]; c = Counter(w for h in sc.headline.str.lower() for w in set(re.findall(r"[a-z][a-z'-]{3,}", h))); sco = {w: (v / len(sc)) / (base[w] / tot) * min(v, 15) for w, v in c.items() if v >= 5}
    top.append(dict(month=mo, scary=round(r.s, 4), top_subject=sc.subject.value_counts().index[0], words=sorted(sco, key=sco.get, reverse=True)[:6]))
print("T4"); [print(t) for t in top]; OUT["top_months"] = top
PER = {"2017-19": (2017, 2019), "2020-21": (2020, 2021), "2022-23": (2022, 2023), "2024-26": (2024, 2026)}; v1 = {}
for k, (a, b) in PER.items():
    g = N[N.year.between(a, b)]; d = g.date.nunique(); v1[k] = dict(n_day=round(len(g) / d, 1), scary_day=round(g.scary.sum() / d, 2), share=round(g.scary.mean(), 4), no_sport=round(g[g.subject != "sport"].scary.mean(), 4), no_sport_culture=round(g[~g.subject.isin(["sport", "culture_lifestyle"])].scary.mean(), 4), notscary_day=round((~g.scary).sum() / d, 1))
print("V1", json.dumps(v1, indent=0)); OUT["V1"] = v1
def ss(g0, g1, col):
    p0 = g0.groupby(col).scary.agg(["size", "mean"]); p1 = g1.groupby(col).scary.agg(["size", "mean"]); k = p0.index.union(p1.index); p0 = p0.reindex(k).fillna(0); p1 = p1.reindex(k).fillna(0); s0 = p0["size"] / p0["size"].sum(); s1 = p1["size"] / p1["size"].sum()
    return dict(total=round(float((s1 * p1["mean"]).sum() - (s0 * p0["mean"]).sum()), 4), agenda=round(float(((s1 - s0) * p0["mean"]).sum()), 4), framing=round(float((s0 * (p1["mean"] - p0["mean"])).sum()), 4), inter=round(float(((s1 - s0) * (p1["mean"] - p0["mean"])).sum()), 4), tag_share=pd.concat([s0, s1], axis=1).round(3).values.tolist(), tag_scary=pd.concat([p0["mean"], p1["mean"]], axis=1).round(3).values.tolist(), tags=list(k))
a = N[N.year.between(2017, 2019)]; b = N[N.year.between(2024, 2026)]; tA, tB = a[a.outlet_subject.notna()], b[b.outlet_subject.notna()]
OUT["V2"] = dict(tag_coverage=[round(len(tA) / len(a), 3), round(len(tB) / len(b), 3)], by_tag=ss(tA, tB, "outlet_subject"), by_model_same_rows=ss(tA, tB, "subject"), scary_subject_agree={k: round(float((g[g.scary].subject == g[g.scary].outlet_subject).mean()), 3) for k, g in (("pre", tA), ("recent", tB))}, calm_subject_agree={k: round(float((g[~g.scary].subject == g[~g.scary].outlet_subject).mean()), 3) for k, g in (("pre", tA), ("recent", tB))})
print("V2", json.dumps(OUT["V2"]))
sys.path.insert(0, str(R / "noesis_fear")); import fear_v7
res = {}; rows = []
for nm, d in (("sealed4_2003_21", "scary_test4_private"), ("sealed5_2022_26", "scary_test5_private")):
    p = R / "data/eval" / d; K = pd.read_parquet(p / "pool_key.parquet")
    if (p / "labelled.parquet").exists(): L = pd.read_parquet(p / "labelled.parquet")[["id", "fear"]]
    else:
        fs = list((p / "claude_u1").glob("*")); rec = []
        for f in fs:
            if f.suffix == ".jsonl": rec += [json.loads(l) for l in open(f) if l.strip()]
        L = pd.DataFrame(rec)[["id", "fear"]].drop_duplicates("id", keep="last")
    K = K.merge(L, on="id"); K = K[K.pool == "random"]; K["era"] = nm; rows.append(K[["text", "fear", "era", "date"]])
A = pd.concat(rows, ignore_index=True); f = np.array([a["fear"] for a in fear_v7.appraise(A.text.tolist())]); A["lv"] = np.clip(np.floor(f + .5), 0, 3).astype(int); A["y"] = A.fear.astype(int); rng = np.random.default_rng(0)
def met(g):
    hi, p = g.y >= 2, g.lv >= 2; return dict(n=len(g), kappa=round(UE.qwk(g.lv.values, g.y.values), 3), exact=round(float((g.lv == g.y).mean()), 3), true_scary=round(float(hi.mean()), 3), model_scary=round(float(p.mean()), 3), n_true_scary=int(hi.sum()), recall=round(float((p & hi).sum() / max(1, hi.sum())), 3), precision=round(float((p & hi).sum() / max(1, p.sum())), 3))
for k, g in list(A.groupby("era")) + [("all", A)]:
    r = met(g); bs = [UE.qwk(*(lambda s: (s.lv.values, s.y.values))(g.iloc[rng.integers(0, len(g), len(g))])) for _ in range(1000)]; r["kappa_ci"] = [round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]; res[k] = r
print("V3/V4", json.dumps(res, indent=0)); OUT["V3"] = res
G = D[(D.source == "guardian") & D.design.str.contains("random_day") & ~D.generic]; gy = G.groupby("year").scary.mean().round(4); print("Guardian yearly", gy.to_dict()); OUT["guardian_year"] = gy.to_dict()
json.dump(OUT, open(O / "analysis_transitions.json", "w"), indent=1, default=str); print("saved")
