"""Plan amendment 1: V5 stratified label-based share, V6 long-run decomposition. Output: data/processed/final/analysis_transitions2.json"""
import sys, re, json, glob, numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1]; O = R / "data/processed/final"; E = R / "data/eval"; sys.path.insert(0, str(R / "noesis_fear")); import fear_v7
src = open(R / "pipeline/prep_scary_test.py").read(); CUE = re.compile(re.search(r'CUE = re.compile\(r"(.+?)", re.I\)', src).group(1), re.I)
P = []
for k in ("", "2", "3", "4", "5"):
    d = E / f"scary_test{k}_private"; K = pd.read_parquet(d / "pool_key.parquet"); recs = []
    if k == "5": L = pd.read_parquet(d / "labelled.parquet")[["id", "fear"]]
    else:
        for f in sorted(glob.glob(str(d / "claude_u1/*"))):
            if f.endswith(".jsonl"): recs += [json.loads(l) for l in open(f) if l.strip()]
        L = pd.DataFrame(recs)[["id", "fear"]].drop_duplicates("id", keep="last")
    print("pool", k or "1", "items", len(K), "labels", len(L), "pools", K.pool.value_counts().to_dict(), "cols", [c for c in K.columns if c in ("date", "text", "pool")])
    K = K.merge(L, on="id"); K["src"] = k or "1"; P.append(K[["text", "date", "pool", "fear", "src"]])
P = pd.concat(P, ignore_index=True); P["y"] = pd.to_numeric(P.fear, errors="coerce"); P = P[P.y.notna()]; P["scary"] = P.y >= 2; P["cue"] = P.pool == "cue"; P["year"] = P.date.str[:4].astype(int)
f = np.array([a["fear"] for a in fear_v7.appraise(P.text.tolist())]); P["m"] = np.floor(f + .5) >= 2
print("cue flag vs regex agreement", round(float((P.text.str.contains(CUE) == P.cue).mean()), 3))
D = pd.read_parquet(O / "all_scored.parquet", columns=["source", "year", "date", "headline", "material", "level", "subject", "fear", "generic"]); N = D[(D.source == "nyt") & ~D.generic & (D.headline.str.split().str.len() >= 4)].copy(); N["cue"] = N.headline.str.contains(CUE); N["scary"] = N.level >= 2
rng = np.random.default_rng(0); OUT = {}; PER = {"2017-19": (2017, 2019), "2024-26": (2024, 2026), "2003-21 all": (2003, 2021), "2022-26 all": (2022, 2026)}; B = {}
for k, (a, b) in PER.items():
    g = P[P.year.between(a, b)]; c, n_ = g[g.cue], g[~g.cue]; w = float(N[N.year.between(a, b)].cue.mean()); pop = float(N[N.year.between(a, b)].scary.mean())
    def est(col, c=c, n_=n_): return w * c[col].mean() + (1 - w) * n_[col].mean()
    bt = np.array([[est(col, c.iloc[rng.integers(0, len(c), len(c))], n_.iloc[rng.integers(0, len(n_), len(n_))]) for col in ("scary", "m")] for _ in range(1000)]); B[k] = bt
    OUT[k] = dict(n_cue=len(c), n_noncue=len(n_), cue_share=round(w, 4), cue_scary=round(float(c.scary.mean()), 3), noncue_scary=round(float(n_.scary.mean()), 3), n_noncue_scary=int(n_.scary.sum()), true_share=round(est("scary"), 4), true_ci=np.percentile(bt[:, 0], [2.5, 97.5]).round(4).tolist(), model_on_items=round(est("m"), 4), model_population=round(pop, 4))
d = B["2024-26"] - B["2017-19"]; OUT["rise"] = dict(true=round(OUT["2024-26"]["true_share"] - OUT["2017-19"]["true_share"], 4), true_ci=np.percentile(d[:, 0], [2.5, 97.5]).round(4).tolist(), p_true_rise_gt0=round(float((d[:, 0] > 0).mean()), 3), model_on_items=round(OUT["2024-26"]["model_on_items"] - OUT["2017-19"]["model_on_items"], 4), model_population=round(OUT["2024-26"]["model_population"] - OUT["2017-19"]["model_population"], 4))
U = P[P.src.isin(["4", "5"])]; w_ = {}
for k, g, per in (("sealed4", U[U.src == "4"], (2003, 2021)), ("sealed5", U[U.src == "5"], (2022, 2026))):
    wc = float(N[N.year.between(*per)].cue.mean()); wt = np.where(g.cue, wc / g.cue.sum(), (1 - wc) / (~g.cue).sum()); tp = wt[g.scary & g.m].sum(); w_[k] = dict(n=len(g), weighted_recall=round(float(tp / wt[g.scary].sum()), 3), weighted_precision=round(float(tp / wt[g.m].sum()), 3), true_share=round(float(wt[g.scary].sum()), 4), model_share=round(float(wt[g.m].sum()), 4))
OUT["unseen_weighted"] = w_; print(json.dumps(OUT, indent=1))
X = N[N.material == "news"]
def ss(a, b):
    g0, g1 = X[X.year.between(*a)], X[X.year.between(*b)]; p0 = g0.groupby("subject").scary.agg(["size", "mean"]); p1 = g1.groupby("subject").scary.agg(["size", "mean"]); s0, s1 = p0["size"] / p0["size"].sum(), p1["size"] / p1["size"].sum()
    return dict(a=round(float(g0.scary.mean()), 4), b=round(float(g1.scary.mean()), 4), agenda=round(float(((s1 - s0) * p0["mean"]).sum()), 4), framing=round(float((s0 * (p1["mean"] - p0["mean"])).sum()), 4), inter=round(float(((s1 - s0) * (p1["mean"] - p0["mean"])).sum()), 4), scary_in_war_dis=[round(float(p0["mean"]["war_terror"]), 3), round(float(p1["mean"]["war_terror"]), 3), round(float(p0["mean"]["disaster_accident"]), 3), round(float(p1["mean"]["disaster_accident"]), 3)])
def split(a, b):
    g0, g1 = X[X.year.between(*a)], X[X.year.between(*b)]; s0, c0, s1, c1 = g0.scary.sum() / g0.date.nunique(), (~g0.scary).sum() / g0.date.nunique(), g1.scary.sum() / g1.date.nunique(), (~g1.scary).sum() / g1.date.nunique(); t = s1 / (s1 + c1) - s0 / (s0 + c0)
    return dict(scary_day=[round(s0, 2), round(s1, 2)], calm_day=[round(c0, 1), round(c1, 1)], total_change=round(float(t), 4), if_only_scary_changed=round(float(s1 / (s1 + c0) - s0 / (s0 + c0)), 4), if_only_calm_changed=round(float(s0 / (s0 + c1) - s0 / (s0 + c0)), 4))
V6 = {"shift 2003-06 to 2024-26": ss((2003, 2006), (2024, 2026)), "shift 2011-14 to 2024-26": ss((2011, 2014), (2024, 2026)), "split 2017-19 to 2024-26": split((2017, 2019), (2024, 2026)), "split 2007-09 to 2024-26": split((2007, 2009), (2024, 2026))}; print(json.dumps(V6, indent=1)); OUT["V6"] = V6
json.dump(OUT, open(O / "analysis_transitions2.json", "w"), indent=1, default=str)
