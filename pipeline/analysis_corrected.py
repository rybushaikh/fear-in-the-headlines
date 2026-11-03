"""Frozen plan: docs/plans/data-audit.md (exploratory, post hoc). Rebuilds the NYT news base after the data audit and reruns the headline results.
Base: type News, not generic, www.nytimes.com only (no blogs or DealBook), same-day repeats removed, no Sports/Games/Learning/Society (sections that closed or are not news).
Output: data/processed/final/analysis_corrected.json"""
import json, gzip, glob, sys, numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1]; O = R / "data/processed/final"; rng = np.random.default_rng(0); B = 1000; OUT = {}
pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 40)
rows = []
for f in sorted(glob.glob(str(R / "data/raw/outlets/nyt/20*.json.gz"))):
    for d in json.load(gzip.open(f))["response"]["docs"]:
        rows.append(dict(id=d.get("_id") or d.get("uri"), url=d.get("web_url") or "", sec=d.get("section_name") or "", desk=d.get("news_desk") or "", pp=bool(d.get("print_page")), ph=(d.get("headline") or {}).get("print_headline") or ""))
M = pd.DataFrame(rows).drop_duplicates("id"); D = pd.read_parquet(O / "all_scored.parquet", columns=["source", "id", "date", "year", "headline", "material", "outlet_subject", "fear", "level", "subject", "generic"])
A = D[(D.source == "nyt") & (D.material == "news") & ~D.generic].merge(M, on="id", how="left"); A["scary"] = A.level >= 2; A["alarm"] = A.fear >= 2.5; A["wd"] = A.subject.isin(["war_terror", "disaster_accident"]); A["month"] = A.date.str[:7]
A["www"] = A.url.str.contains(r"^https?://www\.nytimes\.com", regex=True); A["dup"] = A.assign(k=A.headline.str.lower()).duplicated(["date", "k"])
A["closed"] = A.sec.isin(["Sports", "The Learning Network", "Crosswords & Games"]) | A.desk.isin(["Sports", "Games", "Learning", "Society"])
W = A[A.www]; V = W[~W.dup]; C = V[~V.closed]; C.drop(columns=["ph"]).to_parquet(O / "nyt_news_corrected.parquet")
PRE, REC = (2017, 2019), (2024, 2026)
def boot(x, col="scary"):
    d = []
    for p in (PRE, REC):
        g = x[x.year.between(*p)].groupby("date")[col].agg(["sum", "size"]).values; i = rng.integers(0, len(g), (B, len(g))); d.append(g[:, 0][i].sum(1) / g[:, 1][i].sum(1))
    return [round(float(np.percentile(d[1] - d[0], q)) * 100, 2) for q in (2.5, 97.5)]
def res(x):
    a, b = x[x.year.between(*PRE)], x[x.year.between(*REC)]; da, db = a.date.nunique(), b.date.nunique()
    return dict(n_pre=len(a), n_rec=len(b), scary_pre=round(a.scary.mean() * 100, 2), scary_rec=round(b.scary.mean() * 100, 2), rise=round((b.scary.mean() - a.scary.mean()) * 100, 2), ci=boot(x), wd_pre=round(a.wd.mean() * 100, 2), wd_rec=round(b.wd.mean() * 100, 2),
                scary_day_pre=round(a.scary.sum() / da, 2), scary_day_rec=round(b.scary.sum() / db, 2), calm_day_pre=round((~a.scary).sum() / da, 1), calm_day_rec=round((~b.scary).sum() / db, 1), wd_of_scary_pre=round(a.wd[a.scary].mean() * 100, 1), wd_of_scary_rec=round(b.wd[b.scary].mean() * 100, 1))
VAR = {"original": A, "www only": W, "www, no same-day repeats": V, "CORRECTED": C, "corrected, print only": C[C.pp], "corrected, no Express or Weather": C[~C.desk.isin(["Express", "Weather"])], "corrected, Foreign+National+Metro+Express+Weather": C[C.desk.isin(["Foreign", "National", "Metro", "Express", "Weather"])], "corrected, no politics desks": C[~C.desk.isin(["Washington", "Politics"])]}
OUT["variants"] = {k: res(x) for k, x in VAR.items()}; print(pd.DataFrame(OUT["variants"]).T.to_string())
y = C.groupby("year").agg(n=("scary", "size"), days=("date", "nunique"), scary=("scary", "mean"), wd=("wd", "mean"), alarm=("alarm", "mean")); y = y.assign(orig=A.groupby("year").scary.mean(), n_day=y.n / y.days); print(y.round(3).T.to_string()); OUT["by_year"] = y.round(4).reset_index().to_dict("records")
t = C[C.outlet_subject.notna()]; OUT["outlet_tags"] = {k: round(t[t.year.between(*p)].outlet_subject.isin(["war_terror", "disaster_accident"]).mean() * 100, 1) for k, p in (("pre", PRE), ("rec", REC))}; print("tags", OUT["outlet_tags"])
pa, pb = C[C.year.between(*PRE)], C[C.year.between(*REC)]
def ss(g0, g1):
    s0, s1 = g0.subject.value_counts(normalize=True), g1.subject.value_counts(normalize=True); r0, r1 = g0.groupby("subject").scary.mean(), g1.groupby("subject").scary.mean(); k = s0.index.union(s1.index); s0, s1, r0, r1 = [v.reindex(k).fillna(0) for v in (s0, s1, r0, r1)]
    return dict(agenda=round(float(((s1 - s0) * r0).sum()) * 100, 2), framing=round(float((s0 * (r1 - r0)).sum()) * 100, 2), interaction=round(float(((s1 - s0) * (r1 - r0)).sum()) * 100, 2))
OUT["shift_share_2017"] = ss(pa, pb); OUT["shift_share_2003"] = ss(C[C.year.between(2003, 2006)], pb); print(OUT["shift_share_2017"], OUT["shift_share_2003"])
st = pd.DataFrame({k: g.groupby("subject").agg(share=("scary", lambda s: len(s) / len(g)), scary=("scary", "mean")).stack() for k, g in (("pre", pa), ("rec", pb), ("covid", C[C.year.between(2020, 2021)]))}).unstack().round(3); print(st.to_string()); OUT["subjects"] = st.to_json()
LONG = {"2003-06": (2003, 2006), "2007-09": (2007, 2009), "2017-19": PRE, "2024-26": REC}; OUT["long"] = {k: dict(scary=round(C[C.year.between(*p)].scary.mean() * 100, 2), wd=round(C[C.year.between(*p)].wd.mean() * 100, 2), scary_day=round(C[C.year.between(*p)].scary.sum() / C[C.year.between(*p)].date.nunique(), 2), n_day=round(len(C[C.year.between(*p)]) / C[C.year.between(*p)].date.nunique(), 1)) for k, p in LONG.items()}; print(OUT["long"])
C2 = C.assign(w=C.headline.str.split().str.len()); C2["lb"] = pd.cut(C2.w, [0, 6, 8, 10, 12, 99]); ref = C2[C2.year.between(*PRE)].lb.value_counts(normalize=True)
OUT["length_fixed"] = {k: round(float((C2[C2.year.between(*p)].groupby("lb", observed=True).scary.mean() * ref).sum()) * 100, 2) for k, p in LONG.items()}; OUT["mean_words"] = {k: round(float(C2[C2.year.between(*p)].w.mean()), 1) for k, p in LONG.items()}; print("length fixed", OUT["length_fixed"], OUT["mean_words"])
sd = pb.groupby("date").scary.sum(); OUT["daily_recent"] = dict(days=int(len(sd)), days_with=round(float((sd > 0).mean()) * 100, 1), median=float(sd.median()), mean=round(float(sd.mean()), 2)); print(OUT["daily_recent"])
def segments(N):
    m = N.groupby("month").agg(n=("scary", "size"), s=("scary", "mean")); m = m[m.n >= 300]; x = m.s.values; n = len(x); sig2 = np.var(np.diff(x)) / 2; pen = 3 * np.log(n) * sig2; MS = 6
    cs, cs2 = np.r_[0, np.cumsum(x)], np.r_[0, np.cumsum(x ** 2)]; cost = lambda a, b: cs2[b] - cs2[a] - (cs[b] - cs[a]) ** 2 / (b - a); F = np.full(n + 1, np.inf); F[0] = -pen; prev = np.zeros(n + 1, int)
    for b in range(MS, n + 1):
        for a in range(0, b - MS + 1):
            if np.isfinite(F[a]) and F[a] + cost(a, b) + pen < F[b]: F[b] = F[a] + cost(a, b) + pen; prev[b] = a
    cp = []; b = n
    while b > 0: cp.append((prev[b], b)); b = prev[b]
    out = []
    for a, b in cp[::-1]:
        g = N[N.month.between(m.index[a], m.index[b - 1])]; out.append(dict(start=m.index[a], end=m.index[b - 1], scary=round(g.scary.mean() * 100, 1), wd=round(g.wd.mean() * 100, 1), scary_day=round(g.scary.sum() / g.date.nunique(), 1), months=b - a))
    return out, m
for nm, x in (("corrected", C), ("www, no repeats", V), ("corrected, print only", C[C.pp])):
    sg, mm = segments(x); OUT["segments_" + nm] = sg; print("\nsegments", nm, pd.DataFrame(sg).to_string(index=False)); print("top months", mm[mm.n >= 300].s.sort_values(ascending=False).head(4).round(3).to_dict())
    if nm == "corrected": OUT["monthly"] = {k: round(float(v) * 100, 2) for k, v in mm.s.items()}; OUT["top_months"] = {k: round(float(v) * 100, 1) for k, v in mm.s.sort_values(ascending=False).head(5).items()}
sys.path.insert(0, str(R / "noesis_fear")); import fear_v7; OUT["print_headlines"] = {}
P = A[A.www & ~A.dup & ~A.closed & (A.ph != "")]
for k, p in (("pre", PRE), ("rec", REC)):
    x = P[P.year.between(*p)]; x = x.sample(min(25000, len(x)), random_state=0); f = np.array([r["fear"] for r in fear_v7.appraise(x.ph.tolist())]); lv = np.floor(f + .5); assert (np.floor(x.fear + .5) == x.level).mean() > .99
    OUT["print_headlines"][k] = dict(n=len(x), web_scary=round(float(x.scary.mean()) * 100, 2), print_scary=round(float((lv >= 2).mean()) * 100, 2))
PER = {"2017-19": PRE, "2020-21": (2020, 2021), "2022-23": (2022, 2023), "2024-26": REC}; OUT["periods"] = {k: dict(scary=round(C[C.year.between(*p)].scary.mean() * 100, 2), alarm=round(C[C.year.between(*p)].alarm.mean() * 100, 2), wd=round(C[C.year.between(*p)].wd.mean() * 100, 2)) for k, p in PER.items()}; OUT["alarm_ci"] = boot(C, "alarm"); print(OUT["periods"], OUT["alarm_ci"])
st4 = {k: C[C.year.between(*p)].groupby("subject").agg(share=("scary", lambda s: round(len(s) / C.year.between(*p).sum() * 100, 1)), scary=("scary", lambda s: round(s.mean() * 100))).to_dict("index") for k, p in PER.items()}; OUT["subjects4"] = st4; print(pd.DataFrame({k: {s: f"{v['share']} ({v['scary']})" for s, v in d.items()} for k, d in st4.items()}).to_string())
O2 = D[(D.source == "nyt") & (D.material == "opinion") & ~D.generic].merge(M, on="id", how="left"); O2 = O2[O2.url.str.contains(r"^https?://www\.nytimes\.com", regex=True)]; O2["scary"] = O2.level >= 2; OUT["opinion"] = {k: round(O2[O2.year.between(*p)].scary.mean() * 100, 2) for k, p in PER.items()}; print("opinion www", OUT["opinion"], boot(O2))
G = pd.read_parquet(O / "all_scored.parquet", columns=["source", "design", "year", "date", "headline", "level", "generic"]); G = G[(G.source == "guardian") & G.design.str.contains("random_day") & ~G.generic].copy(); G["scary"] = G.level >= 2; G["w"] = G.headline.str.split().str.len(); G["lb"] = pd.cut(G.w, [0, 6, 8, 10, 12, 99]); gr = G[G.year.between(*PRE)].lb.value_counts(normalize=True)
OUT["guardian"] = {k: dict(scary=round(G[G.year.between(*p)].scary.mean() * 100, 2), words=round(G[G.year.between(*p)].w.mean(), 1), length_fixed=round(float((G[G.year.between(*p)].groupby("lb", observed=True).scary.mean() * gr).sum()) * 100, 2)) for k, p in LONG.items()}; print("guardian", OUT["guardian"])
GS = pd.concat([pd.read_parquet(R / f"data/processed/abc_all/{f}.parquet", columns=["date", "headline", "section"]) for f in ("guardian_alltopics", "recent_guardian")]).drop_duplicates(["date", "headline"]); G2 = G.merge(GS, on=["date", "headline"], how="left"); G2["sport"] = G2.section.fillna("").str.lower().str.contains("sport|football"); OUT["guardian_sections"] = dict(match=round(float(G2.section.notna().mean()), 3), sport={k: round(float(G2[G2.year.between(*p)].sport.mean()) * 100, 1) for k, p in LONG.items()}, no_sport={k: round(float(G2[G2.year.between(*p) & ~G2.sport].scary.mean()) * 100, 2) for k, p in LONG.items()}); print(OUT["guardian_sections"])
B2 = pd.read_parquet(O / "all_scored.parquet", columns=["source", "year", "headline", "level", "generic"]); B2 = B2[(B2.source == "abc") & ~B2.generic]; B2 = B2.assign(scary=B2.level >= 2, w=B2.headline.str.split().str.len()); B2["lb"] = pd.cut(B2.w, [0, 6, 8, 10, 12, 99]); br = B2[B2.year.between(2017, 2019)].lb.value_counts(normalize=True); OUT["abc"] = {k: dict(scary=round(B2[B2.year.between(*p)].scary.mean() * 100, 2), words=round(B2[B2.year.between(*p)].w.mean(), 1), length_fixed=round(float((B2[B2.year.between(*p)].groupby("lb", observed=True).scary.mean() * br).sum()) * 100, 2)) for k, p in (("2003-06", (2003, 2006)), ("2017-19", PRE), ("2021", (2021, 2021)))}; print("abc", OUT["abc"])
A2 = A.assign(k=A.headline.str.lower()); ex = {}
for t in [p.name for p in sorted((R / "data/eval").glob("scary_test*_private")) if (p / "labelled.parquet").exists()]:
    L = pd.read_parquet(R / f"data/eval/{t}/labelled.parquet"); L = L.assign(k=L.text.str.lower()).merge(A2.drop_duplicates("k")[["k", "www", "closed"]], on="k", how="left"); ex[t] = dict(items=len(L), matched_news=int(L.www.notna().sum()), blog=int((L.www == False).sum()), closed_section=int((L.closed == True).sum()))
OUT["sealed_overlap"] = ex; print(ex); OUT["counts"] = dict(original=len(A), corrected=len(C), blog_other_domains=int((~A.www).sum()), repeats=int((A.www & A.dup).sum()), closed=int(V.closed.sum()), dup_2006=round(float(A[A.year == 2006].dup.mean()) * 100, 1), blog_2008=round(float((~A[A.year == 2008].www).mean()) * 100, 1), blog_2012=round(float((~A[A.year == 2012].www).mean()) * 100, 1)); print(OUT["counts"])
print("print headlines", OUT["print_headlines"]); json.dump(OUT, open(O / "analysis_corrected.json", "w"), indent=1, default=str); print("saved")
