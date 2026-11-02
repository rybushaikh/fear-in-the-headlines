"""Frozen plan: Media-Fear/PLAN-analysis-covid-2026-10-27.md. Fear before / during / after COVID by year x subject x outlet, shift-share, day-cluster bootstrap.
Output: data/processed/final/analysis_covid.json + tables printed for RESULTS-covid-2026-10-27.md"""
import json, numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1]; O = R / "data/processed/final"; rng = np.random.default_rng(0); B = 1000
D = pd.read_parquet(O / "all_scored.parquet", columns=["source", "design", "date", "year", "headline", "material", "outlet_subject", "fear", "level", "subject", "subject_tier", "generic"])
D["scary"] = D.level >= 2; D["alarm"] = D.fear >= 2.5
PER = {"pre-COVID 2017-19": (2017, 2019), "COVID 2020-21": (2020, 2021), "after 2022-23": (2022, 2023), "recent 2024-26": (2024, 2026)}
def period(y):
    for k, (a, b) in PER.items():
        if a <= y <= b: return k
D["period"] = D.year.map(period)
SAMPLES = {"NYT news": (D.source == "nyt") & (D.material == "news"), "NYT opinion": (D.source == "nyt") & (D.material == "opinion"),
           "Guardian (random days)": (D.source == "guardian") & D.design.str.contains("random_day"), "ABC": D.source == "abc"}
OUT = {}
def boot_diff(x, a, b, col):
    """day-cluster bootstrap of share(col) in period b minus period a."""
    out = {}
    for p in (a, b):
        g = x[x.period == p].groupby("date")[col].agg(["sum", "size"]).values; idx = rng.integers(0, len(g), (B, len(g)))
        s = g[:, 0][idx].sum(1) / g[:, 1][idx].sum(1); out[p] = s
    d = out[b] - out[a]; return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
pd.set_option("display.width", 250)
for nm, m in SAMPLES.items():
    x = D[m & ~D.generic]; r = {}
    yr = x[x.year >= (2003 if nm == "ABC" else 2015)].groupby("year").agg(n=("scary", "size"), scary=("scary", "mean"), alarm=("alarm", "mean"), fear=("fear", "mean")).round(4)
    r["by_year"] = yr.reset_index().to_dict("records"); print(f"\n=== {nm}: by year\n" + yr.to_string())
    if nm == "ABC": OUT[nm] = r; continue
    x = x[x.period.notna()]
    pt = x.groupby("period").agg(n=("scary", "size"), scary=("scary", "mean"), alarm=("alarm", "mean"), fear=("fear", "mean")).reindex(list(PER)).round(4); r["by_period"] = pt.reset_index().to_dict("records"); print(pt.to_string())
    a, b = "pre-COVID 2017-19", "recent 2024-26"
    r["diff_recent_vs_pre"] = {c: dict(diff=round(float(pt.loc[b, c] - pt.loc[a, c]), 4), ci95=boot_diff(x, a, b, c)) for c in ("scary", "alarm")}
    r["diff_covid_vs_pre"] = {c: dict(diff=round(float(pt.loc["COVID 2020-21", c] - pt.loc[a, c]), 4), ci95=boot_diff(x, a, "COVID 2020-21", c)) for c in ("scary", "alarm")}
    print("recent vs pre:", r["diff_recent_vs_pre"], "| covid vs pre:", r["diff_covid_vs_pre"])
    # subjects: agenda share and scary share per period
    def subj_table(xx, col):
        t = xx.groupby(["period", col]).agg(n=("scary", "size"), scary=("scary", "mean")).reset_index(); t["share"] = t.n / t.groupby("period").n.transform("sum"); return t
    for subjcol, tag in (("subject", "model"), ("outlet_subject", "outlet_tag")):
        xx = x[x[subjcol].notna()] if subjcol == "outlet_subject" else x
        if subjcol == "outlet_subject" and len(xx) == 0: continue
        t = subj_table(xx, subjcol); pv = t.pivot(index=subjcol, columns="period", values=["share", "scary"]).reindex(columns=list(PER), level=1).round(3)
        if tag == "model": print(f"--- {nm}: subject share (agenda) and scary share (framing) by period\n" + pv.to_string())
        # shift-share pre -> recent
        p0 = t[t.period == a].set_index(subjcol); p1 = t[t.period == b].set_index(subjcol); k = p0.index.union(p1.index); p0 = p0.reindex(k).fillna(0); p1 = p1.reindex(k).fillna(0)
        agenda = float(((p1.share - p0.share) * p0.scary).sum()); framing = float((p0.share * (p1.scary - p0.scary)).sum()); inter = float(((p1.share - p0.share) * (p1.scary - p0.scary)).sum())
        contrib = ((p1.share * p1.scary) - (p0.share * p0.scary)).sort_values(); ss = dict(total=round(agenda + framing + inter, 4), agenda=round(agenda, 4), framing=round(framing, 4), interaction=round(inter, 4),
                  by_subject_change_in_scary_points=contrib.round(4).to_dict())
        r[f"shift_share_{tag}"] = ss; r[f"subjects_{tag}"] = {f"{i[0]}|{i[1]}": v for i, v in pv.stack(level=1, future_stack=True).to_dict("index").items()} if False else pv.to_json()
        print(f"shift-share ({tag} subject) pre->recent:", {k2: v for k2, v in ss.items() if k2 != "by_subject_change_in_scary_points"}); print("   per-subject contribution:", ss["by_subject_change_in_scary_points"])
    # robustness
    xc = x[x.subject_tier == "confident"]; tc = subj_table(xc, "subject")
    r["robust_confident_only_scary_by_period"] = xc.groupby("period").scary.mean().reindex(list(PER)).round(4).to_dict()
    xg = D[m & D.period.notna()]; r["robust_with_generic_scary_by_period"] = xg.groupby("period").scary.mean().reindex(list(PER)).round(4).to_dict()
    print("robust: confident-only", r["robust_confident_only_scary_by_period"], "| with generic", r["robust_with_generic_scary_by_period"])
    OUT[nm] = r
# model check on the 80 random sealed-#5 items
S = pd.read_parquet(R / "data/eval/scary_test5_private/labelled.parquet"); S = S[S.pool == "random"]
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "noesis_fear")); import fear_v7
f = np.array([a["fear"] for a in fear_v7.appraise(S.text.tolist())]); OUT["model_check_random80"] = dict(n=len(S), true_scary=round(float((S.fear >= 2).mean()), 3), model_scary=round(float((np.floor(f + .5) >= 2).mean()), 3),
    true_mean=round(float(S.fear.mean()), 3), model_mean=round(float(f.mean()), 3)); print("\nmodel check (80 random NYT 2022-26, blind labels):", OUT["model_check_random80"])
json.dump(OUT, open(O / "analysis_covid.json", "w"), indent=1, default=str); print("saved")
