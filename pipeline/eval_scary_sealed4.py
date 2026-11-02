"""ONE-TIME test on fresh sealed scary set #4 (300 NYT, U1 by Claude: 0=108, 1=42, 2=75, 3=75). PLAN-v11 amendment 5 (e). Frozen 26 Oct 2026.
Models: v6, v7c (installed models/fear_v7, cuts in fear_v7.py), v7e (data/processed/events/v12/student_v7e, its heads.json cuts). Levels = raw ridge head digitized by each model's own cuts.
Decision: v7e replaces v7c only if it beats v7c on BOTH quadratic kappa and true-2 exact. Refuses to run twice."""
import json, sys, time, numpy as np, pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics import cohen_kappa_score, roc_auc_score
R = Path(__file__).resolve().parents[1]; OUT = R / "data/processed/events/v12"; MARK = OUT / "scary_sealed4.DONE"
if MARK.exists(): sys.exit("REFUSING: sealed #4 already opened " + MARK.read_text())
C = Path(__file__).resolve().parents[1] / "noesis_fear"; sys.path.insert(0, str(C)); import fear_v6, fear_v7
M = {"v7c": (C / "models/fear_v7", fear_v7.CUTS), "v7e": (OUT / "student_v7e", tuple(json.load(open(OUT / "student_v7e/heads.json"))["cuts"]))}
MARK.write_text(time.strftime("opened %Y-%m-%d %H:%M"))
T = pd.read_parquet(R / "data/eval/scary_test4_private/SEALED_scary4_300.parquet"); t = T.text.tolist(); y = T.fear.values; tc = ((T.pool == "cue") & (y <= 1)).values
L = {"v6": (np.array([d["fear"] for d in fear_v6.appraise(t)]), None)}
for k, (d, cuts) in M.items():
    h = json.load(open(d / "heads.json")); f = (SentenceTransformer(str(d / "encoder"), device="cpu").encode(t, normalize_embeddings=True) @ np.array(h["fear_coef"]) + h["fear_intercept"]).clip(0, 3)
    L[k] = (f, np.digitize(f, cuts))
res = {}
for k, (f, lv) in L.items():
    lv = np.clip(np.round(f), 0, 3).astype(int) if lv is None else lv; T[k] = lv
    res[k] = dict(kappa=round(cohen_kappa_score(y, lv, weights="quadratic"), 3), exact=round(float((lv == y).mean()), 3), **{f"exact_{i}": round(float((lv[y == i] == i).mean()), 3) for i in range(4)},
                  auc=round(roc_auc_score(y >= 2, f), 3), rec2p=round(((lv >= 2) & (y >= 2)).sum() / (y >= 2).sum(), 3), prec2p=round(((lv >= 2) & (y >= 2)).sum() / max(1, (lv >= 2).sum()), 3),
                  rec3=round(((lv == 3) & (y == 3)).sum() / (y == 3).sum(), 3), fa=round(float((lv[tc] >= 2).mean()), 3))
D = pd.DataFrame(res).T; print(D.to_string())
win = bool(res["v7e"]["kappa"] > res["v7c"]["kappa"] and res["v7e"]["exact_2"] > res["v7c"]["exact_2"]); print("v7e REPLACES v7c:", win)
print(pd.crosstab(T.fear, T.v7e, rownames=["true"], colnames=["v7e"]))
T.to_parquet(OUT / "scary_sealed4_scored.parquet"); json.dump(dict(results=res, v7e_wins=win, opened=MARK.read_text()), open(OUT / "scary_sealed4_results.json", "w"), indent=1)
