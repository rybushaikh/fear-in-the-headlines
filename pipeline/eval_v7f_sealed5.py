"""PLAN-v12 Amendment 2: cut rule B on U1 dev + P5 (5-fold CV), then sealed #5 opened once (v7e live/B, v7f dev/B). Locked by scary_sealed5.DONE."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import u1_eval as UE
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
R = Path(__file__).resolve().parents[1]; O13 = R / "data/processed/events/v13"; P = R / "data/eval/scary_test5_private"; DONE = O13 / "scary_sealed5.DONE"
assert not DONE.exists(), "sealed #5 already opened"
C = pd.read_parquet(R / "data/processed/events/v12/u1_corpus.parquet"); D = C[C.split == "dev"].reset_index(drop=True); P5 = pd.read_parquet(O13 / "p5_labelled.parquet")
T = pd.concat([pd.DataFrame(dict(text=D.text, y=D.fear.astype(int), tricky=(D.pool == "cue") & (D.fear <= 1))), pd.DataFrame(dict(text=P5.text, y=P5.fear.astype(int), tricky=(P5.arm == "ones") & (P5.fear <= 1)))], ignore_index=True)
S = pd.read_parquet(P / "SEALED_scary5_300.parquet"); yS = S.fear.values.astype(int); trS = ((S.pool == "cue") & (S.fear <= 1)).values
NOE = Path(__file__).resolve().parents[1] / "noesis_fear/models/fear_v7"
def raw(d, texts): h = json.load(open(d / "heads.json")); m = SentenceTransformer(str(d / "encoder"), device="cpu"); return (m.encode(list(texts), batch_size=256, normalize_embeddings=True, show_progress_bar=False) @ np.array(h["fear_coef"]) + h["fear_intercept"]).clip(0, 3), h
def tuneB(f, y, tr):
    hi = y >= 2; ok = [c for c in UE.GRID if (f[hi] >= c).mean() >= .90 and (f[tr] >= c).mean() <= .15]
    c2 = max(ok) if ok else UE.tune(f, y, tr)[1]; bk, b = -2, None
    for c1 in UE.GRID[UE.GRID < c2]:
        for c3 in UE.GRID[UE.GRID > c2]:
            k = UE.qwk(np.digitize(f, (c1, c2, c3)), y)
            if k > bk: bk, b = k, (float(c1), float(c2), float(c3))
    return b
y, tr = T.y.values, T.tricky.values.astype(bool); OUT = {}
for nm, d in (("v7e", NOE), ("v7f", O13 / "student_v7f")):
    fT, h = raw(d, T.text); fS, _ = raw(d, S.text)
    lv = np.zeros(len(y), int)
    for a, b in KFold(5, shuffle=True, random_state=0).split(fT): lv[b] = np.digitize(fT[b], tuneB(fT[a], y[a], tr[a]))
    cB = tuneB(fT, y, tr); OUT[f"{nm}_ruleB_cv_dev+P5"] = UE.metrics(lv, y, tr)
    OUT[f"{nm}_ownCuts_dev+P5"] = UE.metrics(np.digitize(fT, h["cuts"]), y, tr)
    OUT[f"{nm}_own cuts {tuple(h['cuts'])}"] = UE.metrics(np.digitize(fS, h["cuts"]), yS, trS)
    OUT[f"{nm}_ruleB cuts {cB}"] = UE.metrics(np.digitize(fS, cB), yS, trS)
    from sklearn.metrics import roc_auc_score; OUT[f"{nm}_AUC"] = round(float(roc_auc_score(yS >= 2, fS)), 3)
DONE.write_text("opened 2026-10-27"); json.dump(OUT, open(O13 / "scary_sealed5_results.json", "w"), indent=1)
pd.set_option("display.width", 250); print(pd.DataFrame({k: v for k, v in OUT.items() if isinstance(v, dict)}).T.to_string()); print({k: v for k, v in OUT.items() if not isinstance(v, dict)})
