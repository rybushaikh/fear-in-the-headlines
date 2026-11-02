"""v7e (26 Oct 2026): distilled from the v12 MPNet teacher trained directly on ~3,900 U1 labels. Sealed NYT set #4 (300, opened once): quadratic kappa .927, exact 81%, true-2 exact 77%, 2-3s caught 94% (precision 92%), 3s caught 83%, false alarms on tricky calm 17%, AUC .977 (v7c: .850 / 69% / 53%). Focus accuracy .75.
Fear (0-3, on the user's own scale "U1") and story type (core / related / side / opinion) for short texts, fully offline, one shared encoder.
Model (v11b, 25 Oct 2026): MiniLM-L6 (max_seq 32) distilled by MSE from the v10 teacher (5 MPNet seeds, averaged). Target = piecewise-linear map of the raw teacher score through the U1 cut-points
(raw 0, .95, 1.9, 2.65, 3 -> 0, .5, 1.5, 2.5, 3, so round(target) = the cut-point level). Focus trained jointly with fear (CE head on disaster train rows, ~1 focus batch per 4 fear batches), then
logistic focus head refit as v6; fear head = ridge on 50k student embeddings. Corpus: 300k ABC headlines + labelled train-side texts, dev/outlet texts excluded. Built by pipeline/build_fear_v7b.py;
encoder in models/fear_v7/encoder (git-ignored), heads.json tracked. Older (v11) build kept in models/fear_v7_old.
Update v11c (25 Oct 2026): retrained with the 2s pack (600 ABC headlines labelled on U1, direct targets, weight x3; ridge refit weighted); previous build kept in models/fear_v7b. Built by pipeline/build_fear_v7c.py.
Cut-point CV (5-fold, 1,500 U1 labels, sealed sets 1-3 + 500 pool + user's 100), v7b -> v7c: kappa .78 -> .81; recall of 2-3s .79 -> .82 (precision .78 -> .79); recall of 3s .68 -> .72; tricky-calm false alarms .14 -> .145. (v7b's own cuts were tuned on part of this set.)
Dev numbers (dev only, no sealed claim), v7c vs v6 (everyday/disaster labels are on the rater scale, not U1):
  everyday-400 rho .742 vs .543; disaster-11-events rho .773 vs .758; weak-spot dev MAE .45 vs .79; synthetic-edit direction mean .83 vs .62; user's 100 rho .772 vs .689, quadratic kappa of levels .77 vs .51.
  Outlet 600 (U1): AUC (label>=2) .956 vs v6 .896; quadratic kappa .68 vs .28; recall of level 3 .33 (3 of 9) vs v6 .33.
  Focus accuracy on disaster dev 76.0% vs v6 75.6% (gate passes). A final claim needs a fresh sealed set #4.
Known limits: scale is one person's (U1); student is a little worse than its teacher (counter-case MAE .71 vs .39, 'far away' edits worse); few level-3 items in the outlet check (9);
informal text untested; not yet validated on a fresh sealed set after the v11c retrain."""
import json, os
from pathlib import Path
import numpy as np
_DIR = Path(__file__).parent / "models" / "fear_v7"; _M = {}
LEVELS = ["no fear", "mild concern", "clear fear", "alarm"]
# Level cut-points on the raw head output (PLAN-v11 amendment 3, v11c): max recall of 2-3s with <= 15% false alarms on tricky calm news, then kappa; fit on 1,500 U1 labels (5-fold CV: recall 2-3 .82, tricky FA .145, recall 3 .72, kappa .81).
# The remap sends each cut to a level boundary, so round(fear) is the level.
CUTS = (0.625, 1.225, 2.475)   # v7e (26 Oct 2026): student of the U1-trained v12 teacher
_remap = lambda f: np.interp(f, (0, *CUTS, 3), (0, .5, 1.5, 2.5, 3))

def _load():
    if not _M:
        from sentence_transformers import SentenceTransformer
        h = json.load(open(_DIR / "heads.json"))
        _M.update(enc=SentenceTransformer(str(_DIR / "encoder"), device="cpu"), fw=np.array(h["fear_coef"]), fb=h["fear_intercept"],
                  cw=np.array(h["focus_coef"]), cb=np.array(h["focus_intercept"]), cls=h["focus_classes"])
        sp = _DIR / "subject_head.json"   # subject head (27 Oct 2026): 10 subjects, logistic on this encoder; silver labels from NYT/Guardian tags + HuffPost
        if sp.exists(): s = json.load(open(sp)); _M.update(sw=np.array(s["coef"]), sb=np.array(s["intercept"]), scls=s["classes"])
    return _M

def appraise(texts):
    """str or list[str] -> dict(s) {fear 0-3, level, fear_range, focus, focus_conf, focus_tier (trusted/usable/unsure), focus_p}."""
    single = isinstance(texts, str); texts = [texts] if single else list(texts); m = _load()
    E = m["enc"].encode(texts, normalize_embeddings=True, show_progress_bar=False)
    f = _remap((E @ m["fw"] + m["fb"]).clip(0, 3)); z = E @ m["cw"].T + m["cb"]; p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    if "sw" in m: zs = E @ m["sw"].T + m["sb"]; ps = np.exp(zs - zs.max(1, keepdims=True)); ps /= ps.sum(1, keepdims=True)
    else: ps = [None] * len(texts)
    out = []
    for a, q, qs in zip(f, p, ps):
        lo, hi = int(np.floor(a)), int(np.ceil(a))
        rng = LEVELS[int(round(a))] if abs(a - round(a)) <= .25 else f"{LEVELS[lo]} to {LEVELS[hi]}"   # honest range: dev answers are within one level 97.5% of the time, exact 58%; this range contains the true level 76%
        c = float(q.max()); tier = "trusted" if c >= .9 else ("usable" if c >= .6 else "unsure")    # dev: >=.9 -> 86% correct (68% of items); .6-.9 -> 56% (24%); <.6 -> 43% (9%)
        out.append(dict(fear=round(float(a), 2), level=LEVELS[int(round(a))], fear_range=rng, focus=m["cls"][int(q.argmax())], focus_conf=round(c, 2), focus_tier=tier,
                        focus_p={k: round(float(x), 2) for k, x in zip(m["cls"], q)}))
        if qs is not None:   # hand check (193 headlines): 67% right overall, 71% on ABC; when conf >= .6 (about half): 85% (ABC 91%)
            sc = float(qs.max()); out[-1].update(subject=m["scls"][int(qs.argmax())], subject_conf=round(sc, 2), subject_tier="confident" if sc >= .6 else "unsure")
    return out[0] if single else out

def enabled(): return os.environ.get("NOESIS_FEAR_V7", "1") != "0"   # on by default since 25 Oct 2026

if __name__ == "__main__":
    for t in ["sandy makes landfall and batters us east coast", "ariana grande leads manchester charity concert", "madrid bombing suspects on remand",
              "why syrian refugee passport must be treated with caution", "u.n. finds 'minimal' impact from tsunami in myanmar", "50000 jobs at risk as factory closures loom"]:
        r = appraise(t); print(r["fear"], r["fear_range"], "|", r["focus"], r["focus_tier"], r["focus_conf"], "|", t)
