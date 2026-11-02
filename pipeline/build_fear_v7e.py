"""v7e student (PLAN-v11 Amendment 5c-e): MiniLM distilled from the v12 U1 teacher (5-seed average of the raw ordinal expected score) on the 300k ABC corpus (text list from v11 teacher_scores.parquet),
U1 TRAIN rows as direct labels x3 (replace the teacher target where the text is in the corpus, else appended), U1 DEV rows never used in training (also dropped from the corpus, focus rows, ridge).
Joint focus as v7c; ridge refit (50k + all U1-train rows, weighted); cuts by the rule on U1 dev (5-fold CV inside dev for reporting; final cuts on all dev). Does NOT ship into Noesis.
Outputs v12/teacher_scores.parquet (chunked, resumable), v12/student_v7e/{encoder,heads.json}, v12/results_v12.json, v12/run_v12_student.log."""
import os, sys, gc, json, time, random, argparse, numpy as np, pandas as pd, torch, torch.nn.functional as F
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v10_model as VM, u1_eval as UE
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge, LogisticRegression
ap = argparse.ArgumentParser(); ap.add_argument("--retrain", action="store_true"); ARGS = ap.parse_args()
R = VM.R; E = R / "data/eval"; OUT = R / "data/processed/events/v12"; DEV = VM.DEV; LOG = open(OUT / "run_v12_student.log", "a")
NOE = Path(__file__).resolve().parents[1] / "noesis_fear"; V6D = NOE / "models/fear_v6"; V7D = NOE / "models/fear_v7"; SD = OUT / "student_v7e"
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
T0 = time.time(); TIM = {}; log("=" * 20, "build_fear_v7e start")
C = pd.read_parquet(OUT / "u1_corpus.parquet"); U_tr = C[C.split == "train"].reset_index(drop=True); U_dv = C[C.split == "dev"].reset_index(drop=True); dev_key = {t.lower() for t in U_dv.text}
yD = U_dv.fear.values; trD = ((U_dv.pool == "cue") & (U_dv.fear <= 1)).values; log(f"U1 train {len(U_tr)}, dev {len(U_dv)}, tricky-calm dev {int(trD.sum())}")
# ---------------- teacher scores on the 300k corpus (chunked, resumable) ----------------
TS = OUT / "teacher_scores.parquet"; CH = OUT / "ts_chunks"; CH.mkdir(exist_ok=True)
if TS.exists(): CORP = pd.read_parquet(TS); log(f"loaded cached teacher scores {len(CORP)}")
else:
    txt = pd.read_parquet(R / "data/processed/events/v11/teacher_scores.parquet").text.tolist(); log(f"corpus text list {len(txt)}"); t1 = time.time(); TM = [VM.V10(OUT / f"teacher/seed_{s}") for s in range(5)]; CS = 20000; parts = []
    for ci, b in enumerate(range(0, len(txt), CS)):
        fp = CH / f"c{ci:03d}.npy"
        if not fp.exists():
            chunk = txt[b:b + CS]; o = np.argsort([len(t) for t in chunk], kind="stable"); st = [chunk[i] for i in o]; acc = np.zeros(len(st))
            with torch.no_grad():
                for V in TM: acc += VM.head_all(V.H, VM.embed_all(V.m, st, bs=256, dev=DEV), None, dev=DEV)[0]
            out = np.zeros(len(st)); out[o] = acc / len(TM); np.save(fp, out); log(f"  scored chunk {ci} ({b + len(chunk)}/{len(txt)}) {time.time() - t1:.0f}s")
        parts.append(np.load(fp))
    CORP = pd.DataFrame(dict(text=txt, teacher_raw=np.concatenate(parts))); CORP.to_parquet(TS); TIM["teacher_score_s"] = round(time.time() - t1); del TM; gc.collect(); torch.mps.empty_cache()
# ---------------- student corpus ----------------
n0 = len(CORP); isdev = CORP.text.str.lower().isin(dev_key); CORP = CORP[~isdev].reset_index(drop=True); log(f"corpus {n0}; U1-dev texts removed {int(isdev.sum())}")
CORP["target"] = CORP.teacher_raw.astype(float); CORP["w"] = 1.0; ut = dict(zip(U_tr.text, U_tr.fear.astype(float))); ov = CORP.text.isin(ut); CORP.loc[ov, "target"] = CORP.loc[ov, "text"].map(ut); CORP.loc[ov, "w"] = 3.0
newp = U_tr[~U_tr.text.isin(set(CORP.text))]; CORP = pd.concat([CORP, pd.DataFrame(dict(text=newp.text.values, teacher_raw=np.nan, target=newp.fear.astype(float).values, w=3.0))], ignore_index=True)
log(f"U1 train rows: in corpus (target replaced) {int(ov.sum())}, appended {len(newp)}; weighted rows {int((CORP.w > 1).sum())}; corpus {len(CORP)}; teacher mean {CORP.teacher_raw.mean():.3f} target mean {CORP.target.mean():.3f}")
# ---------------- focus frame (as build_fear_v7c) ----------------
K = pd.read_parquet(E / "rating_v2/sample_key.parquet"); L = pd.read_csv(E / "rating_v2/labels_all.csv")
D = K.merge(L, on="id")[["id", "event", "text", "fear", "focus"]]
ev = sorted(D.event.unique()); te_d = D.event.isin(set(np.random.default_rng(41).choice(ev, 11, replace=False))).values
RL = pd.read_parquet(E / "relabel/relabel_decisions.parquet").set_index("id"); offt = D.id.isin(RL.index[RL.about_event == "no"]).values
# NB: v7c built D = disaster+everyday concat and used D.src=='disaster'; the event draw (rng 41 over disaster events) is identical
dm_ = te_d & ~offt; Dev_d = D[dm_].reset_index(drop=True)
fm = (~te_d & ~offt & D.focus.notna().values) & ~D.text.str.lower().isin(dev_key).values
FT = D.text[fm].tolist(); fcls = sorted(D.focus[fm].unique()); fy = np.array([fcls.index(c) for c in D.focus[fm]]); log(f"focus train rows {fm.sum()} classes {fcls} counts {np.bincount(fy).tolist()}; focus dev (disaster dev events) {len(Dev_d)}")
# ---------------- student with joint focus ----------------
MODEL = "sentence-transformers/all-MiniLM-L6-v2"; S = 10.
def train_student():
    t1 = time.time(); random.seed(0); np.random.seed(0); torch.manual_seed(0); rng = np.random.default_rng(0)
    m = SentenceTransformer(MODEL, device=DEV); m.max_seq_length = 32; m.train(); head = torch.nn.Linear(384, 1).to(DEV); torch.nn.init.zeros_(head.weight); torch.nn.init.constant_(head.bias, float(CORP.target.mean()))
    fh = torch.nn.Linear(384, len(fcls)).to(DEV); cwt = torch.tensor(len(fy) / (len(fcls) * np.bincount(fy)), dtype=torch.float32, device=DEV); fyt = torch.tensor(fy, device=DEV)
    opt = torch.optim.AdamW([dict(params=m.parameters(), lr=5e-5), dict(params=head.parameters(), lr=1e-3), dict(params=fh.parameters(), lr=1e-3)]); tx = CORP.text.tolist(); y = torch.tensor(CORP.target.values, dtype=torch.float32, device=DEV); wt = torch.tensor(CORP.w.values, dtype=torch.float32, device=DEV); perm = rng.permutation(len(tx)); n = (len(tx) + 63) // 64
    fperm = rng.permutation(len(FT)); fp = 0; nf = 0
    for k, b in enumerate(range(0, len(tx), 64)):
        ix = perm[b:b + 64]; z = head(VM.embed(m, [tx[i] for i in ix], DEV) * S).squeeze(1); loss = (wt[ix] * (z - y[ix]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if k % 4 == 3:
            if fp + 64 > len(fperm): fperm = rng.permutation(len(FT)); fp = 0
            jx = fperm[fp:fp + 64]; fp += 64; zf = fh(VM.embed(m, [FT[i] for i in jx], DEV) * S); lf = F.cross_entropy(zf, fyt[jx], weight=cwt); opt.zero_grad(); lf.backward(); opt.step(); nf += 1
        if k % 250 == 0: log(f"    step {k}/{n} fear loss {loss.item():.4f} focus loss {lf.item() if nf else float('nan'):.4f} {time.time() - t1:.0f}s")
    m.eval(); SD.mkdir(exist_ok=True); m.save(str(SD / "encoder")); torch.save(head.cpu().state_dict(), SD / "head.pt"); TIM["student_train_s"] = round(time.time() - t1); log(f"student trained {n} fear steps + {nf} focus steps in {TIM['student_train_s']}s"); return m
if (SD / "encoder").exists() and (SD / "head.pt").exists() and not ARGS.retrain: log("loading saved student", SD); M = SentenceTransformer(str(SD / "encoder"), device=DEV); M.eval()
else: M = train_student()
hd = torch.load(SD / "head.pt"); lw, lb = hd["weight"].numpy()[0] * S, float(hd["bias"][0])
enc = lambda m, ts: VM.embed_all(m, list(ts), bs=256, dev=DEV).numpy()
t1 = time.time(); sub = np.random.default_rng(0).choice(len(CORP), 50_000, replace=False); pki = np.where(CORP.w.values > 1)[0]; sub = np.unique(np.concatenate([sub[CORP.w.values[sub] == 1], pki]))
rr = Ridge(1.0).fit(enc(M, CORP.text.values[sub]), CORP.target.values[sub], sample_weight=CORP.w.values[sub]); fw, fb = rr.coef_, float(rr.intercept_); log(f"ridge rows {len(sub)} (U1 train {len(pki)}) {time.time() - t1:.0f}s")
E_dev = enc(M, U_dv.text); f_rdg = (E_dev @ fw + fb).clip(0, 3); f_lin = (E_dev @ lw + lb).clip(0, 3)
log(f"dev kappa (rounded raw) linear head {UE.qwk(np.clip(np.round(f_lin), 0, 3), yD):.3f}, ridge head {UE.qwk(np.clip(np.round(f_rdg), 0, 3), yD):.3f} (ridge used per plan)")
f7e = f_rdg
lg = LogisticRegression(max_iter=3000, class_weight="balanced").fit(enc(M, FT), D.focus[fm].values); assert list(lg.classes_) == fcls
# ---------------- reference models on U1 dev ----------------
def shipped(d, texts):
    h = json.load(open(d / "heads.json")); m = SentenceTransformer(str(d / "encoder"), device="cpu"); m.eval(); return h, m.encode(list(texts), batch_size=256, normalize_embeddings=True, show_progress_bar=False)
h6, E6 = shipped(V6D, U_dv.text); f6 = (E6 @ np.array(h6["fear_coef"]) + h6["fear_intercept"]).clip(0, 3)
h7, E7 = shipped(V7D, U_dv.text); f7c = (E7 @ np.array(h7["fear_coef"]) + h7["fear_intercept"]).clip(0, 3); log("installed fear_v7:", h7["version"], "| its own shipped cuts", h7["cuts"])
TP = pd.read_parquet(OUT / "teacher_dev_preds.parquet"); assert (TP.text.values == U_dv.text.values).all(); f12 = TP.avg5.values
def focus_eval(Ed, cw, cb, cls):
    z = Ed @ np.array(cw).T + np.array(cb); p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); pred = np.array(cls)[p.argmax(1)]; return round(float((pred == Dev_d.focus.values).mean()), 3)
_, E6d = shipped(V6D, Dev_d.text); _, E7d = shipped(V7D, Dev_d.text); E7ed = enc(M, Dev_d.text)
FOC = {"v6": focus_eval(E6d, h6["focus_coef"], h6["focus_intercept"], h6["focus_classes"]), "v7c": focus_eval(E7d, h7["focus_coef"], h7["focus_intercept"], h7["focus_classes"]), "v7e": focus_eval(E7ed, lg.coef_, lg.intercept_, list(lg.classes_))}; log("focus accuracy (disaster dev):", FOC)
# ---------------- U1 dev table ----------------
RES = {}; CUTS = {}
for nm, f in (("v6", f6), ("v7c", f7c), ("v12_teacher_avg5", f12), ("v7e", f7e)):
    cvm, fc, cf, ins = UE.cv_eval(f, yD, trD); RES[nm] = cvm; CUTS[nm] = dict(fold_cuts=fc, final=cf, insample=ins)
    from scipy.stats import spearmanr; RES[nm]["rho"] = round(float(spearmanr(f, yD).correlation), 3)
    if nm in FOC: RES[nm]["focus_acc"] = FOC[nm]
pd.set_option("display.width", 250); TB = pd.DataFrame(RES); log("U1 DEV TABLE (cuts tuned per model by the rule, 5-fold CV inside dev):\n" + TB.to_string()); log("final cuts (all dev):", {k: v["final"] for k, v in CUTS.items()}); log("in-sample (final cuts) kappa:", {k: v["insample"]["kappa"] for k, v in CUTS.items()})
b, e = RES["v7c"], RES["v7e"]
G = {"kappa >= v7c + .02": (e["kappa"] >= b["kappa"] + .02, e["kappa"], round(b["kappa"] + .02, 3)), "true-2 exact >= v7c + .05": (e["exact_2"] >= b["exact_2"] + .05, e["exact_2"], round(b["exact_2"] + .05, 3)),
     "rec>=2 >= .78": (e["rec2"] >= .78, e["rec2"], .78), "tricky FA <= .15": (e["fa"] <= .15, e["fa"], .15), "rec3 >= .65": (e["rec3"] >= .65, e["rec3"], .65), "focus >= v6 - .02": (FOC["v7e"] >= FOC["v6"] - .02, FOC["v7e"], round(FOC["v6"] - .02, 3))}
for k, (ok, a, bb) in G.items(): log(f"GATE {'PASS' if ok else 'FAIL'}: {k} (v7e {a} vs {bb})")
heads = dict(version="fear_v7 v12 v7e 2026-10", scale="user U1", target="v12 U1 teacher (5-seed avg, raw ordinal expected score) + U1 train direct x3", fear_coef=[round(float(x), 6) for x in fw], fear_intercept=fb, focus_classes=list(lg.classes_),
             focus_coef=[[round(float(x), 6) for x in r] for r in lg.coef_], focus_intercept=[float(x) for x in lg.intercept_], cuts=list(CUTS["v7e"]["final"]),
             heldout=f"U1 dev CV kappa {e['kappa']}, rec2 {e['rec2']}, tricky FA {e['fa']}, rec3 {e['rec3']}, exact_2 {e['exact_2']}; focus acc {FOC['v7e']}; fear head = ridge")
SD.mkdir(exist_ok=True); json.dump(heads, open(SD / "heads.json", "w"))
M.save(str(SD / "encoder")); chk = SentenceTransformer(str(SD / "encoder"), device="cpu"); Ec = chk.encode(U_dv.text.tolist()[:30], normalize_embeddings=True, show_progress_bar=False); log("reload check max|diff| vs training-time dev scores", round(float(np.abs((Ec @ fw + fb).clip(0, 3) - f7e[:30]).max()), 4))
TIM["total_s"] = round(time.time() - T0)
tj = json.load(open(OUT / "results_v12_teacher.json")); json.dump(dict(corpus=dict(train=len(U_tr), dev=len(U_dv)), teacher=tj, dev_table=RES, cuts=CUTS, focus=FOC, gates={k: dict(passed=bool(v[0]), v7e=v[1], bar=v[2]) for k, v in G.items()}, all_pass=all(bool(v[0]) for v in G.values()), timings=TIM, shipped=False), open(OUT / "results_v12.json", "w"), indent=1, default=str)
log("timings:", TIM); log("done")
