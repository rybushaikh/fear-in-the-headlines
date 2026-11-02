"""v12 teacher (PLAN-v11 Amendment 5b): MPNet, v10 F2F3 recipe (DIRECT + CORAL + synthetic 1/3), trained on the user's U1 labels (train split, weight 2).
Arm i = U1 train only; arm ii = U1 train + the v9-frame G1/v2 rater rows exactly as train_v10 builds them (weight x0.5). Both seed 0 -> pick by dev kappa (cuts by the rule, 5-fold CV inside dev; arm ii must beat arm i by > .005) -> seeds 1-4 of the winner.
Eval on U1 dev only. Saves data/processed/events/v12/teacher/seed_k (v10_model.V10-loadable) + dev_preds.parquet. Resumable: finished models are skipped."""
import os, sys, gc, random, time, json, hashlib, shutil, numpy as np, pandas as pd, torch, torch.nn.functional as F
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v10_model as VM, u1_eval as UE
from sentence_transformers import SentenceTransformer
R = VM.R; E = R / "data/eval"; OUT = R / "data/processed/events/v12"; TD = OUT / "teacher"; TD.mkdir(parents=True, exist_ok=True); LOG = open(OUT / "run_v12_teacher.log", "a"); DEVICE = VM.DEV; SEED = 1023
ENC = "sentence-transformers/all-mpnet-base-v2"
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
hsh = lambda s: int(hashlib.md5(f"{SEED}:{s}".encode()).hexdigest(), 16) % 100
log("=" * 20, "train_v12 start")
# ---------------- U1 corpus ----------------
C = pd.read_parquet(OUT / "u1_corpus.parquet"); U_tr = C[C.split == "train"].reset_index(drop=True); U_dv = C[C.split == "dev"].reset_index(drop=True); dev_txt = set(U_dv.text); dev_key = {t.lower() for t in dev_txt}
yD = U_dv.fear.values; trD = ((U_dv.pool == "cue") & (U_dv.fear <= 1)).values; log(f"U1 train {len(U_tr)}, dev {len(U_dv)} (tricky-calm {int(trD.sum())})")
# ---------------- v9 frame (as train_v10.py) ----------------
K = pd.read_parquet(E / "rating_v2/sample_key.parquet"); L = pd.read_csv(E / "rating_v2/labels_all.csv")
Dd = K.merge(L, on="id")[["id", "event", "text", "fear", "focus"]].assign(src="disaster", topic=None)
Gk = pd.read_parquet(E / "rating_general_private/sample_key.parquet"); Gl = pd.read_csv(E / "rating_general_private/labels_all.csv")
Dg = Gk.merge(Gl, on="id")[["id", "text", "fear", "topic"]].assign(src="everyday", event=None, focus=None)
D = pd.concat([Dd, Dg], ignore_index=True)
ev = sorted(Dd.event.unique()); te_d = ((D.src == "disaster") & D.event.isin(set(np.random.default_rng(41).choice(ev, 11, replace=False)))).values
gi = D.index[D.src == "everyday"].to_numpy(); te_g = D.index.isin(np.random.default_rng(81).choice(gi, 400, replace=False)); tr8 = ~(te_d | te_g)
RL = pd.read_parquet(E / "relabel/relabel_decisions.parquet").set_index("id")
D["fear_rl"] = D.fear.astype(float); hit = D.id.isin(RL.index); D.loc[hit, "fear_rl"] = D.loc[hit, "id"].map(RL.new_fear).astype(float)
offt = D.id.isin(RL.index[RL.about_event == "no"]).values; chg = (D.fear_rl != D.fear).values
W = pd.concat([pd.read_json(E / f"weakspots/out/labels_0{i}.jsonl", lines=True) for i in (1, 2, 3)]).merge(pd.read_parquet(E / "weakspots_private/key.parquet")[["id", "text", "group", "heldout"]], on="id"); te_w = W.heldout.values
A = pd.read_csv(E / "cursor_pilot_private/ADJUDICATED_39.csv"); it = pd.read_json(E / "cursor_pilot/items_40.jsonl", lines=True).set_index("id"); A["text"] = A.id.map(it.text)
Hc = pd.read_csv(E / "rating_general_private/hidden_user_checks.csv")
FS = pd.concat([pd.read_csv(E / f"fear_scale/USER_FEAR_{n}_rated.csv") for n in (100, 60)])
PL = pd.concat([pd.read_json(E / f"v10_parts/out/labels_0{i}.jsonl", lines=True) for i in range(1, 5)]).merge(pd.read_parquet(E / "v10_private/parts_key.parquet")[["id", "pick", "text"]], on="id"); PL["dev"] = PL.id.map(hsh) < 15
SK = pd.read_parquet(E / "v10_private/synth_write_key.parquet"); SK = SK[SK.side == "train"]
SY = pd.concat([pd.read_json(E / f"v10_synth_write/out/edits_{i:02d}.jsonl", lines=True) for i in range(1, 11)]).merge(SK[["base_id", "fear", "text"]].rename(columns={"fear": "base_fear", "text": "base_text"}), on="base_id")
SY["text"] = SY.text.astype(str).str.strip(); SY = SY[(SY.text != SY.base_text) & (SY.text.str.len() > 3)].drop_duplicates(["base_id", "edit"]).reset_index(drop=True); SY["dev"] = SY.base_id.map(hsh) < 15
GCP = R / "data/processed/events/v10/grammar_feats.parquet"; GC = pd.read_parquet(GCP); miss = sorted(set(SY.text) - set(GC.index))
if miss: GC = pd.concat([GC.astype("float32"), VM.gram_feats(miss).set_index(pd.Index(miss))]); log(f"grammar feats computed for {len(miss)} uncached synthetic texts (in memory only)")
dth = GC.loc[SY.text.tolist()].death_word.values > 0
spec = [VM.edit_spec(e, b, t, d) for e, b, t, d in zip(SY.edit, SY.base_fear, SY.text, dth)]; SY["lo"], SY["hi"], SY["dir"] = [s[0] for s in spec], [s[1] for s in spec], [s[2] for s in spec]
# v10's test-side exclusion set, plus the U1 dev texts
test_txt = set(D.text[te_d | te_g]) | set(W.text[te_w]) | set(PL.text[PL.dev]) | set(SY.text[SY.dev]) | set(SY.base_text[SY.dev]) | set(A.text) | set(Hc.text) | set(FS.text)
bad_syn = SY.text.isin(dev_txt) | SY.base_text.isin(dev_txt); log(f"synthetic edits whose text/base is a U1 dev text (dropped from training): {int(bad_syn.sum())}")
SY_train = SY[~SY.dev & ~bad_syn].reset_index(drop=True); SYP = np.array([s[3] for s in spec])[(~SY.dev & ~bad_syn).values]
keep = tr8 & ~offt & ~D.text.isin(test_txt).values & ~D.text.isin(dev_txt).values; Wtr = W[~te_w & ~W.text.isin(test_txt) & ~W.text.isin(dev_txt)]
rs_rl = RL[RL.index.isin(D.id[keep])].reason.dropna(); rs_rl = rs_rl[rs_rl.str.len() > 5]; Di = D.set_index("id")
def rows(text, y, foc, w): return pd.DataFrame(dict(text=list(text), y=np.asarray(y, float), foc=foc, w=w))
wtr_w = np.where(Di.loc[rs_rl.index].fear_rl.values != Di.loc[rs_rl.index].fear.values, 1.5, 1.)
T9 = pd.concat([rows(D.text[keep], D.fear_rl[keep], list(D.focus[keep]), np.where(chg[keep], 1.5, 1.)), rows(Wtr.text, Wtr.fear, [None] * len(Wtr), 2.), rows(Wtr.reason, Wtr.fear, [None] * len(Wtr), 2.),
                rows(rs_rl.tolist(), Di.loc[rs_rl.index].fear_rl, [None] * len(rs_rl), wtr_w)], ignore_index=True)
T9 = T9[~T9.text.str.lower().isin(dev_key)].reset_index(drop=True); T9["w"] = T9.w * 0.5
TU1 = rows(U_tr.text, U_tr.fear, [None] * len(U_tr), 2.)
TRAIN = {"i": TU1, "ii": pd.concat([TU1, T9], ignore_index=True)}; log(f"arm i rows {len(TRAIN['i'])}; arm ii rows {len(TRAIN['ii'])} (U1 {len(TU1)} x2 + v9 frame {len(T9)} x0.5); syn train edits {len(SY_train)}")
CFG = dict(head="coral", pair_w=.5, per=4, epochs=2, syn=1 / 3, syn_w=1., parts_rows="none", heads=False, gram=False, weighted=True)
def train(T, seed):
    t0 = time.time(); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); rng = np.random.default_rng(seed); cfg = CFG
    n = len(T); per, epochs = cfg["per"], cfg["epochs"]; m = SentenceTransformer(ENC, device=DEVICE); m.train(); d = m.get_sentence_embedding_dimension()
    H = VM.Heads(d, "coral", False, 0).to(DEVICE); opt = torch.optim.AdamW([dict(params=m.parameters(), lr=2e-5), dict(params=H.parameters(), lr=1e-3)])
    yT = torch.tensor(T.y.values, dtype=torch.float32, device=DEVICE); wT = torch.tensor(T.w.values, dtype=torch.float32, device=DEVICE); txt = T.text.tolist(); foc = T.foc.tolist(); Ty = T.y.values
    wmean = lambda l, w: (l * w).sum() / w.sum().clamp(min=1e-6)
    def fear_loss(o, y, w):
        tg = (y[:, None] - torch.tensor([0., 1., 2.], device=DEVICE)).clamp(0, 1); return wmean(F.binary_cross_entropy_with_logits(o["coral"], tg, reduction="none").mean(1), w)
    St = SY_train; bs = 16; steps_done = 0; nsteps = None
    for ep in range(epochs):
        pairs = []
        for i in range(n):
            for _ in range(per):
                j = int(rng.integers(n))
                if j == i: continue
                fs = 1 - abs(Ty[i] - Ty[j]) / 3
                pairs.append((i, j, 0.5 * fs + 0.5 * float(foc[i] == foc[j]) if foc[i] is not None and foc[j] is not None else fs))
        rng.shuffle(pairs); ops = [("r", pairs[b:b + bs]) for b in range(0, len(pairs), bs)]
        ns = int(cfg["syn"] * n); sel = rng.integers(len(St), size=ns); ops += [("s", sel[b:b + bs]) for b in range(0, ns, bs)]; rng.shuffle(ops)
        nsteps = len(ops) if nsteps is None else nsteps
        for kind, bt in ops:
            if kind == "r":
                ia, ib = [p[0] for p in bt], [p[1] for p in bt]; emb = VM.embed(m, [txt[i] for i in ia + ib], DEVICE); nb = len(bt)
                cos = F.cosine_similarity(emb[:nb], emb[nb:]); sim = torch.tensor([p[2] for p in bt], device=DEVICE, dtype=cos.dtype); ix = torch.tensor(ia + ib, device=DEVICE)
                pw = (wT[ix[:nb]] + wT[ix[nb:]]) / 2; loss = cfg["pair_w"] * wmean((cos - sim) ** 2, pw); o = H(emb, None); loss = loss + fear_loss(o, yT[ix], wT[ix])
            else:
                b = St.iloc[bt]; nb = len(b); emb = VM.embed(m, b.base_text.tolist() + b.text.tolist(), DEVICE); o = H(emb, None); pb, pe = o["fear"][:nb].detach(), o["fear"][nb:]
                lo, hi = [torch.tensor(b[c].values, dtype=torch.float32, device=DEVICE) for c in ("lo", "hi")]; dr = b.dir.values
                rng_l = (F.relu(lo - pe) + F.relu(pe - hi)).mean()
                mg = torch.tensor([.3 if x == "lower" else 0. for x in dr], device=DEVICE); sgn = torch.tensor([{"lower": 1., "lower_or_equal": 1., "higher_or_equal": -1., "same_high": 0.}[x] for x in dr], device=DEVICE)
                rk = (sgn * (pe - pb) + mg) * (sgn != 0).float(); rk = F.relu(rk) + F.relu(2 - pe) * torch.tensor([float(x == "same_high") for x in dr], device=DEVICE)
                loss = cfg["syn_w"] * (rng_l + rk.mean())
            opt.zero_grad(); loss.backward(); opt.step(); steps_done += 1
            if steps_done % 50 == 0: gc.collect(); torch.mps.empty_cache()   # keep MPS memory flat (memory pressure on this machine)
            if steps_done % 250 == 0: log(f"    step {steps_done}/{nsteps * epochs} {time.time() - t0:.0f}s")
    m.eval(); tt = time.time() - t0; log(f"  trained {steps_done} steps on {n} rows in {tt:.0f}s"); return m, H.cpu(), d, tt
def save(m, H, d, p):
    p.mkdir(parents=True, exist_ok=True); m.save(str(p / "encoder")); torch.save(H.state_dict(), p / "heads.pt")
    json.dump(dict(cfg=CFG, head="coral", heads=False, gram=False, d=d, encoder=ENC, gnames=VM.GN, gmu=[0.] * len(VM.GN), gsd=[1.] * len(VM.GN), parts=VM.PARTS), open(p / "config.json", "w"), default=str)
def dev_pred(m, H): return VM.head_all(H, VM.embed_all(m, U_dv.text.tolist(), dev=DEVICE), None)[0]
def run(arm, seed, p):
    """train+save+dev-predict one model; skip if finished (dev preds cached in p/dev_pred.npy)."""
    if (p / "dev_pred.npy").exists(): log(f"skip {p.name}: done"); return np.load(p / "dev_pred.npy")
    shutil.rmtree(p, ignore_errors=True); log(f"training arm {arm} seed {seed} -> {p}"); m, H, d, tt = train(TRAIN[arm], seed); save(m, H, d, p); f = dev_pred(m, H); np.save(p / "dev_pred.npy", f); json.dump(dict(secs=tt, arm=arm, seed=seed), open(p / "train_info.json", "w")); del m, H; gc.collect(); torch.mps.empty_cache()
    if os.environ.get('V12_ONE'): log('V12_ONE: exiting after one trained model (fresh process per model, avoids MPS memory growth)'); sys.exit(0)
    return f
def report(name, f):
    cvm, fc, cf, ins = UE.cv_eval(f, yD, trD); log(f"{name}: CV {cvm} | final cuts {cf} in-sample {ins['kappa']}"); return dict(cv=cvm, fold_cuts=fc, final_cuts=cf, insample=ins)
RES = {}; t_all = time.time()
fa = {}; arms = {}
for arm in ("i", "ii"):
    fa[arm] = run(arm, 0, TD / f"_arm_{arm}_seed0"); arms[arm] = report(f"arm {arm} seed0", fa[arm])
ki, kii = arms["i"]["cv"]["kappa"], arms["ii"]["cv"]["kappa"]; win = "ii" if kii > ki + .005 else "i"; log(f"ARM CHOICE: arm i kappa {ki}, arm ii kappa {kii} -> {win} (ii must beat i by > .005; tie -> i)")
shutil.rmtree(TD / "seed_0", ignore_errors=True); shutil.copytree(TD / f"_arm_{win}_seed0", TD / "seed_0")
for a in ("i", "ii"): shutil.rmtree(TD / f"_arm_{a}_seed0", ignore_errors=True) if False else None
FS_ = {0: fa[win]}
for s in range(1, 5): FS_[s] = run(win, s, TD / f"seed_{s}")
for s in range(5): RES[f"seed{s}"] = report(f"{win} seed{s}", FS_[s]) if s else arms[win]
avg = np.mean([FS_[s] for s in range(5)], 0); RES["avg5"] = report("TEACHER avg5", avg)
pd.DataFrame({"text": U_dv.text, "fear": yD, "source": U_dv.source, "pool": U_dv.pool, "tricky": trD, **{f"seed{s}": FS_[s] for s in range(5)}, "avg5": avg}).to_parquet(OUT / "teacher_dev_preds.parquet")
for a in ("i", "ii"): shutil.rmtree(TD / f"_arm_{a}_seed0", ignore_errors=True)   # seed_0 of the winner is kept as teacher/seed_0
json.dump(dict(arm_choice=win, arm_i=arms["i"], arm_ii=arms["ii"], results=RES, secs=round(time.time() - t_all)), open(OUT / "results_v12_teacher.json", "w"), indent=1, default=str)
log("done", round(time.time() - t_all), "s")
