"""v10 model pieces shared by train_v10.py, final_eval_v10.py and users: grammar features, heads+combiner, edit specs, predict(texts).
Grammar features (structure first, no word-list scoring): spaCy parse via Noesis appraisal.base.parse; WordNet lexname of the root verb;
negation / modal-hedge / question / old-year / history cue / death-injury word + log1p(largest attached number); Noesis appraise() outputs.
Usage: from v10_model import predict; predict(["two killed in bus crash"]) -> fear array (0-3)."""
import os, re, sys, json, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from functools import lru_cache
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
R = Path(__file__).resolve().parents[1]; MODEL_DIR = R / "data/processed/events/v10/student_v10"
NOESIS = str(Path(__file__).resolve().parents[1] / "noesis_fear")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
PARTS = {"severity": [0, 1, 2, 3], "status": ["live", "coming", "over", "history", "none"], "certainty": ["fact", "possible", "denied", "question"],
         "closeness": ["near", "far", "unclear"], "literal": ["literal", "figurative"]}
PK = list(PARTS); PN = [len(v) for v in PARTS.values()]   # part keys, classes per head (18 probs in the combiner)
enc_part = lambda k, v: PARTS[k].index(v) if v in PARTS[k] else -1

# ---- edit specs (plan table): (lo, hi, direction, implied part labels) ----
def edit_spec(edit, base, text, death=False):
    t = text.rstrip().endswith("?")
    lo, hi, d = {"T1": (0, 1, "lower"), "T2": (0, 1, "lower"), "T3": (0, base - 1, "lower"), "T4": (0, base, "lower_or_equal"), "T5": (0, 0, "lower"),
                 "T6": (max(base, 2) if death else base, 3, "higher_or_equal"), "T7": (0, base, "lower_or_equal"), "T8": (2, 3, "same_high")}[edit]
    p = {"T1": {"status": "over"}, "T2": {"status": "history"}, "T3": {"certainty": "denied"}, "T4": {"certainty": "question" if t else "possible"},
         "T5": {"literal": "figurative"}, "T6": {"severity": 3}, "T7": {"closeness": "far"}, "T8": {"severity": 3}}[edit]
    return float(lo), float(max(lo, hi)), d, [enc_part(k, p[k]) if k in p else -1 for k in PK]
def dir_ok(d, pb, pe, tol=.05):
    return {"lower": pe < pb - tol, "lower_or_equal": pe <= pb + tol, "higher_or_equal": pe >= pb - tol, "same_high": pe >= pb - .5}[d]

# ---- grammar features ----
LEX = [f"verb.{n}" for n in "body change cognition communication competition consumption contact creation emotion motion perception possession social stative weather".split()] + ["none"]
GN = ["neg", "modal", "hedge", "question", "old_year", "hist_word", "death_word", "death_num"] + ["lex_" + l for l in LEX] + \
     ["ap_event", "ap_framing"] + ["ap_" + d for d in "uncertainty control immediacy proximity amplification attenuation".split()]
MODALS = {"could", "may", "might", "would"}
HEDGE_LEM = {"fear", "suspect", "warn", "risk", "likely", "possible", "potential", "tipped", "expect", "threaten", "unconfirmed", "reportedly", "reportedly", "feared"}
HEDGE_SYN = {"suspect.v.01", "fear.v.01", "expect.v.01", "warn.v.01", "predict.v.01"}
HARM_LEM = {"kill", "dead", "die", "death", "dying", "injure", "hurt", "wound", "drown", "perish", "fatal", "fatality", "casualty", "murder", "slain"}
HARM_SYN = {"kill.v.01", "die.v.01", "injure.v.01", "wound.v.01", "death.n.01", "dead.a.01", "murder.v.01"}
NEGW = {"no", "not", "never", "nobody", "none", "without", "neither", "nor", "nothing"}
NEGRE = re.compile(r"\b(rules? out|ruled out|no sign|no one|denies|denied|false alarm)\b")
HISTRE = re.compile(r"anniversar|rememb|memorial|years? (on|after|since|ago)|decades? (on|after|ago)|\bhistory\b|inquest|commemorat|tribute|looks? back")
YEARRE = re.compile(r"\b(1[89]\d\d|20[0-3]\d)\b")
WNUM = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(), range(21)))
WNUM.update(thirty=30, forty=40, fifty=50, sixty=60, seventy=70, eighty=80, ninety=90, hundred=100, dozen=12, dozens=24, hundreds=200, thousands=2000, scores=40)
MULT = {"hundred": 1e2, "thousand": 1e3, "million": 1e6, "billion": 1e9}
_ap = {}
def _load():
    if not _ap:
        if NOESIS not in sys.path: sys.path.insert(0, NOESIS)
        import appraisal; from appraisal.base import parse; from nltk.corpus import wordnet as wn
        _ap.update(appraise=appraisal.appraise, parse=parse, wn=wn)
    return _ap
@lru_cache(maxsize=None)
def _closure(lemma, pos):
    wn = _load()["wn"]; out = set()
    for s in wn.synsets(lemma, pos)[:2]:
        out.add(s.name()); out |= {h.name() for h in s.closure(lambda x: x.hypernyms())}
    return frozenset(out)
@lru_cache(maxsize=None)
def _harm(lemma):   # death/injury word: short explicit list, or WordNet closure (verb/noun/adj) hits a kill/die/injure/death synset
    return lemma in HARM_LEM or any(HARM_SYN & _closure(lemma, p) for p in "vna")
@lru_cache(maxsize=None)
def _lex(lemma):
    s = _load()["wn"].synsets(lemma, "v"); return s[0].lexname() if s else "none"
def _num(tok, doc):
    t = tok.text.lower().replace(",", "")
    try: v = float(t)
    except ValueError: v = WNUM.get(t)
    if v is None: return None
    if 1900 <= v <= 2030 and t.isdigit(): return None
    if tok.i + 1 < len(doc) and doc[tok.i + 1].lower_ in MULT: v *= MULT[doc[tok.i + 1].lower_]
    return v
def _feat(text):
    A = _load(); doc = A["parse"](text); toks = list(doc); low = text.lower(); f = dict.fromkeys(GN, 0.)
    f["neg"] = float(any(t.dep_ == "neg" or t.lower_ in NEGW for t in toks) or bool(NEGRE.search(low)))
    f["modal"] = float(any(t.tag_ == "MD" and t.lower_ in MODALS for t in toks))
    f["hedge"] = float(any(t.lemma_.lower() in HEDGE_LEM or (t.pos_ == "VERB" and HEDGE_SYN & _closure(t.lemma_.lower(), "v")) for t in toks))
    f["question"] = float(text.strip().endswith("?"))
    ys = [int(y) for y in YEARRE.findall(low)]; f["old_year"] = float(any(1900 <= y <= 2000 for y in ys)); f["hist_word"] = float(bool(HISTRE.search(low)))
    harm = [t for t in toks if t.pos_ in ("VERB", "NOUN", "ADJ") and _harm(t.lemma_.lower())]; f["death_word"] = float(bool(harm)); best = 0.
    for h in harm:
        near = [c for c in h.children] + [h.head] + [t for t in toks if abs(t.i - h.i) <= 3]
        for c in near:
            if c.like_num:
                v = _num(c, doc); best = max(best, v or 0.)
    f["death_num"] = math.log1p(best)
    root = next((t for t in toks if t.dep_ == "ROOT" and t.pos_ in ("VERB", "AUX")), None) or next((t for t in toks if t.pos_ == "VERB"), None)
    lx = _lex(root.lemma_.lower()) if root is not None else "none"; f["lex_" + (lx if lx in LEX else "none")] = 1.
    try:
        a = A["appraise"](text); f["ap_event"] = float(a.event); f["ap_framing"] = float(a.framing)
        for k, v in a.dimensions.items(): f["ap_" + k] = float(v.score)
    except Exception: pass
    return f
def gram_feats(texts):
    """DataFrame[len(texts) x len(GN)] of raw grammar features (unstandardised)."""
    rows = []
    for t in texts:
        try: rows.append(_feat(t))
        except Exception: rows.append(dict.fromkeys(GN, 0.))
    return pd.DataFrame(rows).reindex(columns=GN).fillna(0.).astype("float32")

# ---- heads + combiner ----
class Heads(nn.Module):
    """Linear heads on the pooled (unit-norm, x S) embedding: fear (mse scalar or CORAL 3 thresholds), five part heads, and (parts=True)
    a combiner fear = fear_emb + MLP(concat[fear_emb, 18 part probs, grammar]) with one hidden layer of 32."""
    def __init__(s, d, head="mse", parts=False, ng=0, hid=32, S=10.):
        super().__init__(); s.head, s.parts, s.ng, s.S = head, parts, ng, S
        s.fear = nn.Linear(d, 1); nn.init.constant_(s.fear.bias, 1.0 if head == "mse" else 0.)
        if head == "coral": s.b0 = nn.Parameter(torch.tensor([1.5])); s.bd = nn.Parameter(torch.zeros(2))
        if parts:
            s.ph = nn.ModuleList([nn.Linear(d, n) for n in PN]); s.comb = nn.Sequential(nn.Linear(1 + sum(PN) + ng, hid), nn.ReLU(), nn.Linear(hid, 1))
            nn.init.zeros_(s.comb[2].weight); nn.init.zeros_(s.comb[2].bias)
    def forward(s, emb, g=None):
        x = emb * s.S; z = s.fear(x); o = {"coral": None}
        if s.head == "coral":
            th = torch.cat([s.b0, s.b0 - F.softplus(s.bd[:1]), s.b0 - F.softplus(s.bd[:1]) - F.softplus(s.bd[1:])]); o["coral"] = z + th; fe = torch.sigmoid(o["coral"]).sum(1)
        else: fe = z.squeeze(1)
        o["fear_emb"] = fe; o["fear"] = fe
        if s.parts:
            o["parts"] = [h(x) for h in s.ph]; pr = torch.cat([F.softmax(l, -1) for l in o["parts"]], 1); zc = [fe[:, None], pr]
            if s.ng: zc.append(g)
            o["fear"] = fe + s.comb(torch.cat(zc, 1)).squeeze(1); o["probs"] = pr
        return o

def embed(m, texts, dev=DEV):
    f = {k: v.to(dev) for k, v in m.tokenize(list(texts)).items()}; return F.normalize(m(f)["sentence_embedding"], dim=-1)
@torch.no_grad()
def embed_all(m, texts, bs=64, dev=DEV):
    m.eval(); return torch.cat([embed(m, texts[i:i + bs], dev).float().cpu() for i in range(0, len(texts), bs)]) if len(texts) else torch.zeros(0, 1)
@torch.no_grad()
def head_all(H, E, G=None, bs=512, dev="cpu"):
    """E: embeddings (cpu); returns (fear np, probs np or None)."""
    H.eval(); fs, ps = [], []
    for i in range(0, len(E), bs):
        o = H(E[i:i + bs].to(dev), None if G is None else torch.as_tensor(G[i:i + bs]).to(dev)); fs.append(o["fear"].cpu().numpy()); ps.append(o["probs"].cpu().numpy() if "probs" in o else None)
    return np.concatenate(fs).clip(0, 3), (np.concatenate(ps) if ps and ps[0] is not None else None)

class V10:
    def __init__(s, path=MODEL_DIR, dev=DEV):
        from sentence_transformers import SentenceTransformer
        path = Path(path); s.cfg = json.load(open(path / "config.json")); s.dev = dev
        s.m = SentenceTransformer(str(path / "encoder"), device=dev); s.m.eval(); c = s.cfg
        s.H = Heads(c["d"], c["head"], c["heads"], len(c["gnames"]) if c["gram"] else 0); s.H.load_state_dict(torch.load(path / "heads.pt", map_location="cpu")); s.H.to(dev).eval()
        s.mu, s.sd = np.array(c["gmu"], dtype="float32"), np.array(c["gsd"], dtype="float32")
        cal = path / "calibration.json"; s.cal = json.load(open(cal)) if cal.exists() else None
    @torch.no_grad()
    def predict(s, texts, parts=False):
        texts = list(texts); E = embed_all(s.m, texts, dev=s.dev); G = ((gram_feats(texts)[s.cfg["gnames"]].values - s.mu) / s.sd).astype("float32") if s.cfg["gram"] else None
        f, p = head_all(s.H, E, G, dev=s.dev)
        if s.cal: f = np.interp(f, s.cal["x"], s.cal["y"])
        return (f, p) if parts else f
_cache = {}
def predict(texts, parts=False, path=MODEL_DIR):
    """Fear 0-3 for each text (np array) with the saved v10 model; parts=True also returns the 18 part probabilities (severity4|status5|certainty4|closeness3|literal2)."""
    if str(path) not in _cache: _cache[str(path)] = V10(path)
    return _cache[str(path)].predict(texts, parts)
