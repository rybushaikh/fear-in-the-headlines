"""v10 synthetic minimal-edit suite (plan PLAN-v10-2026-10-23, frozen): T1 resolve, T2 history, T3 negate, T4 hedge, T5 figurative, T6 scale, T7 closeness, T8 counter.
Deterministic rule-based edits (regex + spaCy en_core_web_sm) of REAL labelled headlines: disaster (rating_v2, relabels applied, off-topic dropped) + everyday (rating_general_private).
Split logic as train_v9.py: bases in v9's TEST rows (11 disaster events seed 41, 400 everyday seed 81) feed split=test only; other bases -> train/dev 85/15 by md5 hash of id (seed 1023).
Fill lists below are test-generators, not model features: every list is cut in two halves by index parity - even indices only for train+dev, odd indices only for test.
Caps per edit: test 120, dev 40, train 400. An edit is emitted only when its rule applies cleanly; otherwise skipped.
Outputs: data/eval/v10_private/synthetic.parquet (all), synth_check_key.parquet; human check batch data/eval/v10_synth_check/batches/batch_01.jsonl (150 from test, ids c0000.., text only).
Log: data/processed/events/v10/make_synthetic.log"""
import os, re, json, time, shutil, hashlib, numpy as np, pandas as pd, spacy
from functools import lru_cache
from pathlib import Path
R = Path(__file__).resolve().parents[1]; SEED = 1023; E = R / "data/eval"
(R / "data/processed/events/v10").mkdir(parents=True, exist_ok=True); LOG = open(R / "data/processed/events/v10/make_synthetic.log", "w")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

# ---------------- fill lists (halved by parity: even -> train/dev, odd -> test) ----------------
FOUND_SAFE = ["found safe", "rescued", "found alive", "located safe and well", "found unharmed", "located", "pulled out alive", "safe and well"]   # T1: replaces 'missing'
ARRESTED = ["arrested", "caught", "captured", "in custody", "tracked down", "charged", "found and arrested", "surrenders"]                          # T1: after a hunted suspect
CONTAINED = ["contained", "under control", "extinguished", "brought under control", "no longer a threat", "fully contained", "largely contained", "finally controlled"]  # T1: replaces burning/spreading
FREED = ["freed", "rescued", "pulled free", "freed safely"]                                                                                           # T1: replaces 'trapped'
WARN_OFF = ["cancelled", "lifted", "called off", "downgraded", "withdrawn", "all clear given", "scrapped", "ended"]                                  # T1: after 'warning/alert/emergency'
SIEGE_END = ["ends peacefully", "ends with no injuries", "over", "resolved"]                                                                           # T1: after 'siege/standoff continues'
OLD_YEARS = ["1967", "1974", "1983", "1987", "1989", "1991", "1994", "1996", "1998", "1999", "1952", "1942"]                                         # T2
HISTORY_FRAMES = ["{y} anniversary: {b}", "on this day in {y}: {b}", "remembering {y}: {b}", "looking back to {y}: {b}", "from the archive, {y}: {b}", "flashback to {y}: {b}",
                  "memorial service marks {y} tragedy: {b}", "decades on, we recall {y}: {b}"]                                                             # T2
NEG_HARM = ["no one hurt", "no injuries", "nobody killed", "no deaths reported", "no one injured", "everyone safe"]                                   # T3: replaces '<subject> dead/killed'
NEG_THREAT = ["no threat to {y} from {x}", "{x} poses no threat to {y}"]                                                                              # T3: from 'X threatens Y'
MODALS = ["could", "might", "may", "may well"]                                                                                                         # T4
HEDGE_Q = ["{b}?", "is it true? {b}", "{b}, but is it confirmed?", "unconfirmed reports: {b}"]                                                          # T4 question / unconfirmed forms
TEAMS = ["swans", "hawks", "blues", "cats", "tigers", "eagles", "demons", "lions", "storm", "broncos", "roosters", "bulldogs", "sharks", "knights", "raiders", "panthers"]   # T5
ROUNDS = ["grand final", "semi-final", "round one", "quarter-final", "preliminary final", "opening round", "derby", "finals series"]                   # T5
COMPANIES = ["telstra", "qantas", "woolworths", "bunnings", "coles", "optus", "harvey norman", "virgin", "jb hi-fi", "bhp", "anz", "nine"]             # T5 business
MARKETS = ["mobile", "airfare", "grocery", "fuel", "broadband", "energy"]                                                                              # T5 business
FIG_VERBS = [("slaughter", "slaughtered"), ("crush", "crushed"), ("destroy", "destroyed"), ("demolish", "demolished"), ("batter", "battered"), ("hammer", "hammered"),
             ("blast", "blasted"), ("smash", "smashed"), ("devastate", "devastated"), ("kill", "killed off"), ("attack", "attacked"), ("bomb", "bombed"), ("torch", "torched"),
             ("hit", "hit"), ("thrash", "thrashed"), ("shoot", "shot down"), ("sink", "sank"), ("drown", "drowned")]                                    # T5: (harm lemma, past form)
SCALE_NUMS = ["40", "45", "60", "75", "90", "120", "150", "180", "220", "250", "280", "300"]                                                             # T6
DISTANT = [("peru", "peruvian"), ("chile", "chilean"), ("kenya", "kenyan"), ("ukraine", "ukrainian"), ("nepal", "nepali"), ("bolivia", "bolivian"), ("norway", "norwegian"),
           ("ghana", "ghanaian"), ("mongolia", "mongolian"), ("poland", "polish"), ("iceland", "icelandic"), ("morocco", "moroccan")]                   # T7: (country, demonym)
COUNT_WORDS = ["two", "three", "four", "five", "six", "seven", "eight", "nine"]                                                                        # T8 when base has no count
COUNTER_FRAMES = ["{r} declared over but {n} dead", "search ends as bodies found after {r}", "{r} now over, {n} confirmed dead", "all clear given but {n} dead after {r}",
                  "{r} over: bodies recovered, {n} dead", "emergency lifted as death toll from {r} reaches {n}", "{r} over with {n} dead", "rescue called off after {r}, {n} dead"]   # T8

H = lambda L, sp: L[1::2] if sp == "test" else L[0::2]
hh = lambda *k: int(hashlib.md5("|".join(map(str, (SEED,) + k)).encode()).hexdigest(), 16)
choose = lambda L, sp, *k: H(L, sp)[hh(*k) % len(H(L, sp))]
nlp = spacy.load("en_core_web_sm", disable=["ner"])
@lru_cache(None)
def doc(t): return nlp(t)
def np_ok(s, maxw=6):   # short noun phrase with no verb/aux
    return 0 < len(s.split()) <= maxw and not any(t.pos_ in ("VERB", "AUX") for t in doc(s))

# ---------------- bases (same frame/splits as train_v9.py) ----------------
K = pd.read_parquet(R / "data/eval/rating_v2/sample_key.parquet"); L = pd.read_csv(R / "data/eval/rating_v2/labels_all.csv")
Dd = K.merge(L, on="id")[["id", "event", "text", "fear"]].assign(src="disaster")
Gk = pd.read_parquet(R / "data/eval/rating_general_private/sample_key.parquet"); Gl = pd.read_csv(R / "data/eval/rating_general_private/labels_all.csv")
Dg = Gk.merge(Gl, on="id")[["id", "text", "fear"]].assign(src="everyday", event=None)
D = pd.concat([Dd, Dg], ignore_index=True); ev = sorted(Dd.event.unique())
te_d = ((D.src == "disaster") & D.event.isin(set(np.random.default_rng(41).choice(ev, 11, replace=False)))).values
gi = D.index[D.src == "everyday"].to_numpy(); te_g = D.index.isin(np.random.default_rng(81).choice(gi, 400, replace=False))
RL = pd.read_parquet(R / "data/eval/relabel/relabel_decisions.parquet").set_index("id")
D["fear"] = D.fear.astype(float); hit = D.id.isin(RL.index); D.loc[hit, "fear"] = D.loc[hit, "id"].map(RL.new_fear).astype(float)
offt = D.id.isin(RL.index[RL.about_event == "no"]).values
D["test_side"] = te_d | te_g; D = D[~offt].copy(); D["text"] = D.text.str.strip().str.lower()
D["split"] = [("test" if t else ("dev" if hh("split", i) % 100 >= 85 else "train")) for i, t in zip(D.id, D.test_side)]
log(f"bases {len(D)}: {D.split.value_counts().to_dict()}; fear dist {D.fear.value_counts().sort_index().to_dict()}")

# ---------------- rules: each returns None or dict(text, expected..., parts) ----------------
DEATH = r"\b(dead|killed|dies|died|die|death|deaths|bodies|body|fatal\w*|murder\w*)\b"
NUMW = {"one": 1, "two": 2, "three": 3, "1": 1, "2": 2, "3": 3}
PLUR = {"person": "people", "man": "men", "woman": "women", "child": "children", "boy": "boys", "girl": "girls", "worker": "workers", "people": "people", "men": "men", "women": "women",
        "children": "children", "australians": "australians", "workers": "workers", "passengers": "passengers", "tourists": "tourists", "climbers": "climbers", "civilians": "civilians", "soldiers": "soldiers"}
HARMP = re.compile(r"^(?P<subj>.+?)\s+(?P<harm>dead|killed|injured|hurt|dies|die|died)\s+(?P<prep>in|after|at|near|during|on|off|following)\s+(?P<rest>.+)$")
KILLS = re.compile(r"^(?P<x>.+?)\s+(?P<harm>kills|injures)\s+(?:at least |over |more than )?(?P<n>\d[\d,]*|one|two|three|four|five|six|seven|eight|nine|ten|dozens|hundreds)(?:\s+(?:people|more))?(?:\s+(?:in|at|near)\s+.+)?$")
NUM = re.compile(r"^(?:at least |over |more than |about |nearly |up to )?(\d[\d,]*|one|two|three|four|five|six|seven|eight|nine|ten|dozens|hundreds|scores|several)$")
ONE = r"(?:an? )?(?:\w+ ){0,2}?(?:man|woman|person|teen|boy|girl|child|driver|tourist|worker|climber|swimmer|cyclist|motorcyclist|pilot|passenger)"
def harm_parse(t):   # -> dict(harm: 'dead'|'hurt', rest (noun phrase), prep, n (count string or None)); None if no clean parse
    m = HARMP.match(t)
    if m and len(m["subj"].split()) <= 5 and np_ok(m["rest"], 5) and (NUM.match(m["subj"]) or np_ok(m["subj"], 5)):
        nm = NUM.match(m["subj"]); n = nm[1] if nm else ("one" if re.fullmatch(ONE, m["subj"]) else None)
        return dict(harm="dead" if m["harm"] in ("dead", "killed", "dies", "die", "died") else "hurt", rest=m["rest"], prep=m["prep"], n=n)
    m = KILLS.match(t)
    if m and np_ok(m["x"], 5): return dict(harm="dead" if m["harm"] == "kills" else "hurt", rest=m["x"], prep="in", n=m["n"])
    return None

EVW = r"attacks?|bombings?|blasts?|crash(?:es)?|collisions?|fires?|bushfires?|floods?|quakes?|earthquakes?|tsunami|shootings?|sieges?|storms?|cyclones?|hurricanes?|typhoons?|collapse|explosions?|outbreak|massacre|avalanche|derailment|sinking"
def event_phrase(t):   # short noun phrase ending in an event word, e.g. 'madrid train blasts'
    m = re.search(rf"\b((?:[a-z0-9'-]+ ){{0,2}}(?:{EVW}))\b", t)
    return m[1] if m and not re.search(r"\b(in|of|the|a|an|after|at|as|and|to|for|by|from|over|on|death|deaths|dead|killed|toll|fears?)\b", m[1]) and np_ok(m[1], 3) else None
def t1(r, sp):
    t, b = r.text, r.fear
    hunt = re.match(r"^(?:police |huge |massive |intense )?(?:hunt for|hunt|search for|manhunt for) (?P<x>.+)$", t)
    if b < 2 or re.search(DEATH + r"|\b(found|safe|contained|arrested|caught|rescued|under control|charged|ends?|devastated|fears?|funeral|mourn\w*|grie\w*|tribute|toll)\b", t): return None
    if not hunt and re.search(r"\b(search|hunt|looking|seek\w*)\b", t): return None
    m = re.search(r"(?<=\w )(?<!goes )(?<!go )(?<!went )(?<!gone )(?:still )?missing(?= (?:in|near|at|off|after|on|from|since|overnight|for|as)\b|$)", t)
    if m and not hunt and not re.search(r"(keeps?|continues?|\d|as|after|to|for)\s*$", t[:m.start()]): return dict(text=t[:m.start()] + choose(FOUND_SAFE, sp, r.id, "t1a") + t[m.end():])
    m = hunt
    if m and np_ok(m["x"], 5) and re.search(r"\b(gunman|gunmen|man|men|suspect|suspects|killer|offender|robber|attacker|escapee|prisoner|fugitive|bomber|youth|teen|thief|rapist|terrorist)\b", m["x"]) and not re.search(r"missing|lost|bodies|victims|survivors", m["x"]):
        return dict(text=f"{m['x']} {choose(ARRESTED, sp, r.id, 't1b')}")
    m = re.search(r"(?<=\w )(?:still )?(?:at large|on the run|on the loose)\b", t)
    if m: return dict(text=t[:m.start()] + choose(ARRESTED, sp, r.id, "t1d") + t[m.end():])
    m = re.search(r"(?<=\w )(?:still )?(burning|raging|spreading|out of control)\b", t)
    if m and re.search(r"\b(fire|fires|bushfire|bushfires|blaze|flames|virus|outbreak|flood|floodwaters)\b", t) and not re.search(r"(keeps?|continues?|starts?|begins?|began|as|could|may|might|fears?|to)\s*$", t[:m.start()]):
        return dict(text=t[:m.start()] + choose(CONTAINED, sp, r.id, "t1c") + t[m.end():])
    m = re.match(r"^(?P<x>.+?) (?:still )?trapped(?: (?:in|under|inside|at) (?P<w>.+))?$", t)
    if m and np_ok(m["x"], 4) and (not m["w"] or np_ok(m["w"], 4)): return dict(text=f"{m['x']} {choose(FREED, sp, r.id, 't1e')}" + (f" from {m['w']}" if m["w"] else ""))
    m = re.match(r"^(?P<x>.+?) (?:still )?(?:stranded|adrift|lost at sea)(?P<rest>.*)$", t)
    if m and np_ok(m["x"], 4) and len(m["rest"].split()) <= 4: return dict(text=f"{m['x']} {choose(FOUND_SAFE, sp, r.id, 't1g')}{m['rest']}")
    m = re.match(r"^(?P<x>.*\b(?:siege|standoff|stand-off|hostage crisis))\s+(?:continues|drags on|enters.*|goes on)$", t)
    if m and np_ok(m["x"], 4): return dict(text=f"{m['x']} {choose(SIEGE_END, sp, r.id, 't1h')}")
    m = re.match(r"^(?P<x>(?:[\w-]+ ){0,3}?(?:fire|bushfire|blaze))s? (?:threatens?|threatening|menaces|nears|approaches|heads? for|bears? down on|closes in on)\b", t)
    if m: return dict(text=f"{m['x']} {choose(CONTAINED, sp, r.id, 't1i')}")
    m = re.match(r"^(?P<x>.*?\b(?:warning|alert|emergency))(?P<rest> issued| declared| in place| remains)?(?P<tail> (?:for|in|across) .+)?$", t)
    if m and np_ok(m["x"], 5) and len(t.split()) <= 8 and not re.search(r"\b(cancel\w*|lifted|downgraded)\b", t):
        return dict(text=f"{m['x']} {choose(WARN_OFF, sp, r.id, 't1f')}{m['tail'] or ''}")
    return None
def t2(r, sp):
    if r.fear < 2 or re.search(r"\b(19|20)\d\d\b|anniversary|remember", r.text): return None
    y = choose(OLD_YEARS, sp, r.id, "t2y"); return dict(text=choose(HISTORY_FRAMES, sp, r.id, "t2f").format(y=y, b=r.text))
def t3(r, sp):
    t = r.text
    if r.fear < 2: return None
    m = harm_parse(t)
    if m: return dict(text=f"{choose(NEG_HARM, sp, r.id, 't3a')} {m['prep']} {m['rest']}")
    ev = event_phrase(t)
    if ev and re.search(DEATH + r"|injur\w*|hurt|wounded|toll", t): return dict(text=f"{choose(NEG_HARM, sp, r.id, 't3c')} in {ev}")
    m = re.match(r"^(?P<x>.+?) threatens? (?P<y>.+)$", t)
    if m and np_ok(m["x"], 4) and np_ok(m["y"], 4): return dict(text=choose(NEG_THREAT, sp, r.id, "t3b").format(x=m["x"], y=m["y"]))
    m = re.match(r"^(?P<x>.+?) cases? (?P<v>confirmed|reported|detected)(?P<rest>.*)$", t)
    if m and np_ok(m["x"], 3): return dict(text=f"no {m['x']} cases {m['v']}{m['rest']}")
    return None
def t4(r, sp):
    t = r.text
    if r.fear < 1 or "?" in t or re.search(r"\b(could|may|might|fears?|warns?|possible|unconfirmed|alleged|claims?)\b", t): return None
    if hh("t4", r.id) % 2:
        d = doc(t); rt = [x for x in d if x.dep_ == "ROOT"][0]
        if (rt.tag_ in ("VBZ", "VBP") or (rt.tag_ == "VBD" and any(c.dep_ == "dobj" for c in rt.children))) and any(c.dep_ == "nsubj" and c.i < rt.i for c in rt.children) \
           and not any(c.dep_ in ("aux", "neg", "auxpass") for c in rt.children) and not any(x.pos_ == "VERB" for x in d[:rt.i]) and rt.lemma_ not in ("say", "be", "have", "do"):
            return dict(text=t[:rt.idx] + f"{choose(MODALS, sp, r.id, 't4m')} {rt.lemma_}" + t[rt.idx + len(rt.text):], cert="possible")
    f = choose(HEDGE_Q, sp, r.id, "t4q"); return dict(text=f.format(b=t), cert="question" if "?" in f else "possible")
FIGW = {l: v for l, v in FIG_VERBS}; TRIG = r"dead|dies|died|killed|injur|hurt|crash|fire|flood|storm|attack|stab|shot|shoot|explo|collapse|quake|death|murder|bomb|blast|drown"
def t5(r, sp):
    t = r.text
    if r.fear < 1 or not re.search(TRIG, t): return None
    allowed = H(FIG_VERBS, sp); alw = dict(allowed); lem = [x.lemma_ for x in doc(t) if x.lemma_ in FIGW]
    base_v = next((l for l in lem if l in alw), None); l, past = (base_v, alw[base_v]) if base_v else allowed[hh("t5v", r.id) % len(allowed)]
    if hh("t5k", r.id) % 5:
        T = H(TEAMS, sp); a = T[hh("t5a", r.id) % len(T)]; bb = T[(hh("t5a", r.id) + 1 + hh("t5b", r.id) % (len(T) - 1)) % len(T)]
        return dict(text=f"{a} {past} {bb} in {choose(ROUNDS, sp, r.id, 't5r')}", verb=l)
    return dict(text=f"{choose(COMPANIES, sp, r.id, 't5c')} {past} rivals in {choose(MARKETS, sp, r.id, 't5m')} price war", verb=l)
SC = re.compile(r"\b(one|two|three|1|2|3)\s+(?:(person|man|woman|child|boy|girl|worker|people|men|women|children|australians|workers|passengers|tourists|climbers|civilians|soldiers)\s+)?(dead|killed|injured|hurt)\b")
SC2 = re.compile(r"\b(kills|injures) (one|two|three|1|2|3)\b(?! (?:years?|months?))")
SC4 = re.compile(r"\b(several|multiple|a few)\s+(dead|killed|injured|hurt)\b")
SC5 = re.compile(r"\bdeath toll (rises|climbs|grows|mounts|rising)\b(?! (?:to|above|past|beyond|by))")
SC3 = re.compile(rf"^(?P<s>{ONE})\s+(?P<v>dies|died|killed|dead)(?P<rest>.*)$")
def t6(r, sp):
    t = r.text
    if r.fear < 1: return None
    n = choose(SCALE_NUMS, sp, r.id, "t6"); m = SC.search(t)
    if m:
        noun = (PLUR[m[2]] + " ") if m[2] else ""
        return dict(text=t[:m.start()] + f"{n} {noun}{m[3]}" + t[m.end():], death=m[3] in ("dead", "killed"))
    m = SC2.search(t)
    if m: return dict(text=t[:m.start(2)] + n + t[m.end(2):], death=m[1] == "kills")
    m = SC4.search(t)
    if m: return dict(text=t[:m.start()] + f"{n} {m[2]}" + t[m.end():], death=m[2] in ("dead", "killed"))
    m = SC5.search(t)
    if m: return dict(text=t[:m.end()] + f" to {n}" + t[m.end():], death=True)
    m = SC3.match(t)
    if m: return dict(text=f"{n} people {'die' if m['v'] == 'dies' else m['v']}{m['rest']}", death=True)
    return None
AUS = ["new south wales", "south australia", "western australia", "northern territory", "gold coast", "sunshine coast", "central coast", "blue mountains", "hunter valley", "adelaide hills", "snowy mountains",
       "sydney", "melbourne", "brisbane", "perth", "adelaide", "hobart", "darwin", "canberra", "newcastle", "geelong", "wollongong", "townsville", "cairns", "nsw", "queensland", "victoria", "tasmania"]
AUSR = "|".join(AUS); EVN = r"(floods?|flooding|bushfires?|fires?|storms?|cyclone|earthquake|quake|outbreak|drought|heatwave|crash|blast|explosion|lockdown|measles|virus|siege|shooting|attack|terror|blackout|mine|bus|train|plane|helicopter|ferry|school)"
STILL_AUS = r"\b(australia|australian|aust|qld|vic|tas|sa|wa|nt|act|aussies?|abc)\b"
NOTAFTER = r"(?! (?:cbd|city|airport|hospital|council|police|station|court|suburbs?|university|beach|centre|mp|not|man|woman|teen|boy|girl|zoo|mine|port|harbour|bridge|prison|jail))"
PERS = r"(man|woman|teen|boy|girl|couple|family|mother|father|student|tourist|worker|driver|pilot|doctor|nurse|resident|residents)"
def t7(r, sp):
    t = r.text
    if r.fear < 1: return None
    c, dem = H(DISTANT, sp)[hh("t7", r.id) % len(H(DISTANT, sp))]
    if c in t: return None
    m = re.search(rf"\b(in|near|at|across|around|outside|off|to|from|for|throughout|north of|south of|east of|west of)\s+({AUSR})\b{NOTAFTER}", t)
    if m: new = t[:m.start(2)] + c + t[m.end(2):]
    elif (m := re.match(rf"^({AUSR})\s+(?={EVN}\b)", t)): new = c + " " + t[m.end():]
    elif (m := re.match(rf"^({AUSR})\s+(?={PERS}\b)", t)): new = dem + " " + t[m.end():]
    else: return None
    if re.search(rf"\b({AUSR})\b", new) or re.search(STILL_AUS, new): return None
    return dict(text=new)
def t8(r, sp):
    t = r.text
    if r.fear < 2 or not re.search(DEATH, t): return None
    m = harm_parse(t)
    if m and m["harm"] == "dead" and m["n"] and event_phrase(m["rest"]): ev, n = m["rest"], m["n"]
    else:
        ev = event_phrase(t); k = re.search(r"\b(\d[\d,]*|dozens|hundreds|scores)\s+(?:\w+\s)?(?:dead|killed|dies|died)\b", t)
        if not ev: return None
        n = k[1] if k else choose(COUNT_WORDS, sp, r.id, "t8n")
    return dict(text=choose(COUNTER_FRAMES, sp, r.id, "t8f").format(r=ev, n=n))

EDITS = {  # rule, (exp_min, exp_max) as functions of base fear, dir, parts
 "T1": (t1, lambda b, x: (0, 1), "lower", dict(status="over")),
 "T2": (t2, lambda b, x: (0, 1), "lower", dict(status="history")),
 "T3": (t3, lambda b, x: (0, max(0, b - 1)), "lower", dict(certainty="denied")),
 "T4": (t4, lambda b, x: (0, b), "lower_or_equal", dict()),
 "T5": (t5, lambda b, x: (0, 0), "lower", dict(literal="figurative")),
 "T6": (t6, lambda b, x: ((3 if x.get("death") else b), 3), "higher_or_equal", dict()),
 "T7": (t7, lambda b, x: (0, b), "lower_or_equal", dict(closeness="far")),
 "T8": (t8, lambda b, x: (2, 3), "same_high", dict(severity=3)),
}
CAP = dict(test=120, dev=40, train=400); rows = []
for e, (fn, rng_fn, direction, parts) in EDITS.items():
    for sp in ("train", "dev", "test"):
        c = []
        for r in D[D.split == sp].itertuples():
            try: x = fn(r, sp)
            except Exception as ex: log("rule error", e, r.id, ex); x = None
            if not x or x["text"].strip() == r.text or len(x["text"].split()) < 2: continue
            lo, hi = rng_fn(r.fear, x); p = dict(status=None, certainty=None, closeness=None, literal=None, severity=None); p.update(parts)
            if e == "T4": p["certainty"] = x["cert"]
            if e == "T6" and x.get("death"): p["severity"] = 3
            c.append(dict(edit=e, split=sp, base_id=r.id, base_text=r.text, base_fear=r.fear, text=x["text"].strip(), expected_min=lo, expected_max=hi, expected_dir=direction,
                          exp_status=p["status"], exp_certainty=p["certainty"], exp_closeness=p["closeness"], exp_literal=p["literal"], exp_severity=p["severity"]))
        c = pd.DataFrame(c); n0 = len(c)
        if n0: c = c.drop_duplicates("text"); n0 = len(c)
        if n0 > CAP[sp]: c = c.sample(CAP[sp], random_state=SEED)
        log(f"{e} {sp}: candidates {n0}, kept {len(c)}"); rows.append(c)
S = pd.concat([x for x in rows if len(x)], ignore_index=True)
S["id"] = [f"syn{i:05d}" for i in range(len(S))]; S = S[["id", "split", "edit", "base_id", "base_text", "base_fear", "text", "expected_min", "expected_max", "expected_dir", "exp_status", "exp_certainty", "exp_closeness", "exp_literal", "exp_severity"]]
(E / "v10_private").mkdir(exist_ok=True); S.to_parquet(E / "v10_private/synthetic.parquet")
log("counts split x edit\n" + S.pivot_table(index="edit", columns="split", values="id", aggfunc="count").fillna(0).astype(int).to_string())
assert not set(S[S.split == "test"].base_id) & set(S[S.split != "test"].base_id), "base leak"

# ---------------- human check batch ----------------
T = S[S.split == "test"]; per = {e: (19 if i < 6 else 18) for i, e in enumerate(EDITS)}
C = pd.concat([g.sample(min(per[e], len(g)), random_state=SEED) for e, g in T.groupby("edit")])
if len(C) < 150: C = pd.concat([C, T.drop(C.index).sample(150 - len(C), random_state=SEED)])
C = C.sample(frac=1, random_state=SEED).reset_index(drop=True); C["cid"] = [f"c{i:04d}" for i in range(len(C))]
out = E / "v10_synth_check"; (out / "batches").mkdir(parents=True, exist_ok=True)
pd.DataFrame(dict(id=C.cid, publish_date="", text=C.text)).to_json(out / "batches/batch_01.jsonl", orient="records", lines=True, force_ascii=False)
C.rename(columns={"id": "syn_id"}).rename(columns={"cid": "id"}).to_parquet(E / "v10_private/synth_check_key.parquet")
for f in ("CODEBOOK.md", "anchors.md"): shutil.copy(E / "v10_parts" / f, out / f)
log(f"check batch {len(C)}: {C.edit.value_counts().sort_index().to_dict()}")
pd.set_option("display.width", 250, "display.max_colwidth", 90)
for e in EDITS:
    for r in S[(S.edit == e)].sample(10, random_state=SEED).itertuples(): log(f"  {e} [{r.split}] {r.base_text}  ->  {r.text}   exp {r.expected_min}-{r.expected_max} (base {r.base_fear:.0f})")
