"""Subject column (27 Oct 2026, PLAN-v12 "After"): 10 subjects from the outlets' own tags -> silver labels -> logistic head on the live fear_v7 (v7e) encoder,
so Noesis gets subject from the same embedding as fear (no extra model).
Sources: NYT 2003-21 (1 in 6 articles) + NYT 2022-26 (all) via section + Subject keywords; Guardian (random-day, targeted, recent) via section + keyword tags; HuffPost via category.
Rules: first match wins, in SUBJ order below; section rules for sport/culture come first. Unmatched rows are not used.
Test: 10% held out (stratified), per source; plus ABC check set (data/eval/subject_abc/) labelled blind by Claude.
Outputs: data/processed/subject/{silver.parquet, emb.npy, results.json, subject_head.json}"""
import sys, json, gzip, glob, re, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
R = Path(__file__).resolve().parents[1]; A = R / "data/processed/abc_all"; O = R / "data/processed/subject"; O.mkdir(exist_ok=True)
NOE = Path(__file__).resolve().parents[1] / "noesis_fear"; sys.path.insert(0, str(NOE))
SUBJ = ["sport", "culture_lifestyle", "war_terror", "disaster_accident", "crime_justice", "health", "environment", "economy", "science_tech", "politics"]
SEC = {"sport": r"^(sports?|football|cricket|rugby union|tennis|golf|cycling|formula one|boxing|us sports|athletics|olympics)",
       "culture_lifestyle": r"^(arts|books|movies|theater|food|style|travel|real estate|magazine|t magazine|fashion|music|television|television & radio|culture|life and style|film|stage|art and design|dining|home|crosswords|gameplay|games|weddings|lifestyle)"}
KW = {"war_terror": r"terror|war and armed|invasion|israel-gaza|israel-hamas|attack on iran|military|missile|hamas|hezbollah|islamic state|isis|al qaeda|taliban|ukraine|afghanistan war|iraq war|syria|insurgen|drone",
      "disaster_accident": r"earthquake|flood|hurricane|tornado|wildfire|brush fires|storm|tsunami|volcano|landslide|accident|crash|fires and firefighters|explosion|drought|heat wave|weather|snow|natural-disasters|disasters",
      "crime_justice": r"murder|homicide|shooting|crime|sex crimes|kidnap|robbery|assault|gun|prison|fraud|police|ukcrime|us-crime|criminal",
      "health": r"coronavirus|covid|vaccin|disease|epidemic|cancer|health|medicine|hospital|mental|opioid|drug abuse|measles|ebola|influenza|nhs",
      "environment": r"global warming|climate|environment|greenhouse|pollution|wildlife|endangered|renewable|conservation",
      "economy": r"econom|stock|market|tariff|international trade|bank|labor|jobs|unemployment|inflation|prices|oil|recession|company reports|business|interest rates|budget",
      "science_tech": r"artificial intelligence|computers|internet|social media|technology|science|space|astronomy|research|genetics",
      "politics": r"politics|election|presidential|congress|parliament|immigration|deportation|legislation|supreme court|international relations|primaries|labour|conservatives|senate|house of representatives"}
SEC2 = {"war_terror": r"^(defense|military)", "health": r"^(health|well|society)$", "environment": r"^(climate|environment)", "economy": r"^(business|business day|your money|money|economy)",
        "science_tech": r"^(technology|science)", "politics": r"^(politics|u\.s\. politics|us politics)", "crime_justice": r"^(law)$"}
HP = {"SPORTS": "sport", "CRIME": "crime_justice", "POLITICS": "politics", "BUSINESS": "economy", "MONEY": "economy", "WELLNESS": "health", "HEALTHY LIVING": "health",
      "SCIENCE": "science_tech", "TECH": "science_tech", "GREEN": "environment", "ENVIRONMENT": "environment", **{c: "culture_lifestyle" for c in
      ["ENTERTAINMENT", "ARTS", "ARTS & CULTURE", "CULTURE & ARTS", "COMEDY", "STYLE", "STYLE & BEAUTY", "FOOD & DRINK", "TASTE", "TRAVEL", "HOME & LIVING", "WEDDINGS", "MEDIA"]}}
def label(section, keywords):
    s = (section or "").strip().lower(); k = (keywords or "").lower()
    for sub in ("sport", "culture_lifestyle"):
        if re.search(SEC[sub], s): return sub
    if re.search(r"obituar|corrections|briefing|podcasts|learning network", s): return None
    for sub in ["war_terror", "disaster_accident", "crime_justice", "health", "environment", "economy", "science_tech", "politics"]:
        if k and re.search(r"\b(?:" + KW[sub] + ")", k): return sub   # word-start boundary (fix 27 Oct: "isis" matched "climate-crisis", "oil" matched "soil")
        if re.search(SEC2.get(sub, r"^\b$"), s): return sub
    return None
T0 = time.time(); rows = []
# NYT: keep "Subject:" and "Organization:" keyword values (drop Person/Location to avoid geography-as-topic)
def nyt_kw(d): return ";".join(str(x.get("value")) for x in (d.get("keywords") or []) if x.get("name") in ("subject", "Subject", "organizations", "Organization"))
for f in sorted(glob.glob(str(R / "data/raw/outlets/nyt/20*.json.gz"))):
    yr = int(Path(f).name[:4]); docs = json.load(gzip.open(f))["response"]["docs"]
    for i, d in enumerate(docs):
        if yr <= 2021 and i % 6: continue
        if (d.get("document_type") or "").lower() != "article" or d.get("type_of_material") in ("Paid Death Notice", "Correction", "Obituary (Obit)"): continue
        h = ((d.get("headline") or {}).get("main") or "").strip()
        if len(h.split()) >= 4: rows.append(dict(src="nyt", date=(d.get("pub_date") or "")[:10], text=h, subject=label(d.get("section_name"), nyt_kw(d))))
print(f"nyt rows {len(rows)} {time.time() - T0:.0f}s", flush=True)
for f, kwcol in (("guardian_alltopics", None), ("guardian_targeted", "keywords"), ("recent_guardian", "keywords")):
    g = pd.read_parquet(A / f"{f}.parquet")
    for r in g.itertuples():
        kw = (getattr(r, kwcol) or "") if kwcol else ""
        if kwcol and f == "guardian_targeted": kw += ";" + r.query_tag
        rows.append(dict(src="guardian", date=r.date, text=r.headline, subject=label(r.section, kw)))
hp = pd.read_json(R / "data/external/huffpost/data.json", lines=True)
rows += [dict(src="huffpost", date=str(r.date)[:10], text=r.headline, subject=HP.get(r.category)) for r in hp.itertuples()]
D = pd.DataFrame(rows); D = D[D.text.notna() & (D.text.str.split().str.len() >= 4)]; D["k"] = D.text.str.lower(); D = D.drop_duplicates("k")
print("all rows", len(D), "labelled", D.subject.notna().sum()); print(pd.crosstab(D.subject.fillna("-"), D.src))
L = D[D.subject.notna()].copy()
# balance: at most 15k per subject per source (seed 0)
L = L.groupby(["src", "subject"], group_keys=False).apply(lambda g: g.sample(min(len(g), 15000), random_state=0)).reset_index(drop=True); L.to_parquet(O / "silver.parquet")
print("training pool", len(L), L.subject.value_counts().to_dict(), flush=True)
import fear_v7
m = fear_v7._load(); t1 = time.time(); X = m["enc"].encode(L.text.tolist(), batch_size=256, normalize_embeddings=True, show_progress_bar=False); np.save(O / "emb.npy", X); print(f"encoded {len(L)} in {time.time() - t1:.0f}s", flush=True)
tr, te = train_test_split(np.arange(len(L)), test_size=.1, stratify=L.src + L.subject, random_state=0)
clf = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced").fit(X[tr], L.subject.values[tr]); P = clf.predict_proba(X[te]); pred = clf.classes_[P.argmax(1)]; conf = P.max(1)
y = L.subject.values[te]; s = L.src.values[te]; RES = dict(overall=round(float((pred == y).mean()), 3), by_src={k: round(float((pred[s == k] == y[s == k]).mean()), 3) for k in np.unique(s)},
    by_subject={k: round(float((pred[y == k] == k).mean()), 3) for k in clf.classes_}, conf_ge_60=dict(share=round(float((conf >= .6).mean()), 3), acc=round(float((pred[conf >= .6] == y[conf >= .6]).mean()), 3)))
cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred")); print(json.dumps(RES, indent=1)); print(cm.to_string())
clf = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced").fit(X, L.subject.values)
json.dump(dict(classes=list(clf.classes_), coef=[[round(float(x), 6) for x in r] for r in clf.coef_], intercept=[float(x) for x in clf.intercept_], heldout=RES,
               note="logistic on fear_v7 (v7e) encoder embeddings; silver labels from NYT/Guardian tags + HuffPost categories; build_subject.py 27 Oct 2026"), open(O / "subject_head.json", "w"))
json.dump(dict(heldout=RES, confusion=cm.to_dict()), open(O / "results.json", "w"), indent=1); print(f"done {time.time() - T0:.0f}s")
