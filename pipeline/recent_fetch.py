"""Recent collection 2022-01 .. 2026-10 (27 Oct 2026). Extends the 2003-21 archives with the same designs. Safe to rerun: cached, stops at each API's daily budget.
  NYT: full monthly Archive API (same cache folder data/raw/outlets/nyt/). Oct 2026 is partial and is refetched on every run until November.
  Guardian: A. one random day per month (seed 8), B. the 10 threat-topic tags by year, C. yearly agenda counts.
Output: data/processed/abc_all/recent_nyt.parquet, recent_guardian.parquet, recent_guardian_counts.csv, recent_fetch.DONE
Log: data/processed/abc_all/recent_fetch.log"""
import sys, json, gzip, time, random, calendar, datetime as dt, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import outlets_fetch as F
from pathlib import Path
R = Path(__file__).resolve().parents[1]; O = R / "data/processed/abc_all"; DONE = O / "recent_fetch.DONE"
LOG = open(O / "recent_fetch.log", "a")
def log(s):
    s = time.strftime("%Y-%m-%d %H:%M:%S ") + s; print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
TAGS = ["world/natural-disasters", "politics/terrorism", "uk/ukcrime", "us-news/us-crime", "society/health", "world/coronavirus-outbreak",
        "business/economics", "business/global-economy", "society/mental-health", "environment/climate-crisis"]
GURL = "https://content.guardianapis.com/search"
MONTHS = [(y, m) for y in range(2022, 2027) for m in range(1, 13) if (y, m) <= (2026, 10)]
YEARS = range(2022, 2027)
if DONE.exists(): sys.exit(0)
F.load_env()

def nyt():
    cl = F.Client("nyt", dry=False); log(f"NYT start; calls today {cl.today()}")
    for y, m in MONTHS:
        p = cl.root / f"{y}-{m:02d}.json.gz"
        if (y, m) == (2026, 10) and p.exists() and not (O / "recent_nyt.parquet").exists(): p.unlink()  # partial month: refresh once per assembly
        cl.cached(p, F.CFG["nyt"]["url"].format(y=y, m=m), {}, None, gz=True)
    rows = []
    for y, m in MONTHS:
        for d in json.load(gzip.open(cl.root / f"{y}-{m:02d}.json.gz"))["response"]["docs"]:
            rows.append(dict(id=d.get("_id") or d.get("uri"), date=(d.get("pub_date") or "")[:10], headline=(d.get("headline") or {}).get("main"),
                             section=d.get("section_name"), news_desk=d.get("news_desk"), type_of_material=d.get("type_of_material"),
                             document_type=d.get("document_type"), keywords=";".join(f'{k.get("name")}:{k.get("value")}' for k in (d.get("keywords") or []))))
    df = pd.DataFrame(rows).drop_duplicates("id"); df.to_parquet(O / "recent_nyt.parquet"); log(f"NYT done: {len(df):,} articles")

def guardian():
    cl = F.Client("guardian", dry=False); C = cl.root; log(f"Guardian start; calls today {cl.today()}")
    rng = random.Random(8); days = [dt.date(y, m, rng.randint(1, calendar.monthrange(y, m)[1])) for y, m in MONTHS]
    days = [d for d in days if d < dt.date.today()]
    for d in days:
        page, pages = 1, 1
        while page <= pages:
            j, _ = cl.cached(C / "alltopics_recent" / f"{d}_p{page}.json", GURL, {"from-date": str(d), "to-date": str(d), "page-size": 200, "page": page,
                             "show-fields": "headline", "show-tags": "tone,keyword", "order-by": "oldest"}, None)
            pages = j["response"].get("pages", 1); page += 1
    log("A. random days complete")
    for t in TAGS:
        safe = t.replace("/", "_")
        for y in YEARS:
            page, pages = 1, 1
            while page <= pages:
                j, _ = cl.cached(C / "tags_recent" / safe / f"{y}_p{page:04d}.json", GURL, {"tag": t, "from-date": f"{y}-01-01", "to-date": f"{y}-12-31",
                                 "page-size": 200, "page": page, "show-fields": "headline", "show-tags": "tone,keyword", "order-by": "oldest"}, None)
                pages = j["response"].get("pages", 1); page += 1
        log(f"B. {t}: complete (calls today {cl.today()})")
    for y in YEARS:
        for t in [None] + TAGS:
            p = {"from-date": f"{y}-01-01", "to-date": f"{y}-12-31", "page-size": 1}
            if t: p["tag"] = t
            cl.cached(C / "counts_recent" / f"{y}_{(t or 'ALL').replace('/', '_')}.json", GURL, p, None)
    log("C. counts complete")
    rows = []
    def add(f, src):
        for x in json.load(open(f))["response"].get("results", []):
            tags = x.get("tags", [])
            rows.append(dict(id=x["id"], source=src, date=x["webPublicationDate"][:10], headline=(x.get("fields") or {}).get("headline") or x.get("webTitle"),
                             section=x.get("sectionName"), type=x.get("type"), pillar=x.get("pillarName"),
                             tone=";".join(g["id"].split("/")[-1] for g in tags if g.get("type") == "tone"),
                             keywords=";".join(g["id"] for g in tags if g.get("type") == "keyword")))
    for f in sorted((C / "alltopics_recent").glob("*.json")): add(f, "random_day")
    for t in TAGS:
        for f in sorted((C / "tags_recent" / t.replace("/", "_")).glob("*.json")): add(f, t)
    df = pd.DataFrame(rows); df.to_parquet(O / "recent_guardian.parquet")
    cnt = [dict(year=int(f.stem[:4]), tag=f.stem[5:], total=json.load(open(f))["response"]["total"]) for f in (C / "counts_recent").glob("*.json")]
    pd.DataFrame(cnt).to_csv(O / "recent_guardian_counts.csv", index=False)
    log(f"Guardian done: {len(df):,} rows ({df.id.nunique():,} unique articles)")

try:
    if not (O / "recent_nyt.parquet").exists(): nyt()
    guardian()
    DONE.write_text(time.strftime("%Y-%m-%d %H:%M:%S")); log("ALL DONE")
except F.StopFetch as e:
    log(f"paused: {e}")
