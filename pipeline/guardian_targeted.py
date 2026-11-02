"""Guardian targeted collection (23 Oct 2026). Safe to run any number of times: everything is cached, and each run stops at the daily budget.
Limits (read from the API's own headers on 23 Oct): 500 calls/day, 60/minute. Our client: 1.1 s between calls (~55/min), stops at 450/day, backs off on 429.
Work, in order:
  A. finish the random-day all-topic sample (guardian_alltopics.py days)
  B. complete archives 2003-2021 for the threat-topic tags below (page-size 200, with tone + keyword tags)
  C. agenda counts: articles per tag per year, and all articles per year (page-size 1 each)
When everything is cached it assembles data/processed/abc_all/guardian_targeted.parquet + guardian_counts.csv, writes DONE, and later runs exit at once.
Log: data/processed/abc_all/guardian_targeted.log"""
import sys, json, time, random, calendar, datetime as dt, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import outlets_fetch as F
from pathlib import Path
R = Path(__file__).resolve().parents[1]; O = R / "data/processed/abc_all"; DONE = O / "guardian_targeted.DONE"
LOG = open(O / "guardian_targeted.log", "a")
def log(s):
    s = time.strftime("%Y-%m-%d %H:%M:%S ") + s; print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
TAGS = ["world/natural-disasters", "politics/terrorism", "uk/ukcrime", "us-news/us-crime", "society/health", "world/coronavirus-outbreak",
        "business/economics", "business/global-economy", "society/mental-health", "environment/climate-crisis"]
URL = "https://content.guardianapis.com/search"; FROM, TO = "2003-01-01", "2021-12-31"
if DONE.exists(): sys.exit(0)
F.load_env(); cl = F.Client("guardian", dry=False); C = cl.root
log(f"run start; calls already used today: {cl.today()}")
try:
    # A. random-day sample (same days and cache paths as guardian_alltopics.py)
    rng = random.Random(7); days = [dt.date(y, m, rng.randint(1, calendar.monthrange(y, m)[1])) for y in range(2003, 2022) for m in range(1, 13)]
    for d in days:
        page, pages = 1, 1
        while page <= pages:
            j, _ = cl.cached(C / "alltopics" / f"{d}_p{page}.json", URL, {"from-date": str(d), "to-date": str(d), "page-size": 200, "page": page, "show-fields": "headline", "order-by": "oldest"}, None)
            pages = j["response"].get("pages", 1); page += 1
    log("A. random-day sample complete")
    # B. complete tag archives
    # The API refuses deep paging (~35k results). Tags with <= 150 pages over the whole range are fetched in one sweep (tags/<tag>/);
    # bigger tags are fetched one year at a time (tags_y/<tag>/<year>_pNNNN.json) so every query stays shallow.
    def sweep(folder, t, a, b, prefix=""):
        page, pages = 1, 1
        while page <= pages:
            j, fresh = cl.cached(folder / f"{prefix}p{page:04d}.json", URL, {"tag": t, "from-date": a, "to-date": b, "page-size": 200, "page": page,
                                 "show-fields": "headline", "show-tags": "tone,keyword", "order-by": "oldest"}, None)
            pages = j["response"].get("pages", 1)
            if fresh and page % 25 == 0: log(f"B. {t} {prefix}: page {page}/{pages}  (calls today {cl.today()})")
            page += 1
        return pages
    for t in TAGS:
        safe = t.replace("/", "_")
        first, _ = cl.cached(C / "tags" / safe / "p0001.json", URL, {"tag": t, "from-date": FROM, "to-date": TO, "page-size": 200, "page": 1,
                             "show-fields": "headline", "show-tags": "tone,keyword", "order-by": "oldest"}, None)
        if first["response"].get("pages", 1) <= 150: n = sweep(C / "tags" / safe, t, FROM, TO); log(f"B. {t}: complete ({n} pages, one sweep)")
        else:
            for y in range(2003, 2022): sweep(C / "tags_y" / safe, t, f"{y}-01-01", f"{y}-12-31", prefix=f"{y}_")
            log(f"B. {t}: complete (by year)")
    # C. agenda counts per year
    for y in range(2003, 2022):
        for t in [None] + TAGS:
            p = {"from-date": f"{y}-01-01", "to-date": f"{y}-12-31", "page-size": 1}
            if t: p["tag"] = t
            cl.cached(C / "counts" / f"{y}_{(t or 'ALL').replace('/', '_')}.json", URL, p, None)
    log("C. counts complete")
    # assemble
    rows = []
    for t in TAGS:
        safe = t.replace("/", "_"); byyear = sorted((C / "tags_y" / safe).glob("*_p*.json"))
        files = byyear if byyear else sorted((C / "tags" / safe).glob("p*.json"))
        for f in files:
            for x in json.load(open(f))["response"].get("results", []):
                tags = x.get("tags", []); tone = [g["id"].split("/")[-1] for g in tags if g.get("type") == "tone"]
                rows.append(dict(id=x["id"], query_tag=t, date=x["webPublicationDate"][:10], headline=(x.get("fields") or {}).get("headline") or x.get("webTitle"),
                                 section=x.get("sectionName"), type=x.get("type"), pillar=x.get("pillarName"), tone=";".join(tone),
                                 keywords=";".join(g["id"] for g in tags if g.get("type") == "keyword")))
    pd.DataFrame(rows).to_parquet(O / "guardian_targeted.parquet")
    cnt = [dict(year=int(f.stem[:4]), tag=f.stem[5:], total=json.load(open(f))["response"]["total"]) for f in (C / "counts").glob("*.json")]
    pd.DataFrame(cnt).to_csv(O / "guardian_counts.csv", index=False)
    DONE.write_text(time.strftime("%Y-%m-%d %H:%M:%S")); log(f"ALL DONE: {len(rows):,} tagged articles saved; counts saved")
except F.StopFetch as e:
    log(f"paused: {e}")
