"""Guardian all-topic sample (22 Oct): one random day per month, 2003-2021, every article that day (all sections, all types).
Reuses outlets_fetch.Client (key from .env, never printed; 1.1 s between calls; daily cap 450; retries; atomic cache -> resumable).
Log: data/processed/abc_all/guardian_fetch.log   Output: data/processed/abc_all/guardian_alltopics.parquet"""
import sys, json, time, random, datetime as dt, calendar, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import outlets_fetch as F
from pathlib import Path
R = Path(__file__).resolve().parents[1]; O = R / "data/processed/abc_all"; LOG = open(O / "guardian_fetch.log", "a")
def log(s):
    s = time.strftime("%H:%M:%S ") + s; print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
F.load_env(); cl = F.Client("guardian", dry=False); C = cl.root / "alltopics"
rng = random.Random(7); days = [dt.date(y, m, rng.randint(1, calendar.monthrange(y, m)[1])) for y in range(2003, 2022) for m in range(1, 13)]
log(f"{len(days)} sampled days; calls used today so far: {cl.today()}")
rows = []
try:
    for k, d in enumerate(days):
        page, pages = 1, 1
        while page <= pages:
            p = {"from-date": str(d), "to-date": str(d), "page-size": 200, "page": page, "show-fields": "headline", "order-by": "oldest"}
            j, fresh = cl.cached(C / f"{d}_p{page}.json", F.CFG["guardian"]["url"], p, None)
            resp = j["response"]; pages = resp.get("pages", 1)
            for x in resp.get("results", []):
                rows.append(dict(date=str(d), headline=(x.get("fields") or {}).get("headline") or x.get("webTitle"), section=x.get("sectionName"), type=x.get("type"), pillar=x.get("pillarName")))
            page += 1
        if k % 12 == 11: log(f"{d.year} done  ({k + 1}/{len(days)} days, {len(rows):,} articles, calls today {cl.today()})")
except F.StopFetch as e:
    log(f"STOPPED: {e}")
pd.DataFrame(rows).to_parquet(O / "guardian_alltopics.parquet"); log(f"saved {len(rows):,} articles from cached days")
