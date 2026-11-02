"""Fetch Guardian + NYT headlines for the 54 kept events (window = onset day 0..29). Cached, resumable, polite.
  venv/bin/python pipeline/outlets_fetch.py --outlet guardian|nyt [--dry-run] [--events id1,id2] [--plan] [--daily-totals]
Keys: .env (GUARDIAN_API_KEY, NYT_API_KEY). Keys missing => dry-run automatically (fixtures, separate cache, nothing live).
Raw caches: data/raw/outlets/<outlet>/ (dry-run: data/raw/outlets_dryrun/). A cached file is never fetched again.
Limits assumed (conservative; see README): Guardian developer key 1 call/s, 500/day; NYT 5 calls/min, 500/day.
Guardian: per event 1 call for the window total (type=article) + paged candidate search (OR of kw terms, headline field);
exact event matching happens locally in outlets_scan.py. NYT: Archive API, one call per month overlapping any window."""
import argparse, datetime as dt, gzip, hashlib, json, os, sys, time
from pathlib import Path

R = Path(__file__).resolve().parents[1]
EVENTS = R / "data/processed/events/v2/events_v2.json"
DROP = set("birdflu_2005 haiti_cholera_2010 mers_2015 thunderasthma_2016 listeria_2018 roleystone_2011 tathra_2018 virginiatech_2007 sandyhook_2012 airfrance447_2009 deepwater_2010 maria_2017".split())
CFG = {  # min seconds between calls, calls allowed per local day (below the published free limits on purpose)
    "guardian": dict(interval=1.1, daily=450, url="https://content.guardianapis.com/search", env="GUARDIAN_API_KEY"),
    "nyt": dict(interval=13.0, daily=450, url="https://api.nytimes.com/svc/archive/v1/{y}/{m}.json", env="NYT_API_KEY"),
}
GUARDIAN_PAGE_SIZE, GUARDIAN_MAX_PAGES = 200, 40       # per-event safety cap (8000 candidates); truncation is flagged in the meta file
MAX_TRIES = 6


# Foreign-outlet supplement (pilot 2026-10-11): the ABC rules name Australian towns ("kinglake"), but the NYT and
# Guardian write "Death Toll in Australian Fires" / "Australia wildfires". For Australian bushfire/flood/cyclone events
# only, add one symmetric alt rule: Australian place context AND the type's generic noun. ABC rules are unchanged.
AU_CONTEXT = ["australia", "australian", "australians", "victoria", "victorian", "queensland", "new south wales", "nsw",
              "tasmania", "tasmanian", "south australia", "western australia", "melbourne", "sydney", "brisbane", "adelaide", "perth"]
TYPE_NOUNS = {"bushfire": ["fire", "fires", "wildfire", "wildfires", "bushfire", "bushfires", "blaze", "blazes", "firestorm"],
              "flood": ["flood", "floods", "flooding", "floodwater", "floodwaters", "deluge"],
              "cyclone": ["cyclone", "cyclones", "storm", "storms"]}


def _foreign_alt(e):
    if e.get("local") == "australia" and e.get("type") in TYPE_NOUNS:
        e = dict(e, alt=list(e.get("alt", [])) + [{"kw": AU_CONTEXT, "require": TYPE_NOUNS[e["type"]], "exclude": e.get("exclude", [])}])
    return e


def load_events(only=None):
    j = json.load(open(EVENTS)); out = [_foreign_alt(e) for e in j["events"] if e["id"] not in DROP]
    if only:
        miss = set(only) - {e["id"] for e in out}
        if miss: sys.exit(f"unknown/dropped event ids: {sorted(miss)}")
        out = [e for e in out if e["id"] in only]
    return out, j["window_days"]


def window(e, W):
    t0 = dt.date.fromisoformat(e["onset"]); return t0, t0 + dt.timedelta(days=W - 1)


def load_env():
    p = R / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ---------- query building (Guardian) ----------
def _forms(w):
    """The Guardian search does not stem (pilot 2026-10-11: 'bushfire' 4 hits vs 'bushfires' 13), so each query
    term also searches its plural/verb forms; a trailing * (prefix rule) gets the common suffixes. The local
    event_scan rules still decide the match, so this only widens the candidate net."""
    pre = w.endswith("*"); w = w.replace("*", "").strip()
    if not w: return []
    if " " in w: return [w, w + "s"] if not w.endswith("s") else [w]
    out = [w, w + "es" if w.endswith(("s", "sh", "ch", "x", "z")) else w + "s"]
    if pre: out += [w + x for x in ("ed", "ing", "er", "ers")]
    return out
def _terms(words):
    out = []
    for w in words:
        for f in _forms(w):
            if f not in out: out.append(f)
    return out
def _or(words): return " OR ".join(f'"{w}"' if " " in w else w for w in _terms(words))
def guardian_query(e):
    """Superset query: per rule, (kw) AND (require) when require has no wildcards (stemming-safe), else kw only; rules OR-ed."""
    parts = []
    for r in [e] + e.get("alt", []):
        rq = r.get("require") or []
        q = f"({_or(r['kw'])})"
        if rq and not any("*" in w for w in rq): q += f" AND ({_or(rq)})"
        parts.append(q)
    return " OR ".join(f"({p})" for p in parts) if len(parts) > 1 else parts[0]


# ---------- transport ----------
class StopFetch(Exception): pass

class Client:
    def __init__(self, outlet, dry):
        self.o, self.dry, self.c = outlet, dry, CFG[outlet]
        self.root = R / "data/raw" / ("outlets_dryrun" if dry else "outlets") / outlet
        self.root.mkdir(parents=True, exist_ok=True)
        self.key = None if dry else os.environ.get(self.c["env"])
        self.last = 0.0; self.calls = 0
        self.cf = self.root / "_calls.json"
        self.log = json.load(open(self.cf)) if self.cf.exists() else {}
    def _count(self):
        d = dt.date.today().isoformat(); self.log[d] = self.log.get(d, 0) + 1; self.calls += 1
        json.dump(self.log, open(self.cf, "w"))
    def today(self): return self.log.get(dt.date.today().isoformat(), 0)

    def fetch(self, url, params, fixture):
        """One HTTP GET (live) or fixture (dry). Returns parsed JSON text. Backoff on 429/5xx/network errors; stops on 401/403/daily cap."""
        if self.dry: return json.dumps(fixture())
        import requests
        if self.today() >= self.c["daily"]: raise StopFetch(f"daily budget {self.c['daily']} calls reached; rerun tomorrow (resumes from cache)")
        params = dict(params); params["api-key"] = self.key
        for i in range(MAX_TRIES):
            wait = self.c["interval"] - (time.time() - self.last)
            if wait > 0: time.sleep(wait)
            self.last = time.time(); self._count()
            try:
                r = requests.get(url, params=params, timeout=(10, 120), headers={"User-Agent": "fear-in-the-headlines/0.1 (academic; polite)"})
            except requests.RequestException as ex:                         # message can contain the URL (key): print the type only
                print(f"  network error {type(ex).__name__}, retry {i+1}", flush=True); time.sleep(min(2 ** i * 5, 120)); continue
            if r.status_code == 200: return r.text
            if r.status_code in (401, 403): raise StopFetch(f"HTTP {r.status_code}: key rejected/forbidden (check .env)")
            if r.status_code == 429 or r.status_code >= 500:
                ra = r.headers.get("Retry-After"); w = float(ra) if ra and ra.replace(".", "").isdigit() else min(2 ** i * 10, 300)
                print(f"  HTTP {r.status_code}, backing off {w:.0f}s (try {i+1}/{MAX_TRIES})", flush=True); time.sleep(w)
                if r.status_code == 429 and i >= 2: raise StopFetch("repeated 429: probably rate/daily limit hit; rerun later (resumes from cache)")
                continue
            raise StopFetch(f"HTTP {r.status_code} (not retried): {r.text[:200]!r}")
        raise StopFetch("gave up after retries; rerun to resume")

    def cached(self, path, url, params, fixture, gz=False):
        """Return (parsed_json, was_fetched). Writes cache atomically."""
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return json.load(gzip.open(path, "rt") if gz else open(path)), False
        txt = self.fetch(url, params, fixture); obj = json.loads(txt)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with (gzip.open(tmp, "wt") if gz else open(tmp, "w")) as f: f.write(txt)
        tmp.replace(path); return obj, True


# ---------- Guardian ----------
def guardian_fixture(e, p):
    """Hand-made response in the documented shape ({"response": {status,total,pages,currentPage,results:[{id,type,sectionName,webPublicationDate,webTitle,fields:{headline}}]}})."""
    t0 = dt.date.fromisoformat(e["onset"]); kw0 = e["kw"][0].replace("*", ""); rq = [w.replace("*", "") for w in (e.get("require") or [])]
    rq0 = rq[0] if rq else ""; ex = [w.replace("*", "") for w in (e.get("exclude") or [])]
    def item(day, title, sec="World news", n=[0]):
        n[0] += 1; d = t0 + dt.timedelta(days=day)
        return dict(id=f"fx/{e['id']}/{n[0]}", type="article", sectionName=sec, webPublicationDate=f"{d}T0{n[0] % 9}:10:00Z", webTitle=title, fields=dict(headline=title))
    hit = f"{kw0.title()} {rq0} update".replace("  ", " ")
    if "page-size" in p and p["page-size"] == 1:
        tot = 120 + (int(p["from-date"][-2:]) if p.get("from-date") == p.get("to-date") else 4200)
        return dict(response=dict(status="ok", total=tot, startIndex=1, pageSize=1, currentPage=1, pages=tot, results=[]))
    pg = int(p.get("page", 1)); pages = 2
    if pg == 1:
        res = [item(1, hit), item(1, hit), item(2, "Local council approves budget"), item(2, f"{kw0} {rq0} {ex[0]}" if ex else "Unrelated sport result")]
        if rq0 and rq0 != kw0: res.append(item(3, f"{kw0} in other news"))      # kw without require: must not match
    else:
        res = [item(5, hit.upper(), "Australia news")]
    return dict(response=dict(status="ok", total=len(res) * 2, startIndex=1, pageSize=len(res), currentPage=pg, pages=pages, results=res))

def run_guardian(cl, events, W, daily_totals, plan):
    qf = ARGS.query_fields
    for e in events:
        a, b = window(e, W); d = cl.root / e["id"]; q = guardian_query(e); h = hashlib.md5((q + str(qf)).encode()).hexdigest()[:8]
        base = {"from-date": str(a), "to-date": str(b), "tag": "type/article", "order-by": "oldest", "use-date": "published"}
        if plan: print(f"{e['id']:22} q={q[:150]}"); continue
        fx = lambda p: (lambda: guardian_fixture(e, p))
        p = dict(base, **{"page-size": 1}); tot, f1 = cl.cached(d / "window_total.json", CFG["guardian"]["url"], p, fx(p))
        total = tot["response"]["total"]
        p = dict(base, q=q, **{"page-size": GUARDIAN_PAGE_SIZE, "show-fields": "headline", "page": 1}); 
        if qf != "none": p["query-fields"] = qf
        r, f2 = cl.cached(d / f"cand_{h}_p001.json", CFG["guardian"]["url"], p, fx(p)); pages = r["response"]["pages"]; n = r["response"]["total"]
        trunc = pages > GUARDIAN_MAX_PAGES; nf = int(f1) + int(f2)
        for pg in range(2, min(pages, GUARDIAN_MAX_PAGES) + 1):
            p2 = dict(p, page=pg); _, f = cl.cached(d / f"cand_{h}_p{pg:03d}.json", CFG["guardian"]["url"], p2, fx(p2)); nf += int(f)
        if daily_totals:
            for k in range(W):
                day = a + dt.timedelta(days=k); p3 = dict(base, **{"from-date": str(day), "to-date": str(day), "page-size": 1})
                _, f = cl.cached(d / f"day_total_{day}.json", CFG["guardian"]["url"], p3, fx(p3)); nf += int(f)
        json.dump(dict(event=e["id"], query=q, query_hash=h, query_fields=qf, window_total=total, candidates=n, pages=pages, truncated=trunc,
                       daily_totals=bool(daily_totals), fetched=dt.datetime.now().isoformat(timespec="seconds")), open(d / "meta.json", "w"), indent=1)
        print(f"{e['id']:22} window_total={total:7} candidates={n:6} pages={pages:3}{' TRUNCATED' if trunc else ''} new_calls={nf} (today {cl.today()})", flush=True)


# ---------- NYT ----------
def nyt_months(events, W):
    ms = set()
    for e in events:
        a, b = window(e, W)
        for x in (a, b): ms.add((x.year, x.month))
    return sorted(ms)

def nyt_fixture(y, m, events, W):
    docs = []
    def doc(d, title, tom="News", dt_="article", sec="World", i=[0]):
        i[0] += 1; return dict(web_url=f"https://fx/{i[0]}", headline=dict(main=title, print_headline=title + " (print)"), pub_date=f"{d}T0{i[0] % 9}:00:00+0000",
                               section_name=sec, type_of_material=tom, document_type=dt_, _id=f"nyt://fx/{i[0]}")
    for day in range(1, 29):
        d = dt.date(y, m, day)
        for k in range(3): docs.append(doc(d, f"Filler story {k} for {d}"))
    docs.append(doc(dt.date(y, m, 2), "Obituary of someone", "Obituary (Obit)")); docs.append(doc(dt.date(y, m, 2), "Paid Notice: Deaths SMITH", "Paid Death Notice"))
    docs.append(doc(dt.date(y, m, 3), "Correction: earlier article", "Correction")); docs.append(doc(dt.date(y, m, 3), "A video", "Video", "multimedia"))
    for e in events:
        t0 = dt.date.fromisoformat(e["onset"]); kw0 = e["kw"][0].replace("*", ""); rq = [w.replace("*", "") for w in (e.get("require") or [])]
        hit = f"{kw0.title()} {rq[0] if rq else ''} update".replace("  ", " ")
        for k in (3, 20):
            d = t0 + dt.timedelta(days=k)
            if (d.year, d.month) == (y, m):
                docs.append(doc(d, hit)); docs.append(doc(d, hit))                  # exact same-day duplicate
                docs.append(doc(d, hit + " opinion", "Op-Ed"))                       # excluded by type
    return dict(copyright="fixture", response=dict(meta=dict(hits=len(docs)), docs=docs))

def run_nyt(cl, events, W, plan):
    months = nyt_months(events, W); print(f"{len(months)} archive months needed for {len(events)} events")
    if plan: print(", ".join(f"{y}-{m:02d}" for y, m in months)); return
    for y, m in months:
        p = cl.root / f"{y}-{m:02d}.json.gz"; url = CFG["nyt"]["url"].format(y=y, m=m)
        _, f = cl.cached(p, url, {}, lambda: nyt_fixture(y, m, events, W), gz=True)
        print(f"{y}-{m:02d} {'fetched' if f else 'cached '} (calls today {cl.today()})", flush=True)


def main():
    global ARGS
    ap = argparse.ArgumentParser(); ap.add_argument("--outlet", required=True, choices=["guardian", "nyt"])
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--events"); ap.add_argument("--plan", action="store_true", help="print the call plan only")
    ap.add_argument("--daily-totals", action="store_true", help="Guardian: also 30 calls/event for per-day totals (default: window total only)")
    ap.add_argument("--query-fields", default="headline", help="Guardian query-fields value, or 'none' (headline = search headlines only)")
    ARGS = ap.parse_args(); load_env()
    dry = ARGS.dry_run
    if not dry and not os.environ.get(CFG[ARGS.outlet]["env"]):
        print(f"{CFG[ARGS.outlet]['env']} missing in .env -> DRY RUN (fixtures only)"); dry = True
    events, W = load_events(ARGS.events.split(",") if ARGS.events else None)
    cl = Client(ARGS.outlet, dry); print(f"outlet={ARGS.outlet} events={len(events)} window={W}d mode={'DRY-RUN' if dry else 'LIVE'} cache={cl.root}")
    try:
        run_guardian(cl, events, W, ARGS.daily_totals, ARGS.plan) if ARGS.outlet == "guardian" else run_nyt(cl, events, W, ARGS.plan)
    except StopFetch as ex:
        print(f"STOPPED: {ex}\nProgress is cached; rerun the same command to resume."); sys.exit(3)
    print(f"done. live calls this run: {cl.calls}")

if __name__ == "__main__": main()
