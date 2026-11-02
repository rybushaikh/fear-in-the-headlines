"""Subject tag rules shared by build_subject.py and score_all_v7e.py (copied from build_subject.py, 27 Oct 2026)."""
import re
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

def nyt_kw(d): return ";".join(str(x.get("value")) for x in (d.get("keywords") or []) if x.get("name") in ("subject", "Subject", "organizations", "Organization"))
