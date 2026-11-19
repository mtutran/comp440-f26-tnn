"""Map citizenship QIDs to countries in data/reference/regions.csv. Run
src/build_regions.py first.

QIDs mapped: every citizenship in data/processed/people_sample.csv, plus every value
in data/reference/citizenship_counts.csv (from find_missing_citizenships.py) held by
at least MIN_PEOPLE people in the candidate population.

Outputs
  data/reference/country_aliases.csv     alias_qid, alias_label, country_qid
      Other Wikidata items for a current country (e.g. Kingdom of the Netherlands
      -> Netherlands). Rows already in the file are kept, so hand edits survive.
  data/reference/historical_states.csv   historical_qid, historical_label,
      modern_qids, modern_labels, regions, rule  (multi-valued fields joined by "|")

For each citizenship QID not in regions.csv or country_aliases.csv:
  - Alias: the item has no dissolution date (P576), and its ISO alpha-3 (P298),
    M49 code (P2082), or "country" (P17) points to exactly one country in
    regions.csv.
  - Historical state: everything else. Starting from the item, follow "replaced by"
    (P1366) breadth-first. Stop on any item that is a current country or alias, and
    keep following the rest until no new items remain. The modern countries reached
    decide the rule:
        single_successor   exactly one modern country
        same_region_split  several modern countries, all in the same region
        ambiguous_split    several modern countries in different regions
        not_found          the P1366 chain never reaches a modern country
        manual             set by hand in MANUAL_HISTORICAL (applied after the
                           automatic search, so hand edits persist across reruns;
                           these rows are always written, even if no one in the
                           current sample has that citizenship)
"""

import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "data" / "reference"
PEOPLE_PATH = ROOT / "data" / "processed" / "people_sample.csv"
REGIONS_PATH = REF / "regions.csv"
ALIASES_PATH = REF / "country_aliases.csv"
HISTORICAL_PATH = REF / "historical_states.csv"
COUNTS_PATH = REF / "citizenship_counts.csv"
MIN_PEOPLE = 50  # map candidate-population citizenships held by at least this many

ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "COMP440-PopQA-Research/0.1 (Macalester College course project; "
                  "python-requests) ",
    "Accept": "application/sparql-results+json",
}
SEP = "|"
BATCH_SIZE = 100
MAX_DEPTH = 20  # safety cap on P1366 chain length

SEED_ALIASES = [
    ("Q29999", "Kingdom of the Netherlands", "Q55"),
    ("Q756617", "Kingdom of Denmark", "Q35"),
]

# Hand-checked successors, overriding the automatic P1366 search. Values are lists
# of modern country QIDs (must be in regions.csv), or ("same_as", QID) to copy the
# automatic successors of another historical state; both get rule "manual".
# ("successors", [QIDs]) sets the successors by hand but keeps the computed rule
# (single_successor / same_region_split / ambiguous_split).
WEST_INDIES_SUCCESSORS = [
    "Q781",  # Antigua and Barbuda
    "Q244",  # Barbados
    "Q784",  # Dominica
    "Q769",  # Grenada
    "Q766",  # Jamaica
    "Q763",  # Saint Kitts and Nevis
    "Q760",  # Saint Lucia
    "Q757",  # Saint Vincent and the Grenadines
    "Q754",  # Trinidad and Tobago
]
# Ottoman Empire: modern states covering land under direct Ottoman rule in 1900
# (holders were born 1900 or later). Wikidata's P1366 lists only Turkey, the
# Principality of Serbia and 1919-23 transitional entities.
OTTOMAN_SUCCESSORS = [
    "Q43",      # Turkey
    "Q222",     # Albania
    "Q41",      # Greece
    "Q221",     # North Macedonia
    "Q1246",    # Kosovo
    "Q796",     # Iraq
    "Q858",     # Syria
    "Q822",     # Lebanon
    "Q810",     # Jordan
    "Q801",     # Israel
    "Q219060",  # Palestine
    "Q851",     # Saudi Arabia
    "Q805",     # Yemen
    "Q1016",    # Libya
]
MANUAL_HISTORICAL = {
    "Q170072": ["Q55"],                 # Dutch Republic -> Netherlands
    "Q762943": ["Q55"],                 # County of Holland -> Netherlands
    "Q1031430": ["Q55", "Q31"],         # Habsburg Netherlands -> Netherlands, Belgium
    "Q27306": ["Q183"],                 # Kingdom of Prussia -> Germany
    "Q154195": ["Q183"],                # Kingdom of Bavaria -> Germany
    "Q1290149": ("same_as", "Q83286"),  # FPR Yugoslavia -> as SFR Yugoslavia
    "Q33946": ["Q213", "Q214"],         # Czechoslovakia -> Czech Republic, Slovakia
    "Q652560": WEST_INDIES_SUCCESSORS,  # West Indies Federation (sovereign successors)
    "Q335070": ["Q38"],                 # Alba Longa -> Italy
    "Q179876": ["Q145"],                # Kingdom of England -> United Kingdom
    "Q161885": ["Q145"],                # Kingdom of Great Britain -> United Kingdom
    # Candidate-population citizenships (>= MIN_PEOPLE), where P1366 fails or is too broad
    "Q1054923": ["Q148"],               # British Hong Kong -> China
    "Q13426199": ["Q148", "Q865"],      # Republic of China -> China, Taiwan
    "Q8733": ["Q148", "Q865"],          # Qing dynasty -> as Republic of China (P1366: Taiwan only)
    "Q29520": ["Q148", "Q865"],         # China (the general item) -> as Republic of China
    "Q36704": ("same_as", "Q83286"),    # Yugoslavia -> as SFR Yugoslavia (P1366: Serbia, Montenegro only)
    "Q12560": ("successors", OTTOMAN_SUCCESSORS),  # Ottoman Empire (rule: ambiguous_split)
    "Q2017684": ["Q948"],               # French protectorate of Tunisia -> Tunisia
    "Q3087763": ["Q79"],                # Republic of Egypt (1953-58) -> Egypt (P1366 via UAR: + Syria)
    "Q124943": ["Q79"],                 # Kingdom of Egypt -> Egypt (P1366 via UAR: + Syria)
    "Q146885": ["Q858"],                # Second Syrian Republic -> Syria (P1366 via UAR: + Egypt)
    "Q2305208": ["Q159"],               # Russian SFSR (1917-22) -> Russia (P1366 via USSR: 15 states)
    "Q139319": ("same_as", "Q34266"),   # Russian Republic (1917) -> as Russian Empire
    "Q193714": ["Q801", "Q219060"],     # Mandatory Palestine -> Israel, Palestine (P1366: + Egypt, Jordan)
    "Q23681": ["Q229"],                 # Northern Cyprus -> Cyprus (internationally recognised)
    "Q31354462": ["Q230"],              # Republic of Abkhazia -> Georgia (internationally recognised)
    "Q149805": ["Q796"],                # Kingdom of Iraq -> Iraq
    "Q3041595": ["Q796"],               # First Republic of Iraq -> Iraq
    "Q3108185": ["Q796"],               # Ba'athist Iraq -> Iraq
    "Q26988": ["Q664"],                 # Cook Islands (free association) -> New Zealand
}

ITEM_QUERY = """
SELECT ?item ?itemLabel (SAMPLE(?dissolved) AS ?dissolvedAt)
       (GROUP_CONCAT(DISTINCT ?next; separator="|") AS ?replacedBy)
       (GROUP_CONCAT(DISTINCT ?iso3; separator="|") AS ?iso3s)
       (GROUP_CONCAT(DISTINCT ?m49; separator="|") AS ?m49s)
       (GROUP_CONCAT(DISTINCT ?country; separator="|") AS ?countries)
WHERE {
  VALUES ?item { %s }
  OPTIONAL { ?item wdt:P576 ?dissolved }
  OPTIONAL { ?item wdt:P1366 ?next }
  OPTIONAL { ?item wdt:P298 ?iso3 }
  OPTIONAL { ?item wdt:P2082 ?m49 }
  OPTIONAL { ?item wdt:P17 ?country }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
GROUP BY ?item ?itemLabel
"""


def qid_of(uri):
    return uri.rsplit("/", 1)[-1]


def split(value):
    return [v for v in value.split(SEP) if v] if value else []


def sparql(query, retries=4):
    for attempt in range(retries):
        r = requests.get(ENDPOINT, params={"query": query}, headers=HEADERS, timeout=120)
        if r.status_code == 429 or r.status_code >= 500:
            wait = int(r.headers.get("Retry-After", 5 * (attempt + 1)))
            print(f"  HTTP {r.status_code}, retrying in {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()["results"]["bindings"]
    r.raise_for_status()


_cache = {}


def fetch_items(qids):
    """Return {qid: info} for the given QIDs, querying Wikidata only for new ones."""
    todo = [q for q in dict.fromkeys(qids) if q not in _cache]
    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i:i + BATCH_SIZE]
        rows = sparql(ITEM_QUERY % " ".join(f"wd:{q}" for q in batch))
        for b in rows:
            g = lambda k: b.get(k, {}).get("value", "")
            _cache[qid_of(g("item"))] = {
                "label": g("itemLabel"),
                "dissolved": bool(g("dissolvedAt")),
                "replaced_by": [qid_of(u) for u in split(g("replacedBy"))],
                "iso3s": split(g("iso3s")),
                "m49s": [m.zfill(3) for m in split(g("m49s"))],
                "countries": [qid_of(u) for u in split(g("countries"))],
            }
        for q in batch:  # items with no statements still get an entry
            _cache.setdefault(q, {"label": q, "dissolved": False, "replaced_by": [],
                                  "iso3s": [], "m49s": [], "countries": []})
    return {q: _cache[q] for q in qids}


def alias_target(qid, info, regions):
    """The single current country this non-dissolved item stands for, else None."""
    if info["dissolved"]:
        return None
    by_iso = dict(zip(regions["iso3"], regions["qid"]))
    by_m49 = dict(zip(regions["m49_code"], regions["qid"]))
    modern = set(regions["qid"])
    targets = ({by_iso[c] for c in info["iso3s"] if c in by_iso}
               | {by_m49[c] for c in info["m49s"] if c in by_m49}
               | {c for c in info["countries"] if c in modern and c != qid})
    return targets.pop() if len(targets) == 1 else None


def resolve_successors(start, known):
    """BFS along P1366 from `start`; return the set of modern country QIDs reached.

    `known` maps every current country QID to itself and every alias QID to its
    country.
    """
    reached, seen = set(), {start}
    frontier, depth = [start], 0
    while frontier and depth < MAX_DEPTH:
        info = fetch_items(frontier)
        nxt = []
        for q in frontier:
            for succ in info[q]["replaced_by"]:
                if succ in known:
                    reached.add(known[succ])
                elif succ not in seen:
                    seen.add(succ)
                    nxt.append(succ)
        frontier, depth = nxt, depth + 1
    return reached


def classify(modern, region_of):
    regs = sorted({region_of[m] for m in modern})
    if not modern:
        return regs, "not_found"
    if len(modern) == 1:
        return regs, "single_successor"
    return regs, "same_region_split" if len(regs) == 1 else "ambiguous_split"


def main():
    regions = pd.read_csv(REGIONS_PATH, dtype=str, keep_default_na=False)
    region_of = dict(zip(regions["qid"], regions["region"]))
    label_of = dict(zip(regions["qid"], regions["country"]))

    if ALIASES_PATH.exists():
        aliases = pd.read_csv(ALIASES_PATH, dtype=str, keep_default_na=False)
    else:
        aliases = pd.DataFrame(columns=["alias_qid", "alias_label", "country_qid"])
    for row in SEED_ALIASES:
        if row[0] not in set(aliases["alias_qid"]):
            aliases.loc[len(aliases)] = row
    bad = aliases[~aliases["country_qid"].isin(region_of)]
    if len(bad):
        raise SystemExit(f"Aliases point to QIDs not in regions.csv:\n{bad}")

    old_aliases = set(aliases["alias_qid"])
    old_hist = (set(pd.read_csv(HISTORICAL_PATH, dtype=str)["historical_qid"])
                if HISTORICAL_PATH.exists() else set())

    people = pd.read_csv(PEOPLE_PATH, dtype=str)
    cit = people["citizenship_qids"].dropna().str.split(SEP).explode().map(qid_of)
    counts = cit.value_counts()
    n_only = pd.Series(dtype=int)
    if COUNTS_PATH.exists():  # candidate-population counts take precedence
        pop = pd.read_csv(COUNTS_PATH, dtype=str, keep_default_na=False)
        pop = pop.astype({"n_people": int, "n_only": int}).set_index("qid")
        pop = pop[pop["n_people"] >= MIN_PEOPLE]
        counts = pd.concat([counts.drop(pop.index, errors="ignore"), pop["n_people"]])
        n_only = pop["n_only"]
    known = {q: q for q in regions["qid"]} | dict(zip(aliases["alias_qid"],
                                                       aliases["country_qid"]))
    missing = [q for q in counts.index if q not in known]
    print(f"{len(counts)} citizenship QIDs to map; {len(missing)} not in "
          f"regions.csv or country_aliases.csv")

    info = fetch_items(missing)
    new_aliases, historical = [], []
    for q in missing:
        target = alias_target(q, info[q], regions)
        if target:
            new_aliases.append((q, info[q]["label"], target))
            known[q] = target
        else:
            historical.append(q)
    for row in new_aliases:
        aliases.loc[len(aliases)] = row

    # Automatic search, then hand overrides on top.
    successors = {q: (resolve_successors(q, known), None) for q in historical}
    fetch_items(list(MANUAL_HISTORICAL))  # labels for overrides not in the sample
    for q, spec in MANUAL_HISTORICAL.items():
        rule = "manual"
        if isinstance(spec, tuple) and spec[0] == "same_as":
            modern = resolve_successors(spec[1], known)
        elif isinstance(spec, tuple) and spec[0] == "successors":
            modern, rule = set(spec[1]), None
        else:
            modern = set(spec)
        unknown = modern - set(region_of)
        if unknown:
            raise SystemExit(f"MANUAL_HISTORICAL[{q}] uses QIDs not in regions.csv: {unknown}")
        successors[q] = (modern, rule)

    rows = []
    for q, (modern, override) in successors.items():
        modern = sorted(modern, key=lambda m: label_of[m])
        regs, rule = classify(modern, region_of)
        rule = override or rule
        rows.append({
            "historical_qid": q,
            "historical_label": info[q]["label"],
            "modern_qids": SEP.join(modern),
            "modern_labels": SEP.join(label_of[m] for m in modern),
            "regions": SEP.join(regs),
            "rule": rule,
            "n_people": int(counts.get(q, 0)),
            "n_only": int(n_only.get(q, 0)),
        })
    hist = pd.DataFrame(rows, columns=["historical_qid", "historical_label",
                                       "modern_qids", "modern_labels", "regions",
                                       "rule", "n_people", "n_only"])

    aliases.sort_values("alias_qid").to_csv(ALIASES_PATH, index=False)
    hist.drop(columns=["n_people", "n_only"]).sort_values("historical_label").to_csv(
        HISTORICAL_PATH, index=False)
    print(f"Wrote {len(aliases)} aliases to {ALIASES_PATH.relative_to(ROOT)}")
    print(f"Wrote {len(hist)} historical states to {HISTORICAL_PATH.relative_to(ROOT)}")

    print(f"\nNew aliases ({len(new_aliases)}):")
    for q, label, target in new_aliases:
        print(f"  {q} {label} -> {target} {label_of[target]} [{region_of[target]}] "
              f"(people: {counts.get(q, 0)}, only citizenship: {n_only.get(q, 0)})")
    added = [q for q in new_aliases if q[0] not in old_aliases]
    if len(added) != len(new_aliases):
        print(f"  ({len(new_aliases) - len(added)} of these were already in the file)")

    new_hist = hist[~hist["historical_qid"].isin(old_hist)].sort_values(
        "n_people", ascending=False)
    print(f"\nNew historical states ({len(new_hist)}):")
    for _, r in new_hist.iterrows():
        print(f"  [{r['rule']}] {r['historical_qid']} {r['historical_label']} "
              f"(people: {r['n_people']}, only citizenship: {r['n_only']})")
        print(f"      -> {r['modern_labels'] or '(none)'}  [{r['regions']}]")

    print("\nRules:")
    print(hist["rule"].value_counts().to_string())

    check = hist[hist["rule"].isin(["ambiguous_split", "not_found"])]
    print(f"\nRows to check by hand ({len(check)}):")
    for _, r in check.sort_values(["rule", "n_people"], ascending=[True, False]).iterrows():
        print(f"  [{r['rule']}] {r['historical_qid']} {r['historical_label']} "
              f"(people: {r['n_people']})")
        if r["modern_qids"]:
            print(f"      -> {r['modern_labels']}")
            print(f"      regions: {r['regions']}")

    print("\nManual overrides:")
    for _, r in hist[hist["rule"] == "manual"].iterrows():
        print(f"  {r['historical_qid']} {r['historical_label']} -> {r['modern_labels']} "
              f"[{r['regions']}]")


if __name__ == "__main__":
    main()
