"""Assign each person in data/processed/people_sample_filtered.csv a region.

Inputs: data/reference/regions.csv, country_aliases.csv, historical_states.csv
(run build_regions.py and build_country_mappings.py first).

Every citizenship QID is mapped to one or more current countries:
  current country -> itself; alias -> its country; historical state -> its
  modern successors (all of them, so e.g. the Soviet Union maps to 15 countries
  spanning Asia and Europe).

Each of region, region_detail and global_south is then filled independently by:
  1. citizenship  all mapped citizenship countries agree on one value
  2. birthplace   otherwise, the country of the place of birth (P19 -> P17,
                  ignoring P17 statements with an end time) gives one value
  3. ambiguous    otherwise; region/region_detail = "ambiguous", global_south empty

region_rule records which step set region; region_detail_rule and
global_south_rule do the same for the other two columns.
Output: data/processed/people_regions.csv
"""

import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "data" / "reference"
IN_PATH = ROOT / "data" / "processed" / "people_sample_filtered.csv"
OUT_PATH = ROOT / "data" / "processed" / "people_regions.csv"

ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "COMP440-PopQA-Research/0.1 (Macalester College course project; "
                  "python-requests) ",
    "Accept": "application/sparql-results+json",
}
SEP = "|"
BATCH_SIZE = 100
AMBIGUOUS = "ambiguous"
FIELDS = ["region", "region_detail", "global_south"]

BIRTHPLACE_QUERY = """
SELECT ?person (GROUP_CONCAT(DISTINCT ?country; separator="|") AS ?countries)
WHERE {
  VALUES ?person { %s }
  ?person wdt:P19 ?place .
  ?place p:P17 ?st . ?st ps:P17 ?country ; wikibase:rank ?rank .
  FILTER(?rank != wikibase:DeprecatedRank)
  FILTER NOT EXISTS { ?st pq:P582 ?end }
}
GROUP BY ?person
"""


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


def split(value):
    return [v for v in value.split(SEP) if v] if isinstance(value, str) and value else []


def load_country_map():
    """Return ({any QID: [modern country QIDs]}, regions DataFrame indexed by qid)."""
    regions = pd.read_csv(REF / "regions.csv", dtype=str, keep_default_na=False)
    aliases = pd.read_csv(REF / "country_aliases.csv", dtype=str, keep_default_na=False)
    hist = pd.read_csv(REF / "historical_states.csv", dtype=str, keep_default_na=False)
    to_modern = {q: [q] for q in regions["qid"]}
    to_modern |= {a: [c] for a, c in zip(aliases["alias_qid"], aliases["country_qid"])}
    to_modern |= {h: split(m) for h, m in zip(hist["historical_qid"], hist["modern_qids"])}
    return to_modern, regions.set_index("qid")


def fetch_birthplace_countries(qids):
    out = {}
    for i in range(0, len(qids), BATCH_SIZE):
        batch = qids[i:i + BATCH_SIZE]
        for b in sparql(BIRTHPLACE_QUERY % " ".join(f"wd:{q}" for q in batch)):
            person = b["person"]["value"].rsplit("/", 1)[-1]
            out[person] = [u.rsplit("/", 1)[-1] for u in split(b["countries"]["value"])]
        time.sleep(1)
    return out


def modern_of(qids, to_modern, unmapped):
    """Map citizenship/country QIDs to modern countries; record unmapped QIDs."""
    modern = set()
    for q in qids:
        if q in to_modern:
            modern |= set(to_modern[q])
        else:
            unmapped.add(q)
    return modern


def assign_person(cit_qids, birth_country_qids, to_modern, regions,
                  unmapped_cit, unmapped_birth):
    """Return {field: value, f"{field}_rule": rule} for one person (see module doc)."""
    cit = modern_of(cit_qids, to_modern, unmapped_cit)
    birth = modern_of(birth_country_qids, to_modern, unmapped_birth)

    def values(modern, field):
        return {regions.at[m, field] for m in modern if regions.at[m, field] != ""}

    out = {}
    for field in FIELDS:
        from_cit, from_birth = values(cit, field), values(birth, field)
        if len(from_cit) == 1:
            value, rule = from_cit.pop(), "citizenship"
        elif len(from_birth) == 1:
            value, rule = from_birth.pop(), "birthplace"
        else:
            value = pd.NA if field == "global_south" else AMBIGUOUS
            rule = AMBIGUOUS
        out[field], out[f"{field}_rule"] = value, rule
    return out


def add_region_columns(people, cit_lists, birth_lists, to_modern, regions):
    """Add region/region_detail/global_south (+ *_rule) columns to `people` in place.

    cit_lists / birth_lists: per-row lists of citizenship / birthplace-country QIDs.
    Returns (unmapped citizenship QIDs, unmapped birthplace-country QIDs).
    """
    unmapped_cit, unmapped_birth = set(), set()
    rows = [assign_person(c, b, to_modern, regions, unmapped_cit, unmapped_birth)
            for c, b in zip(cit_lists, birth_lists)]
    for col in FIELDS + [f"{f}_rule" for f in FIELDS]:
        people[col] = [r[col] for r in rows]
    people["global_south"] = people["global_south"].map(
        {"True": True, "False": False}).astype("boolean")
    return unmapped_cit, unmapped_birth


def main():
    people = pd.read_csv(IN_PATH, dtype=str)
    to_modern, regions = load_country_map()

    print("Fetching birthplace countries (P19 -> P17) ...")
    birth_countries = fetch_birthplace_countries(people["resolved_qid"].tolist())

    unmapped_cit, unmapped_birth = add_region_columns(
        people,
        [split(c) for c in people["citizenship_qids"]],
        [birth_countries.get(q, []) for q in people["resolved_qid"]],
        to_modern, regions)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    people.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(people)} rows to {OUT_PATH.relative_to(ROOT)}\n")

    if unmapped_cit:
        print(f"WARNING: citizenship QIDs not in any reference table (ignored): "
              f"{sorted(unmapped_cit)} -- rerun build_country_mappings.py")
    if unmapped_birth:
        print(f"Note: birthplace-country QIDs not in any reference table (ignored): "
              f"{sorted(unmapped_birth)}\n")

    gender = people["gender"].fillna("(missing)")
    print("Region x gender:")
    print(pd.crosstab(people["region"], gender, margins=True, margins_name="total")
          .to_string(), "\n")
    print("Region detail x gender:")
    print(pd.crosstab(people["region_detail"], gender, margins=True, margins_name="total")
          .to_string(), "\n")
    print("Rule used (people per rule):")
    rules = pd.DataFrame({f: people[f"{f}_rule"].value_counts() for f in FIELDS})
    print(rules.reindex(["citizenship", "birthplace", AMBIGUOUS]).fillna(0).astype(int)
          .to_string())


if __name__ == "__main__":
    main()
