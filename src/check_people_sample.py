"""Sanity checks on data/processed/people_sample.csv.

1. Every sampled entity should be an instance of human (P31 = Q5) on Wikidata.
   Print anyone who isn't, with their label and P31 values.
2. Count how many people were born before 1900 (by birth_date year).
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
PEOPLE_PATH = ROOT / "data" / "processed" / "people_sample.csv"

ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "COMP440-PopQA-Research/0.1 (Macalester College course project; "
                  "python-requests) ",
    "Accept": "application/sparql-results+json",
}
BATCH_SIZE = 100

QUERY = """
SELECT ?item ?itemLabel
       (GROUP_CONCAT(DISTINCT ?type; separator="|") AS ?types)
       (GROUP_CONCAT(DISTINCT ?typeLabel; separator="|") AS ?typeLabels)
WHERE {
  VALUES ?item { %s }
  OPTIONAL { ?item wdt:P31 ?type .
             OPTIONAL { ?type rdfs:label ?typeLabel . FILTER(LANG(?typeLabel) = "en") } }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
GROUP BY ?item ?itemLabel
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


def birth_year(value):
    """Year from a Wikidata time string like '1932-03-08T00:00:00Z' or '-0450-...'."""
    if not isinstance(value, str):
        return None
    m = re.match(r"^([+-]?)(\d+)-", value)
    if not m:
        return None
    year = int(m.group(2))
    return -year if m.group(1) == "-" else year


def main():
    people = pd.read_csv(PEOPLE_PATH, dtype=str)
    qids = people["qid"].str.rsplit("/", n=1).str[-1].tolist()

    types = {}
    for i in range(0, len(qids), BATCH_SIZE):
        batch = qids[i:i + BATCH_SIZE]
        for b in sparql(QUERY % " ".join(f"wd:{q}" for q in batch)):
            g = lambda k: b.get(k, {}).get("value", "")
            qid = g("item").rsplit("/", 1)[-1]
            types[qid] = {
                "label": g("itemLabel"),
                "types": [t.rsplit("/", 1)[-1] for t in g("types").split("|") if t],
                "type_labels": [t for t in g("typeLabels").split("|") if t],
            }

    not_human = [(q, types.get(q)) for q in qids
                 if q not in types or "Q5" not in types[q]["types"]]
    print(f"{len(qids)} people checked; {len(not_human)} not instance of human (Q5):")
    for q, info in not_human:
        if info is None:
            print(f"  {q}  (not returned by Wikidata: deleted or redirected?)")
            continue
        p31 = ", ".join(f"{t} ({l})" for t, l in
                        zip(info["types"], info["type_labels"] + [""] * len(info["types"])))
        print(f"  {q}  {info['label']}  P31: {p31 or 'none'}")

    years = people["birth_date"].map(birth_year)
    before = (years < 1900).sum()
    print(f"\nBorn before 1900: {before} of {len(people)} "
          f"({years.notna().sum()} have a birth date, {years.isna().sum()} missing)")


if __name__ == "__main__":
    main()
