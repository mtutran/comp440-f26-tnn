#!/usr/bin/env python3
"""Download every country's English label, aliases and demonyms from Wikidata
into data/country_aliases.csv (columns: canonical, alias), for the grader's
--aliases-csv option. Run it on your own machine:

    python scripts/fetch_country_aliases.py
"""

import csv
import os

import requests

QUERY = """
SELECT ?country ?label ?alias WHERE {
  { ?country wdt:P31 wd:Q6256 } UNION { ?country wdt:P31 wd:Q3624078 }
  UNION { ?country wdt:P31 wd:Q3024240 }          # historical countries too
  ?country rdfs:label ?label FILTER(LANG(?label) = "en")
  OPTIONAL {
    { ?country skos:altLabel ?alias FILTER(LANG(?alias) = "en") }
    UNION { ?country wdt:P1549 ?alias FILTER(LANG(?alias) = "en") }   # demonym
  }
}
"""


def main(path="data/country_aliases.csv"):
    r = requests.get("https://query.wikidata.org/sparql",
                     params={"query": QUERY, "format": "json"},
                     headers={"User-Agent": "COMP440-LLM-coverage-project (student research)"},
                     timeout=120)
    r.raise_for_status()
    rows = set()
    for b in r.json()["results"]["bindings"]:
        label = b["label"]["value"]
        rows.add((label, label))
        if "alias" in b:
            rows.add((label, b["alias"]["value"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["canonical", "alias"])
        w.writerows(sorted(rows))
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
