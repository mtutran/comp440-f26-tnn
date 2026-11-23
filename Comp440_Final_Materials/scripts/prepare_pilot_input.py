#!/usr/bin/env python3
"""Turn Tu's sample (data/processed/study_sample_pilot.csv at the repo root)
into the input run_eval.py expects, plus the two grader lookup tables.

Writes into Comp440_Final_Materials/data/:
    pilot_input.csv           one row per person, columns run_eval.py reads by
                              default (qid, name, description, birth_year,
                              citizenship, is_deceased) plus Tu's stratum,
                              gender, region and coverage columns for analysis
    historical_to_modern.csv  Tu's historical-state table as historical,modern
                              (several successors joined with "|")
    country_aliases.csv       every country's label, aliases and demonyms from
                              Wikidata (only fetched if missing; --refresh-aliases)

What it fixes:
    - citizenship is stored as Wikidata IDs; it becomes country labels
    - people with no Wikidata English label get their Wikipedia title as name
    - is_deceased comes from Wikidata's date of death (P570). People with no
      death date who were born more than 100 years ago get "unknown", so the
      question says "is or was" instead of claiming they are alive.

Run from Comp440_Final_Materials/:
    python scripts/prepare_pilot_input.py
    python scripts/prepare_pilot_input.py --sample ../data/processed/study_sample.csv \
        --out data/study_input.csv          # same thing for the full sample later
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.dirname(HERE)                 # Comp440_Final_Materials
REPO = os.path.dirname(FOLDER)
REF = os.path.join(REPO, "data", "reference")

SPARQL = "https://query.wikidata.org/sparql"
USER_AGENT = "COMP440-LLM-coverage-project (student research)"


def fetch_death_dates(qids: list[str]) -> dict[str, str]:
    """qid -> date of death (ISO string) for those who have one in Wikidata."""
    out = {}
    for i in range(0, len(qids), 200):
        values = " ".join(f"wd:{q}" for q in qids[i:i + 200])
        query = f"""SELECT ?p (SAMPLE(?d) AS ?death) WHERE {{
            VALUES ?p {{ {values} }} ?p wdt:P570 ?d }} GROUP BY ?p"""
        r = requests.post(SPARQL, data={"query": query, "format": "json"},
                          headers={"User-Agent": USER_AGENT}, timeout=120)
        r.raise_for_status()
        for b in r.json()["results"]["bindings"]:
            out[b["p"]["value"].rsplit("/", 1)[1]] = b["death"]["value"]
    return out


def citizenship_labels(sample: pd.DataFrame) -> pd.Series:
    counts = pd.read_csv(os.path.join(REF, "citizenship_counts.csv"), dtype=str)
    label = dict(zip(counts["qid"], counts["label"]))
    regions = pd.read_csv(os.path.join(REF, "regions.csv"), dtype=str)
    label.update({q: c for q, c in zip(regions["qid"], regions["country"]) if q not in label})
    missing = sorted({q for s in sample["citizenship_qids"] for q in s.split("|")} - set(label))
    if missing:
        raise SystemExit(f"No country label for citizenship QIDs {missing}; "
                         "add them to data/reference/citizenship_counts.csv")
    return sample["citizenship_qids"].map(lambda s: "|".join(label[q] for q in s.split("|")))


def write_historical(path: str):
    h = pd.read_csv(os.path.join(REF, "historical_states.csv"), dtype=str)
    out = pd.DataFrame({"historical": h["historical_label"], "modern": h["modern_labels"]})
    out.dropna().to_csv(path, index=False)
    print(f"wrote {len(out)} historical states to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default=os.path.join(REPO, "data", "processed",
                                                     "study_sample_pilot.csv"))
    ap.add_argument("--out", default=os.path.join(FOLDER, "data", "pilot_input.csv"))
    ap.add_argument("--refresh-aliases", action="store_true",
                    help="re-download data/country_aliases.csv even if it exists")
    args = ap.parse_args()

    p = pd.read_csv(args.sample, dtype=str)
    need = {"qid", "label", "description", "citizenship_qids", "birth_date", "enwiki_title"}
    if need - set(p.columns):
        raise SystemExit(f"{args.sample} is missing columns {sorted(need - set(p.columns))}")

    no_label = p["label"].isna()
    p["name"] = p["label"].fillna(p["enwiki_title"])
    if no_label.any():
        print(f"{no_label.sum()} people had no label; using their Wikipedia title: "
              + ", ".join(p.loc[no_label, "name"]))
    if p["name"].isna().any():
        raise SystemExit(f"no name for {list(p.loc[p['name'].isna(), 'qid'])}")

    p["citizenship"] = citizenship_labels(p)
    p["birth_year"] = p["birth_date"]

    deaths = fetch_death_dates(list(p["qid"]))
    p["death_date"] = p["qid"].map(deaths)
    born = pd.to_numeric(p["birth_date"].str.extract(r"^(-?\d+)")[0], errors="coerce")
    very_old = born < dt.date.today().year - 100
    p["is_deceased"] = "false"
    p.loc[p["death_date"].notna(), "is_deceased"] = "true"
    p.loc[p["death_date"].isna() & very_old, "is_deceased"] = ""   # unknown -> "is or was"
    print(f"is_deceased: {(p.is_deceased == 'true').sum()} true, "
          f"{(p.is_deceased == 'false').sum()} false, {(p.is_deceased == '').sum()} unknown")

    cols = ["qid", "name", "description", "birth_year", "citizenship", "is_deceased",
            "death_date", "stratum", "gender", "region", "region_detail", "global_south",
            "coverage", "n_sitelinks", "enwiki_title", "citizenship_qids"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    p[[c for c in cols if c in p.columns]].to_csv(args.out, index=False)
    print(f"wrote {len(p)} people to {args.out}")

    data = os.path.join(FOLDER, "data")
    write_historical(os.path.join(data, "historical_to_modern.csv"))
    aliases = os.path.join(data, "country_aliases.csv")
    if args.refresh_aliases or not os.path.exists(aliases):
        sys.path.insert(0, HERE)
        import fetch_country_aliases
        fetch_country_aliases.main(aliases)
    else:
        print(f"keeping existing {aliases} (use --refresh-aliases to re-download)")


if __name__ == "__main__":
    main()
