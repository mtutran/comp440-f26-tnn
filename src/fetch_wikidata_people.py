"""Sample PopQA people (occupation / place of birth) and fetch Wikidata attributes.

For each person: gender (P21), countries of citizenship (P27), birth date (P569),
and number of Wikipedia sitelinks. Output: one row per person, with multi-valued
fields joined by "|".
"""

import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "popqa.csv"
OUT_PATH = ROOT / "data" / "processed" / "people_sample.csv"

ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "COMP440-PopQA-Research/0.1 (Macalester College course project; "
                  "python-requests) ",
    "Accept": "application/sparql-results+json",
}
SAMPLE_SIZE = 500
BATCH_SIZE = 100
SEED = 440
SEP = "|"

QUERY_TEMPLATE = """
SELECT ?person
       (SAMPLE(?sitelinks) AS ?n_sitelinks)
       (GROUP_CONCAT(DISTINCT ?genderLabel; separator="|") AS ?gender)
       (GROUP_CONCAT(DISTINCT ?countryLabel; separator="|") AS ?citizenship)
       (GROUP_CONCAT(DISTINCT ?country; separator="|") AS ?citizenship_qids)
       (MIN(?birth) AS ?birth_date)
WHERE {
  VALUES ?person { %s }
  OPTIONAL { ?person wikibase:sitelinks ?sitelinks . }
  OPTIONAL { ?person wdt:P21 ?g .
             ?g rdfs:label ?genderLabel . FILTER(LANG(?genderLabel) = "en") }
  OPTIONAL { ?person wdt:P27 ?country .
             ?country rdfs:label ?countryLabel . FILTER(LANG(?countryLabel) = "en") }
  OPTIONAL { ?person wdt:P569 ?birth . }
}
GROUP BY ?person
"""


def query_batch(qids, retries=3):
    values = " ".join(f"wd:{q}" for q in qids)
    for attempt in range(retries):
        r = requests.get(ENDPOINT, params={"query": QUERY_TEMPLATE % values},
                         headers=HEADERS, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            wait = int(r.headers.get("Retry-After", 5 * (attempt + 1)))
            print(f"  HTTP {r.status_code}, retrying in {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()["results"]["bindings"]
    raise RuntimeError("Wikidata query failed after retries")


def val(row, key):
    return row[key]["value"] if key in row else None


def main():
    popqa = pd.read_csv(RAW_PATH)
    # NOTE: PopQA's subj_id is an internal integer ID, not a Wikidata QID.
    # The QID lives in s_uri (http://www.wikidata.org/entity/Q...); subj_id <-> s_uri is 1:1.
    popqa["qid"] = popqa["s_uri"].str.rsplit("/", n=1).str[-1]
    people = popqa.loc[popqa["prop"].isin(["occupation", "place of birth"]),
                       ["subj_id", "qid", "subj", "s_pop"]].drop_duplicates("subj_id")
    print(f"Unique people (occupation + place of birth): {len(people)}")

    sampled = people.sample(n=SAMPLE_SIZE, random_state=SEED)
    sample = sampled["qid"].tolist()

    records = []
    for i in range(0, len(sample), BATCH_SIZE):
        batch = sample[i:i + BATCH_SIZE]
        print(f"Querying batch {i // BATCH_SIZE + 1} ({len(batch)} QIDs)...")
        for row in query_batch(batch):
            records.append({
                "qid": val(row, "person").rsplit("/", 1)[-1],
                "gender": val(row, "gender") or None,
                "citizenship": val(row, "citizenship") or None,
                "citizenship_qids": (val(row, "citizenship_qids") or "")
                    .replace("http://www.wikidata.org/entity/", "") or None,
                "birth_date": val(row, "birth_date"),
                "n_sitelinks": val(row, "n_sitelinks"),
            })
        if i + BATCH_SIZE < len(sample):
            time.sleep(1)

    df = pd.DataFrame(records)
    # Keep every sampled QID, even ones Wikidata returned nothing for.
    df = sampled.merge(df, on="qid", how="left")
    df["n_sitelinks"] = pd.to_numeric(df["n_sitelinks"]).astype("Int64")
    df = df[["subj_id", "qid", "subj", "s_pop", "gender", "citizenship", "citizenship_qids",
             "birth_date", "n_sitelinks"]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {OUT_PATH.relative_to(ROOT)}\n")

    # --- Summaries ---
    print("Gender breakdown:")
    print(df["gender"].fillna("(missing)").value_counts().to_string(), "\n")

    n_cit = df["citizenship_qids"].fillna("").map(lambda s: len(s.split(SEP)) if s else 0)
    cit_bucket = pd.cut(n_cit, [-1, 0, 1, float("inf")], labels=["zero", "one", "more than one"])
    print("Number of citizenships:")
    print(cit_bucket.value_counts().reindex(["zero", "one", "more than one"]).to_string(), "\n")

    print("Top 15 countries of citizenship (people with multiple counted once per country):")
    countries = df["citizenship"].dropna().str.split(SEP).explode()
    print(countries.value_counts().head(15).to_string(), "\n")

    missing = df["birth_date"].isna().sum()
    print(f"Missing birth date: {missing} of {len(df)} ({missing / len(df):.1%})")


if __name__ == "__main__":
    main()
