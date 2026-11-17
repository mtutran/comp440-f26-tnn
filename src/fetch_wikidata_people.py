"""Sample PopQA people (occupation / place of birth) and fetch Wikidata attributes.

For each person: gender (P21), countries of citizenship (P27), birth date (P569),
whether they are an instance of human (P31 = Q5), and number of Wikipedia
sitelinks. Output: one row per person, with multi-valued fields joined by "|".

PopQA QIDs can be stale: items merged since PopQA was built are now redirects,
which the SPARQL endpoint does not follow. Each QID is first resolved through the
Wikidata API; `qid` keeps the original PopQA QID and `resolved_qid` is the current
item that was actually queried.

Birth dates ignore "unknown value" (blank-node) claims, so birth_date is either a
real date or empty.

Then writes people_sample_filtered.csv, dropping (in this order) non-humans,
people with no usable birth date, and people born before 1900.
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "popqa.csv"
OUT_PATH = ROOT / "data" / "processed" / "people_sample.csv"
FILTERED_PATH = ROOT / "data" / "processed" / "people_sample_filtered.csv"
API_ENDPOINT = "https://www.wikidata.org/w/api.php"
API_BATCH = 50  # wbgetentities limit
MIN_BIRTH_YEAR = 1900

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
       (SAMPLE(?human) AS ?is_human)
WHERE {
  VALUES ?person { %s }
  OPTIONAL { ?person wikibase:sitelinks ?sitelinks . }
  OPTIONAL { ?person wdt:P21 ?g .
             ?g rdfs:label ?genderLabel . FILTER(LANG(?genderLabel) = "en") }
  OPTIONAL { ?person wdt:P27 ?country .
             ?country rdfs:label ?countryLabel . FILTER(LANG(?countryLabel) = "en") }
  OPTIONAL { ?person wdt:P569 ?birth . FILTER(isLiteral(?birth)) }
  OPTIONAL { ?person wdt:P31 wd:Q5 . BIND(true AS ?human) }
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


def resolve_redirects(qids):
    """Map each QID to its current QID, following merges (redirects).

    Missing or deleted items map to themselves.
    """
    resolved = {}
    for i in range(0, len(qids), API_BATCH):
        batch = qids[i:i + API_BATCH]
        for attempt in range(3):
            r = requests.get(API_ENDPOINT, params={
                "action": "wbgetentities", "ids": "|".join(batch),
                "props": "info", "format": "json"},
                headers={"User-Agent": HEADERS["User-Agent"]}, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        entities = r.json().get("entities", {})
        for q in batch:
            ent = entities.get(q, {})
            resolved[q] = q if "missing" in ent else ent.get("id", q)
    return resolved


def birth_year(value):
    """Year from a Wikidata time string ('1932-03-08T00:00:00Z', '-0450-01-01...')."""
    if not isinstance(value, str):
        return None
    m = re.match(r"^([+-]?)(\d+)-", value)
    if not m:
        return None
    return -int(m.group(2)) if m.group(1) == "-" else int(m.group(2))


def filter_sample(df):
    """Drop non-humans, missing birth dates, pre-1900 births; print counts per reason."""
    years = df["birth_date"].map(birth_year)
    reasons = pd.Series(pd.NA, index=df.index, dtype="object")
    reasons[~df["is_human"]] = "not human (Q5)"
    reasons[reasons.isna() & years.isna()] = "no usable birth date"
    reasons[reasons.isna() & (years < MIN_BIRTH_YEAR)] = f"born before {MIN_BIRTH_YEAR}"
    print(f"Filtering {len(df)} people (each counted under the first reason that applies):")
    for reason in ["not human (Q5)", "no usable birth date", f"born before {MIN_BIRTH_YEAR}"]:
        print(f"  dropped, {reason}: {(reasons == reason).sum()}")
    kept = df[reasons.isna()]
    print(f"  kept: {len(kept)}")
    return kept


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
    print("Resolving redirects ...")
    redirects = resolve_redirects(sampled["qid"].tolist())
    sampled["resolved_qid"] = sampled["qid"].map(redirects)
    changed = sampled[sampled["qid"] != sampled["resolved_qid"]]
    print(f"  {len(changed)} redirected QIDs")
    for _, r in changed.iterrows():
        print(f"    {r['qid']} -> {r['resolved_qid']} ({r['subj']})")
    dup = sampled["resolved_qid"].duplicated(keep=False)
    if dup.any():
        print(f"  WARNING: {dup.sum()} rows share a resolved QID:")
        print(sampled.loc[dup, ["qid", "resolved_qid", "subj"]].to_string(index=False))
    sample = sampled["resolved_qid"].drop_duplicates().tolist()

    records = []
    for i in range(0, len(sample), BATCH_SIZE):
        batch = sample[i:i + BATCH_SIZE]
        print(f"Querying batch {i // BATCH_SIZE + 1} ({len(batch)} QIDs)...")
        for row in query_batch(batch):
            records.append({
                "resolved_qid": val(row, "person").rsplit("/", 1)[-1],
                "gender": val(row, "gender") or None,
                "citizenship": val(row, "citizenship") or None,
                "citizenship_qids": (val(row, "citizenship_qids") or "")
                    .replace("http://www.wikidata.org/entity/", "") or None,
                "birth_date": val(row, "birth_date"),
                "is_human": val(row, "is_human") == "true",
                "n_sitelinks": val(row, "n_sitelinks"),
            })
        if i + BATCH_SIZE < len(sample):
            time.sleep(1)

    df = pd.DataFrame(records)
    # Keep every sampled QID, even ones Wikidata returned nothing for.
    df = sampled.merge(df, on="resolved_qid", how="left")
    df["n_sitelinks"] = pd.to_numeric(df["n_sitelinks"]).astype("Int64")
    df["is_human"] = df["is_human"].fillna(False).astype(bool)
    df = df[["subj_id", "qid", "resolved_qid", "subj", "s_pop", "gender", "citizenship",
             "citizenship_qids", "birth_date", "is_human", "n_sitelinks"]]

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
    print(f"Missing birth date: {missing} of {len(df)} ({missing / len(df):.1%})\n")

    filtered = filter_sample(df)
    filtered.to_csv(FILTERED_PATH, index=False)
    print(f"Saved {len(filtered)} rows to {FILTERED_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
