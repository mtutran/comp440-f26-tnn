"""Count every citizenship (P27) value held by the candidate population, via QLever.

Population: humans (P31=Q5) with a birth year >= 1900 and an English Wikipedia
article (any gender, unlike the candidate pool, which keeps female/male only).
"Unknown value" citizenships (blank nodes) are ignored.

For each citizenship QID:
  n_people  people who hold it
  n_only    people for whom it is their only citizenship
  mapped    whether it is in regions.csv, country_aliases.csv or historical_states.csv

Output: data/reference/citizenship_counts.csv (read by build_country_mappings.py,
which maps every unmapped value with n_people >= MIN_PEOPLE).
Prints the unmapped values with n_people >= MIN_PEOPLE, and the totals for the rest.
"""

import pandas as pd

import sample_people as s
from assign_region import REF, load_country_map

OUT_PATH = REF / "citizenship_counts.csv"
MIN_PEOPLE = 50

QUERY = s.PREFIXES + """
SELECT ?cit ?label (COUNT(?person) AS ?n_people) (SUM(?only) AS ?n_only) WHERE {
  { SELECT ?person (COUNT(DISTINCT ?c) AS ?k) WHERE {
      ?person wdt:P31 wd:Q5 .
      ?person wdt:P569 ?b . FILTER(isLiteral(?b) && YEAR(?b) >= 1900)
      ?a schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
      ?person wdt:P27 ?c . FILTER(!CONTAINS(STR(?c), "/.well-known/genid/"))
    } GROUP BY ?person }
  ?person wdt:P27 ?cit . FILTER(!CONTAINS(STR(?cit), "/.well-known/genid/"))
  BIND(IF(?k = 1, 1, 0) AS ?only)
  OPTIONAL { ?cit rdfs:label ?label . FILTER(LANG(?label) = "en") }
}
GROUP BY ?cit ?label
"""


def main():
    print(f"QLever Wikidata index: {s.snapshot_info()}\n")
    df = s.run_query(QUERY)
    df["cit"] = s.short(df["cit"])
    df = (df.astype({"n_people": int, "n_only": int})
          .drop_duplicates("cit")
          .sort_values(["n_people", "cit"], ascending=[False, True])
          .rename(columns={"cit": "qid"}))
    to_modern, _ = load_country_map()
    df["mapped"] = df["qid"].isin(to_modern)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} citizenship values to {OUT_PATH.relative_to(s.ROOT)}")

    unmapped = df[~df["mapped"]]
    big = unmapped[unmapped["n_people"] >= MIN_PEOPLE]
    rest = unmapped[unmapped["n_people"] < MIN_PEOPLE]
    print(f"Unmapped: {len(unmapped)} values. With >= {MIN_PEOPLE} people: {len(big)} "
          f"(n_people {big['n_people'].sum()}, n_only {big['n_only'].sum()}). "
          f"Below: {len(rest)} (n_people {rest['n_people'].sum()}, "
          f"n_only {rest['n_only'].sum()})\n")
    print(big[["qid", "label", "n_people", "n_only"]].to_string(index=False))


if __name__ == "__main__":
    main()
