"""Build data/reference/regions.csv: one row per current sovereign country, keyed by
Wikidata QID, with UN M49 region / sub-region / intermediate region and derived
columns:

  region         Africa, Northern America, Latin America & Caribbean, Asia,
                 Europe, Oceania (M49 regions, with the Americas split into the
                 M49 sub-regions Northern America / Latin America and the Caribbean)
  region_detail  same as region, but Latin America & Caribbean is split into the
                 M49 intermediate regions Caribbean / Central America / South America
  global_south   True if the UN M49 table classifies the country as "Developing",
                 False if "Developed", empty if the country is not in the table

Sources
  - Wikidata SPARQL: the country list and each country's M49 code (P2082) and
    ISO 3166-1 alpha-3 code (P298).
  - UN M49 overview table (https://unstats.un.org/unsd/methodology/m49/overview/):
    the region hierarchy and the developed/developing classification.

Which countries count as "current sovereign"
  A Wikidata item is included if it is (a) a current UN member state (P463 = Q1065,
  no end-time qualifier) or (b) a current instance of "sovereign state" (P31 =
  Q3624078, no end-time qualifier), it has not been dissolved (no P576), and it has
  an M49 or ISO alpha-3 code. Requiring a code drops the historical and regime items
  that Wikidata also tags this way, such as "French Fifth Republic" and "Kingdom of
  Laos". If two items resolve to the same M49 country (e.g. Netherlands and
  Kingdom of the Netherlands), the one with a direct P2082 claim is kept.
  Items listed in EXTRA_COUNTRIES are always added (currently only Denmark; see below).

Countries are matched to the UN table by M49 code first and by ISO alpha-3
second. Countries not in the table (e.g. Kosovo, Taiwan) are kept with empty
region columns unless listed in MANUAL_REGIONS (source="manual"); any still
without a region are reported at the end.
"""

import html
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "reference" / "regions.csv"

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
M49_URL = "https://unstats.un.org/unsd/methodology/m49/overview/"
HEADERS = {
    "User-Agent": "COMP440-PopQA-Research/0.1 (Macalester College course project; "
                  "python-requests) ",
}

COUNTRY_QUERY = """
SELECT ?c ?cLabel ?inUN ?isSov
       (GROUP_CONCAT(DISTINCT ?m49; separator="|") AS ?m49s)
       (GROUP_CONCAT(DISTINCT ?iso3; separator="|") AS ?iso3s)
WHERE {
  { ?c p:P463 ?ms . ?ms ps:P463 wd:Q1065 .
    FILTER NOT EXISTS { ?ms pq:P582 ?msEnd }
    BIND(true AS ?inUN) }
  UNION
  { VALUES ?c { %s } BIND(true AS ?isSov) }
  UNION
  { ?c p:P31 ?st . ?st ps:P31 wd:Q3624078 .
    FILTER NOT EXISTS { ?st pq:P582 ?stEnd }
    BIND(true AS ?isSov) }
  FILTER NOT EXISTS { ?c wdt:P576 ?dissolved }
  OPTIONAL { ?c wdt:P2082 ?m49 }
  OPTIONAL { ?c wdt:P298 ?iso3 }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
GROUP BY ?c ?cLabel ?inUN ?isSov
"""

# Countries whose own Wikidata item is not tagged as a UN member or sovereign
# state, because Wikidata puts those claims on a "realm" item with no country
# codes. Q35 Denmark (the realm is Q756617 Kingdom of Denmark) is the item that
# M49 code 208 and most citizenship (P27) claims use.
EXTRA_COUNTRIES = ["Q35"]

# Countries absent from the UN M49 table, assigned by hand (source="manual").
# The UN does not classify them as developed/developing, so global_south follows
# the IMF World Economic Outlook grouping: Taiwan is an "advanced economy",
# Kosovo an "emerging market and developing economy".
MANUAL_REGIONS = {  # qid: (region, global_south)
    "Q1246": ("Europe", True),  # Kosovo
    "Q865": ("Asia", False),    # Taiwan
}

REGION_ORDER = ["Africa", "Northern America", "Latin America & Caribbean",
                "Asia", "Europe", "Oceania"]
DETAIL_ORDER = ["Africa", "Northern America", "Caribbean", "Central America",
                "South America", "Asia", "Europe", "Oceania"]


def get(url, retries=4, **kwargs):
    for attempt in range(retries):
        r = requests.get(url, headers={**HEADERS, **kwargs.pop("headers", {})},
                         timeout=120, **kwargs)
        if r.status_code == 429 or r.status_code >= 500:
            wait = int(r.headers.get("Retry-After", 5 * (attempt + 1)))
            print(f"  HTTP {r.status_code} from {url}, retrying in {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def fetch_m49_table():
    """Parse the English table on the UN M49 overview page into a DataFrame."""
    page = get(M49_URL).text
    start = page.find('id = "downloadTableEN"')
    if start < 0:
        raise RuntimeError("Could not find the English M49 table on the UN page")
    table = page[start:page.find("</table>", start)]

    def cells(row):
        raw = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
        return [html.unescape(re.sub(r"<[^>]+>|\s+", " ", c)).strip() for c in raw]

    rows = [cells(r) for r in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S)]
    header, body = rows[0], [r for r in rows[1:] if len(r) == len(rows[0])]
    df = pd.DataFrame(body, columns=header)

    expected = ["Region Name", "Sub-region Name", "Intermediate Region Name",
                "Country or Area", "M49 Code", "ISO-alpha3 Code",
                "Developed / Developing Countries"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise RuntimeError(f"M49 table is missing columns: {missing}")
    return df


def fetch_wikidata_countries():
    extra = " ".join(f"wd:{q}" for q in EXTRA_COUNTRIES)
    r = get(SPARQL_ENDPOINT, params={"query": COUNTRY_QUERY % extra},
            headers={"Accept": "application/sparql-results+json"})
    records = {}
    for b in r.json()["results"]["bindings"]:
        qid = b["c"]["value"].rsplit("/", 1)[1]
        rec = records.setdefault(qid, {
            "qid": qid, "country": b["cLabel"]["value"],
            "in_un": False, "is_sovereign": False, "m49s": set(), "iso3s": set(),
        })
        rec["in_un"] |= "inUN" in b
        rec["is_sovereign"] |= "isSov" in b
        for key, field in (("m49s", "m49s"), ("iso3s", "iso3s")):
            vals = b.get(field, {}).get("value", "")
            rec[key] |= {v.strip() for v in vals.split("|") if v.strip()}
    return list(records.values())


def to_region(region, subregion):
    if region == "Americas":
        return {"Northern America": "Northern America",
                "Latin America and the Caribbean": "Latin America & Caribbean"
                }.get(subregion)
    return region if region in REGION_ORDER else None


def main():
    print("Fetching UN M49 overview table ...")
    m49 = fetch_m49_table()
    m49 = m49[m49["M49 Code"] != ""]
    by_code = {row["M49 Code"].zfill(3): row for _, row in m49.iterrows()}
    by_iso3 = {row["ISO-alpha3 Code"]: row for _, row in m49.iterrows()
               if row["ISO-alpha3 Code"]}
    print(f"  {len(m49)} countries/areas in M49 table")

    print("Querying Wikidata for current sovereign states ...")
    candidates = fetch_wikidata_countries()
    print(f"  {len(candidates)} candidate items")

    rows = []
    for c in candidates:
        if not (c["m49s"] or c["iso3s"]):
            continue  # historical/regime items without country codes
        un_row, match = None, ""
        for code in sorted(c["m49s"]):
            if code.zfill(3) in by_code:
                un_row, match = by_code[code.zfill(3)], "m49"
                break
        if un_row is None:
            for iso in sorted(c["iso3s"]):
                if iso in by_iso3:
                    un_row, match = by_iso3[iso], "iso3"
                    break

        out = {
            "qid": c["qid"],
            "country": c["country"],
            "m49_code": un_row["M49 Code"] if un_row is not None
                        else "|".join(sorted(c["m49s"])),
            "iso3": un_row["ISO-alpha3 Code"] if un_row is not None
                    else "|".join(sorted(c["iso3s"])),
            "un_member": c["in_un"],
            "m49_name": "", "m49_region": "", "m49_subregion": "",
            "m49_intermediate_region": "", "region": None, "region_detail": None,
            "global_south": pd.NA, "match": match,
        }
        if un_row is not None:
            region = to_region(un_row["Region Name"], un_row["Sub-region Name"])
            detail = (un_row["Intermediate Region Name"]
                      if region == "Latin America & Caribbean" else region)
            dev = un_row["Developed / Developing Countries"]
            out.update({
                "m49_name": un_row["Country or Area"],
                "m49_region": un_row["Region Name"],
                "m49_subregion": un_row["Sub-region Name"],
                "m49_intermediate_region": un_row["Intermediate Region Name"],
                "region": region,
                "region_detail": detail or None,
                "global_south": {"Developing": True, "Developed": False}.get(dev, pd.NA),
            })
        rows.append(out)

    df = pd.DataFrame(rows)

    # Two Wikidata items for one M49 country: keep the direct P2082 match,
    # then UN members, then the lower QID number for determinism.
    df["_qnum"] = df["qid"].str[1:].astype(int)
    df["_rank"] = (df["match"] != "m49").astype(int) * 2 + (~df["un_member"]).astype(int)
    df = df.sort_values(["_rank", "_qnum"])
    matched = df[df["match"] != ""]
    dupes = matched[matched.duplicated("m49_code", keep="first")]
    for _, d in dupes.iterrows():
        kept = matched[matched["m49_code"] == d["m49_code"]].iloc[0]
        print(f"  dropping duplicate {d['qid']} ({d['country']}) -> "
              f"kept {kept['qid']} ({kept['country']})")
    df = df.drop(dupes.index).drop(columns=["_qnum", "_rank", "match"])

    df["global_south"] = df["global_south"].astype("boolean")
    df["source"] = "M49"
    for qid, (region, south) in MANUAL_REGIONS.items():
        mask = df["qid"] == qid
        if not mask.any():
            print(f"  WARNING: manual region for {qid} but it is not in the country list")
            continue
        if df.loc[mask, "region"].notna().any():
            print(f"  WARNING: {qid} already has an M49 region; manual override skipped")
            continue
        df.loc[mask, ["region", "region_detail", "source"]] = [region, region, "manual"]
        df.loc[mask, "global_south"] = south
    df = df.sort_values("country").reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(df)} countries to {OUT_PATH.relative_to(ROOT)}")

    print("\nCountries per region:")
    print(df["region"].value_counts().reindex(REGION_ORDER, fill_value=0).to_string())
    print("\nCountries per region_detail:")
    print(df["region_detail"].value_counts().reindex(DETAIL_ORDER, fill_value=0).to_string())
    print("\nGlobal South:")
    print(df["global_south"].value_counts(dropna=False).to_string())

    none = df[df["region"].isna()]
    print(f"\nCountries with no region ({len(none)}):")
    for _, r in none.iterrows():
        print(f"  {r['qid']}  {r['country']}  (m49={r['m49_code'] or '-'}, "
              f"iso3={r['iso3'] or '-'})")


if __name__ == "__main__":
    main()
