"""Build a stratified sample of people from Wikidata (via QLever).

Data source: the QLever Wikidata endpoint (https://qlever.cs.uni-freiburg.de/api/wikidata).
Its index reports: "Complete Wikidata, TTL from https://dumps.wikimedia.org/wikidatawiki/
entities (latest-all.ttl.bz2 from 10.08.2026 munged, latest-lexemes.ttl.bz2 from
07.08.2026), updated LIVE" -- i.e. the 10 Aug 2026 Wikidata dump with live edits applied
since. The index description is printed at the start of each run; cite it with the run
date. (WDQS, query.wikidata.org, could not answer the per-country queries for large
countries within its 60 s limit, even split by year.) Birthplace countries for the
citizenship-unresolved minority come from WDQS via assign_region.fetch_birthplace_countries.

Step A  Candidate pool. One query per citizenship QID in regions.csv,
        country_aliases.csv and historical_states.csv, for people who are human
        (P31=Q5), female/male (P21), born 1900 or later (P569), and have an English
        Wikipedia article. A query that times out is split by birth decade (and a
        decade that times out, by year). Each query's rows are cached in
        data/raw/candidates_qlever/ so reruns skip finished queries.
        Output: data/processed/candidate_pool.csv, one row per person.
Step B  Regions, via assign_region.assign_person / add_region_columns. The
        citizenship rule runs first; birthplace countries (P19 -> P17) are fetched,
        batched and cached, only for people whose citizenships do not give a single
        value for region, region_detail or global_south. Then drops people with
        conflicting gender (both female and male), ambiguous region, or no English
        description.
Step C  Coverage level: sitelink tertile cutoffs computed once over the remaining
        pool; low <= q1 < medium <= q2 < high.
Step D  Equal-allocation stratified sample over region x gender x coverage, fixed
        seed. size // n_cells per cell; the remainder goes to randomly chosen cells
        (one extra each). Cells smaller than their target are taken whole.
        Output: data/processed/study_sample_pilot.csv (--size 200),
        study_sample_full.csv (--size 3000), otherwise study_sample_n<size>.csv.
        Each drawn person's enwiki_title is then checked with the Action API: if the
        article is missing or the title redirects to a page whose Wikidata item is
        not the person's own (a list, team, ...), the person is replaced by another
        random member of the same cell (separate seeded stream, so the rest of the
        draw is unchanged), until every cell is full or out of candidates.

Birth dates: a person matches if any birth date has year >= 1900; birth_date is the
earliest such date. Pre-1900 alternative dates are not seen, so a person with
conflicting dates 1899/1901 is kept.

Usage: python3 src/sample_people.py --size 200
"""

import argparse
import io
import threading
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import fetch_coverage as fc
from assign_region import (AMBIGUOUS, FIELDS, REF, add_region_columns, assign_person,
                           fetch_birthplace_countries, load_country_map, split)

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "raw" / "candidates_qlever"
BIRTH_CACHE = CACHE_DIR / "birthplace_countries.csv"
POOL_PATH = ROOT / "data" / "processed" / "candidate_pool.csv"
OUT_DIR = ROOT / "data" / "processed"

ENDPOINT = "https://qlever.cs.uni-freiburg.de/api/wikidata"
HEADERS = {
    "User-Agent": "COMP440-PopQA-Research/0.1 (Macalester College course project; "
                  "python-requests)",
    "Accept": "text/csv",
}
SEP = "|"
SEED = 440
PAUSE = 1        # seconds between queries
RETRIES = 8
BLANK = "/.well-known/genid/"  # Wikidata "unknown value" (blank node)
STREAM_ERROR = "An error has occurred while exporting the query result"
DEADLINE = 300   # seconds for a whole query (request + download)
BIRTH_CHUNK = 1000  # people per cached birthplace-lookup chunk (100 per WDQS query)
MIN_YEAR, MAX_YEAR = 1900, 2100
GENDERS = {"Q6581072": "female", "Q6581097": "male"}
LEVELS = ["low", "medium", "high"]

PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""
# QLever needs YEAR() here: comparing ?birth to an xsd:dateTime literal returns nothing.
QUERY = PREFIXES + """
SELECT ?person ?gender ?birth ?sitelinks ?title ?label ?description ?cit ?place
WHERE {
  ?person wdt:P27 wd:%(qid)s .
  ?person wdt:P31 wd:Q5 .
  ?person wdt:P21 ?gender . VALUES ?gender { wd:Q6581072 wd:Q6581097 }
  ?person wdt:P569 ?birth .
  FILTER(isLiteral(?birth) && YEAR(?birth) >= %(lo)d && YEAR(?birth) < %(hi)d)
  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> ;
           schema:name ?title .
  ?person wikibase:sitelinks ?sitelinks .
  ?person wdt:P27 ?cit .
  OPTIONAL { ?person rdfs:label ?label . FILTER(LANG(?label) = "en") }
  OPTIONAL { ?person schema:description ?description . FILTER(LANG(?description) = "en") }
  OPTIONAL { ?person wdt:P19 ?place . }
}
"""
COLUMNS = ["person", "gender", "birth", "sitelinks", "title", "label", "description",
           "cit", "place"]


class QueryTimeout(Exception):
    pass


# ---------------------------------------------------------------- Step A

def snapshot_info():
    r = requests.get(ENDPOINT, params={"cmd": "stats"}, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json().get("name-index", "(unknown)")


def post_with_deadline(query):
    """POST the query; raise QueryTimeout if request + download exceed DEADLINE."""
    box = {}

    def work():
        try:
            box["r"] = requests.post(ENDPOINT, data={"query": query}, headers=HEADERS,
                                     timeout=(30, DEADLINE))
        except Exception as e:  # handed back to the caller
            box["e"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(DEADLINE)
    if t.is_alive():
        raise QueryTimeout()
    if "e" in box:
        if isinstance(box["e"], requests.Timeout):
            raise QueryTimeout()
        raise box["e"]
    return box["r"]


def run_query(query):
    """Return the result as a DataFrame of strings. QueryTimeout on a timeout."""
    for attempt in range(RETRIES):
        try:
            r = post_with_deadline(query)
        except requests.ConnectionError as e:
            print(f"    connection error ({e.__class__.__name__}), retrying")
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code != 200 and "timeout" in r.text.lower():
            raise QueryTimeout()
        if r.status_code == 429 or r.status_code >= 500:
            wait = max(int(r.headers.get("Retry-After", 0)), 15 * (attempt + 1))
            print(f"    HTTP {r.status_code}, retrying in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        text = r.content.decode("utf-8")
        # QLever reports errors during streaming by appending them to a 200 response.
        if STREAM_ERROR in text:
            message = text[text.index(STREAM_ERROR):][:300].replace("\n", " ")
            if "timed out" in message.lower() or "timeout" in message.lower():
                raise QueryTimeout()
            print(f"    incomplete result ({message}), retrying")
            time.sleep(15 * (attempt + 1))
            continue
        return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    raise RuntimeError(f"query failed after {RETRIES} attempts")


def check_rows(df, name):
    """Fail loudly if any row is malformed (e.g. a truncated or error-appended result)."""
    ok = (df["person"].str.fullmatch(r"Q\d+") & df["gender"].isin(list(GENDERS))
          & df["sitelinks"].str.fullmatch(r"\d+")
          & (df["cit"].str.fullmatch(r"Q\d+") | df["cit"].str.contains(BLANK, regex=False)))
    if not ok.all():
        raise RuntimeError(f"{name}: {(~ok).sum()} malformed rows; delete its cache file "
                           f"and rerun")


def short(series):
    return series.str.replace(r"^http://www\.wikidata\.org/entity/", "", regex=True)


def fetch_range(qid, lo, hi):
    """Rows for one citizenship QID and birth years [lo, hi), cached; splits on timeout."""
    name = qid if (lo, hi) == (MIN_YEAR, MAX_YEAR) else f"{qid}_{lo}_{hi}"
    cache, marker = CACHE_DIR / f"{name}.csv", CACHE_DIR / f"{name}.timeout"
    if cache.exists():
        df = pd.read_csv(cache, dtype=str, keep_default_na=False)
        check_rows(df, cache.name)
        return [df]
    if not marker.exists():
        try:
            df = run_query(QUERY % {"qid": qid, "lo": lo, "hi": hi})
            time.sleep(PAUSE)
            df = df.reindex(columns=COLUMNS, fill_value="")
            for c in ["person", "gender", "cit", "place"]:
                df[c] = short(df[c])
            check_rows(df, name)
            tmp = cache.with_suffix(".tmp")
            df.to_csv(tmp, index=False)
            tmp.rename(cache)
            print(f"  {name}: {df['person'].nunique()} people")
            return [df]
        except QueryTimeout:
            marker.touch()
            print(f"  {name}: timed out, splitting")
            time.sleep(PAUSE * 10)
    if hi - lo <= 1:
        raise RuntimeError(f"{name} timed out even for a single year")
    if (lo, hi) == (MIN_YEAR, MAX_YEAR):
        edges = list(range(MIN_YEAR, 2030, 10)) + [MAX_YEAR]
    else:
        edges = list(range(lo, hi)) + [hi]
    return [df for a, b in zip(edges, edges[1:]) for df in fetch_range(qid, a, b)]


def citizenship_qids():
    regions = pd.read_csv(REF / "regions.csv", dtype=str, keep_default_na=False)
    aliases = pd.read_csv(REF / "country_aliases.csv", dtype=str, keep_default_na=False)
    hist = pd.read_csv(REF / "historical_states.csv", dtype=str, keep_default_na=False)
    return list(dict.fromkeys([*regions["qid"], *aliases["alias_qid"],
                               *hist["historical_qid"]]))


def joined(values):
    return SEP.join(sorted({v for v in values if v}))


def build_pool():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    qids = citizenship_qids()
    print(f"Step A: querying {len(qids)} citizenship QIDs (cache: "
          f"{CACHE_DIR.relative_to(ROOT)})")
    frames = []
    for i, qid in enumerate(qids, 1):
        print(f"[{i}/{len(qids)}] {qid}")
        frames += fetch_range(qid, MIN_YEAR, MAX_YEAR)
    rows = pd.concat(frames, ignore_index=True)

    g = rows.groupby("person", sort=True)
    pool = pd.DataFrame({
        "label": g["label"].max(),
        "description": g["description"].max(),
        "gender": g["gender"].agg(lambda s: joined(GENDERS[v] for v in s)),
        # "unknown value" citizenships are blank nodes, not QIDs: leave them out
        "citizenship_qids": g["cit"].agg(lambda s: joined(v for v in s if BLANK not in v)),
        "birth_date": g["birth"].min().str[:10],
        "birthplace_qids": g["place"].agg(joined),
        "n_sitelinks": g["sitelinks"].agg(lambda s: max(int(v) for v in s)),
        "enwiki_title": g["title"].max(),
    }).rename_axis("qid").reset_index()
    pool.to_csv(POOL_PATH, index=False)
    print(f"Saved {len(pool)} people to {POOL_PATH.relative_to(ROOT)}\n")
    return pool


# ---------------------------------------------------------------- Steps B-D

def crosstab(df):
    return pd.crosstab(df["region"], df["gender"], margins=True,
                       margins_name="total").to_string()


def indent(text):
    return "  " + text.replace("\n", "\n  ")


def birthplace_countries(qids):
    """{qid: [birthplace-country QIDs]} via WDQS, cached in BIRTH_CACHE."""
    cache = (pd.read_csv(BIRTH_CACHE, dtype=str, keep_default_na=False)
             if BIRTH_CACHE.exists() else pd.DataFrame(columns=["qid", "countries"]))
    known = dict(zip(cache["qid"], cache["countries"]))
    todo = [q for q in qids if q not in known]
    print(f"  birthplace lookup: {len(qids) - len(todo)} cached, {len(todo)} to fetch")
    for i in range(0, len(todo), BIRTH_CHUNK):
        chunk = todo[i:i + BIRTH_CHUNK]
        found = fetch_birthplace_countries(chunk)
        new = pd.DataFrame({"qid": chunk,
                            "countries": [SEP.join(found.get(q, [])) for q in chunk]})
        new.to_csv(BIRTH_CACHE, mode="a", header=not BIRTH_CACHE.exists(), index=False)
        known.update(zip(new["qid"], new["countries"]))
        print(f"    fetched {min(i + BIRTH_CHUNK, len(todo))}/{len(todo)}")
    return {q: split(known[q]) for q in qids}


def assign_regions(pool):
    print("Step B: assigning regions")
    to_modern, regions = load_country_map()
    pool = pool.copy()
    cit_lists = [split(c) for c in pool["citizenship_qids"]]

    # Citizenship rule alone first; birthplace only where it leaves any field unresolved.
    scratch = set()
    needs = [any(assign_person(c, [], to_modern, regions, scratch, scratch)[f"{f}_rule"]
                 != "citizenship" for f in FIELDS) for c in cit_lists]
    need_qids = pool.loc[needs, "qid"].tolist()
    print(f"  {len(need_qids)} of {len(pool)} people need the birthplace lookup "
          f"(citizenship gives no single region, region_detail or global_south)")
    birth = birthplace_countries(need_qids)
    birth_lists = [birth.get(q, []) for q in pool["qid"]]
    pool["birthplace_country_qids"] = [SEP.join(b) for b in birth_lists]

    unmapped_cit, unmapped_birth = add_region_columns(
        pool, cit_lists, birth_lists, to_modern, regions)
    pool = pool.drop(columns=["region_detail_rule", "global_south_rule"])
    print(f"  QIDs not in any reference table (ignored): {len(unmapped_cit)} "
          f"citizenship, {len(unmapped_birth)} birthplace-country")
    print("  region_rule: " + ", ".join(
        f"{k} {v}" for k, v in pool["region_rule"].value_counts().items()))

    reasons = [
        ("conflicting gender (female and male)",
         pool["gender"].str.contains(SEP, regex=False)),
        ("ambiguous region", pool["region"] == AMBIGUOUS),
        ("no English description", pool["description"] == ""),
    ]
    keep = pd.Series(True, index=pool.index)
    for reason, mask in reasons:
        dropped = pool[keep & mask]
        keep &= ~mask
        print(f"\n  Dropped for {reason}: {len(dropped)}")
        if len(dropped):
            print(indent(crosstab(dropped)))
    pool = pool[keep].reset_index(drop=True)
    print(f"\n  Remaining: {len(pool)}\n" + indent(crosstab(pool)) + "\n")
    return pool


def label_coverage(pool):
    print("Step C: coverage levels (sitelink tertiles over the whole remaining pool)")
    q1, q2 = pool["n_sitelinks"].quantile([1 / 3, 2 / 3]).tolist()
    pool["coverage"] = np.select(
        [pool["n_sitelinks"] <= q1, pool["n_sitelinks"] <= q2], LEVELS[:2], LEVELS[2])
    print(f"  cutoffs: low <= {q1:g} < medium <= {q2:g} < high")
    print("  share of pool: " + ", ".join(
        f"{lvl} {pool['coverage'].eq(lvl).mean():.1%}" for lvl in LEVELS))
    print("\n  Pool size by region x gender x coverage:")
    table = pd.crosstab(pool["region"], [pool["gender"], pool["coverage"]])
    table = table.reindex(columns=pd.MultiIndex.from_product([["female", "male"], LEVELS]),
                          fill_value=0)
    print(indent(table.to_string()) + "\n")
    return pool


def not_own_biography(people):
    """QIDs whose enwiki_title is missing or redirects to a page about something else.

    Action API (prop=info|pageprops, redirects=1), 50 titles per request: a redirect
    is accepted only if the target page's Wikidata item is the person's own QID.
    """
    bad = set()
    for i in range(0, len(people), fc.BATCH):
        chunk = people.iloc[i:i + fc.BATCH]
        info = fc.page_info(chunk["enwiki_title"].tolist())
        for qid, title in zip(chunk["qid"], chunk["enwiki_title"]):
            _, page_id, is_redirect, _, item = info[title]
            if page_id is None or (is_redirect and item != qid):
                bad.add(qid)
    return bad


def draw_sample(pool, size):
    print(f"Step D: stratified sample, size {size}, seed {SEED}")
    rng = np.random.default_rng(SEED)
    regions = sorted(pool["region"].unique())
    cells = [(r, g, c) for r in regions for g in ["female", "male"] for c in LEVELS]
    base, extra = divmod(size, len(cells))
    targets = {cell: base for cell in cells}
    for i in rng.choice(len(cells), size=extra, replace=False):
        targets[cells[i]] += 1
    print(f"  {len(cells)} cells; target {base} per cell, +1 for {extra} random cells")

    groups = pool.sort_values("qid").groupby(["region", "gender", "coverage"])
    members = {cell: groups.get_group(cell) if cell in groups.groups else pool.iloc[:0]
               for cell in cells}
    parts = [members[cell].sample(n=min(targets[cell], len(members[cell])),
                                  random_state=rng) for cell in cells]
    sample = pd.concat(parts, ignore_index=True)

    # Replace people without an English Wikipedia biography of their own. Replacements
    # use a separate seeded stream, so everyone who passes keeps their place.
    replace_rng = np.random.default_rng([SEED, 1])
    cell_of = lambda df: list(zip(df["region"], df["gender"], df["coverage"]))
    tried = set(sample["qid"])
    replaced = {cell: 0 for cell in cells}
    to_check = sample
    while len(to_check):
        bad = not_own_biography(to_check)
        if not bad:
            break
        removed = sample[sample["qid"].isin(bad)]
        sample = sample[~sample["qid"].isin(bad)]
        new_parts = []
        for cell in cells:
            k = sum(c == cell for c in cell_of(removed))
            if not k:
                continue
            untried = members[cell][~members[cell]["qid"].isin(tried)]
            draw = untried.sample(n=min(k, len(untried)), random_state=replace_rng)
            tried |= set(draw["qid"])
            replaced[cell] += len(draw)
            new_parts.append(draw)
        to_check = pd.concat(new_parts) if new_parts else sample.iloc[:0]
        sample = pd.concat([sample, to_check])
    order = {cell: i for i, cell in enumerate(cells)}
    sample = (sample.assign(_cell=[order[c] for c in cell_of(sample)])
              .sort_values("_cell", kind="stable").drop(columns="_cell")
              .reset_index(drop=True))
    sample.insert(0, "stratum", sample["region"] + SEP + sample["gender"] + SEP
                  + sample["coverage"])

    print(f"  Replacements (drawn person's enwiki_title is missing or redirects to a "
          f"page about something else): {sum(replaced.values())} in total")
    table = pd.Series(replaced).unstack([1, 2]).reindex(
        columns=pd.MultiIndex.from_product([["female", "male"], LEVELS]))
    print(indent(table.to_string()))

    counts = dict(zip(*np.unique(np.array([order[c] for c in cell_of(sample)]),
                                 return_counts=True)))
    shortfalls = [(cell, targets[cell], counts.get(order[cell], 0)) for cell in cells
                  if counts.get(order[cell], 0) < targets[cell]]
    if shortfalls:
        print(f"  Shortfalls ({sum(t - a for _, t, a in shortfalls)} people short):")
        for cell, t, a in shortfalls:
            print(f"    {' x '.join(cell)}: target {t}, available {a}")
    else:
        print("  No shortfalls.")

    name = {200: "pilot", 3000: "full"}.get(size, f"n{size}")
    out = OUT_DIR / f"study_sample_{name}.csv"
    sample.to_csv(out, index=False)
    print(f"  Saved {len(sample)} people to {out.relative_to(ROOT)}\n")

    print("Final counts by region x gender x coverage:")
    table = pd.crosstab(sample["region"], [sample["gender"], sample["coverage"]],
                        margins=True, margins_name="total")
    print(table.to_string())
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--size", type=int, default=200,
                        help="total sample size (200 = pilot, 3000 = full)")
    args = parser.parse_args()

    print(f"Run date: {date.today().isoformat()}")
    print(f"QLever Wikidata index: {snapshot_info()}\n")
    pool = build_pool()
    pool = assign_regions(pool)
    pool = label_coverage(pool)
    draw_sample(pool, args.size)


if __name__ == "__main__":
    main()
