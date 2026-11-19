"""Collect English Wikipedia coverage measures for each person in a study sample.

Input:  data/processed/study_sample_pilot.csv (--sample pilot, default) or
        study_sample_full.csv (--sample full); uses qid and enwiki_title.
Output: data/processed/coverage_pilot.csv / coverage_full.csv, one row per person.

Measures
  page_id, length_bytes, resolved_title, is_redirect, is_biography_redirect
      MediaWiki Action API (action=query&prop=info|pageprops&redirects=1), 50
      titles per request. resolved_title follows title normalisation and redirects.
      is_biography_redirect (redirects only; empty otherwise) is True when the
      resolved page's Wikidata item (pageprops wikibase_item) is the person's own
      QID, and False when the title redirects to a page about something else (a
      list, team, delegation ...). For False, all five coverage measures are left
      missing with the note "redirects to non-biography": the person has no
      English Wikipedia biography of their own.
  n_wikipedias, n_sitelinks
      Wikidata wbgetentities (props=sitelinks), 50 QIDs per request.
      n_wikipedias counts sitelinks whose site ID ends in "wiki" (language
      Wikipedias), minus NON_WIKIPEDIA_SITES (Commons, Wikispecies, Meta, ...).
      n_sitelinks counts all sitelinks (Wikipedias and sister projects), measured
      now; it can differ slightly from the sample's n_sitelinks (QLever, used for
      stratification).
  edit_count, n_editors (+ *_capped)
      REST API /w/rest.php/v1/page/{title}/history/counts/{edits,editors} on the
      resolved title. The API stops counting at a limit (30,000 edits) and returns
      limit=true; *_capped records that, and the value is then a lower bound.
  pageviews_12m, pageviews_monthly_avg
      Wikimedia Pageviews API, per-article, en.wikipedia, all-access, agent=user,
      monthly, Nov 2025 - Oct 2026, on the resolved title. A 404 (no data) counts
      as 0 views. pageviews_monthly_avg = pageviews_12m / 12 (months before an
      article existed count as 0).
  measured_at
      Date the person's measures were fetched.

Progress is saved per person to data/raw/coverage_cache.jsonl (one JSON object per
line, including a "notes" list explaining missing values); people already in the
cache are skipped on rerun. Timing for each run that fetched anything is appended
to data/raw/coverage_runs.jsonl; the report sums over those runs. Requests pause
0.5 s between per-person REST and Pageviews calls, retry on 429/5xx (Retry-After, or at least 15, 30, 45 ... s; up to
8 attempts), and are abandoned and retried if they take longer than 2 minutes.

Analysis notes (pilot, 2026-11-19): edit_count and n_editors correlate at 0.96
(Spearman), so the analysis uses n_editors. n_wikipedias includes English Wikipedia
itself, so its minimum is 1.

After fetching, prints a report: timing, missing/capped values, medians and ranges
by region x gender, and Spearman correlations between the coverage measures.
--spot-check N also prints N random people with links for checking by hand.

Usage: python3 src/fetch_coverage.py [--sample pilot|full] [--spot-check 3]
                                     [--spot-check-qids Q1,Q2,...]
"""

import argparse
import json
import threading
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "data" / "raw" / "coverage_cache.jsonl"
RUNS_PATH = ROOT / "data" / "raw" / "coverage_runs.jsonl"
NON_BIO = "redirects to non-biography"
PROCESSED = ROOT / "data" / "processed"

HEADERS = {"User-Agent": "COMP440-coverage-study/0.1 "
                         "(https://github.com/mtutran/comp440-f26-tnn)"}
ACTION_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
REST_COUNTS = "https://en.wikipedia.org/w/rest.php/v1/page/{title}/history/counts/{kind}"
PAGEVIEWS = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
             "en.wikipedia/all-access/user/{title}/monthly/{start}/{end}")
PV_START, PV_END, PV_MONTHS = "2025110100", "2026103100", 12

BATCH = 50
PAUSE = 0.5
ATTEMPTS = 8
DEADLINE = 120  # seconds per request
NON_WIKIPEDIA_SITES = {
    "commonswiki", "specieswiki", "metawiki", "mediawikiwiki", "wikidatawiki",
    "sourceswiki", "incubatorwiki", "wikimaniawiki", "outreachwiki", "foundationwiki",
    "testwiki", "test2wiki", "testwikidatawiki", "wikifunctionswiki", "abstractwiki",
}
COLUMNS = ["qid", "enwiki_title", "resolved_title", "page_id", "is_redirect",
           "is_biography_redirect", "length_bytes", "edit_count", "edit_count_capped", "n_editors",
           "n_editors_capped", "pageviews_12m", "pageviews_monthly_avg",
           "n_wikipedias", "n_sitelinks", "measured_at"]
MEASURES = {"length_bytes": "length", "edit_count": "edits", "n_editors": "editors",
            "pageviews_12m": "page views", "n_wikipedias": "Wikipedias"}

STATS = {"rate_limit_wait": 0.0, "pause_wait": 0.0, "retries": 0, "deadline_retries": 0,
         "requests": 0}


# ---------------------------------------------------------------- HTTP

class Deadline(Exception):
    pass


def get_with_deadline(url, params=None):
    """GET with a hard wall-clock limit of DEADLINE seconds (request + body)."""
    box = {}

    def work():
        try:
            box["r"] = requests.get(url, params=params, headers=HEADERS,
                                    timeout=(30, DEADLINE))
        except Exception as e:  # handed back to the caller
            box["e"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(DEADLINE)
    if t.is_alive():
        raise Deadline()
    if "e" in box:
        raise box["e"]
    return box["r"]


def get(url, params=None, ok=(200,)):
    """GET with retries; returns the response once its status is in `ok`."""
    for attempt in range(1, ATTEMPTS + 1):
        STATS["requests"] += 1
        try:
            r = get_with_deadline(url, params)
        except (Deadline, requests.Timeout):
            STATS["deadline_retries"] += 1
            print(f"    request took over {DEADLINE}s, retrying ({attempt}/{ATTEMPTS})")
            continue
        except requests.ConnectionError as e:
            wait = 15 * attempt
            print(f"    {e.__class__.__name__}, retrying in {wait}s")
        else:
            if r.status_code in ok:
                return r
            if r.status_code != 429 and r.status_code < 500:
                raise RuntimeError(f"HTTP {r.status_code} for {r.url}: {r.text[:200]}")
            retry_after = r.headers.get("Retry-After", "")
            wait = max(int(retry_after) if retry_after.isdigit() else 0, 15 * attempt)
            print(f"    HTTP {r.status_code}, retrying in {wait}s ({attempt}/{ATTEMPTS})")
        STATS["retries"] += 1
        STATS["rate_limit_wait"] += wait
        time.sleep(wait)
    raise RuntimeError(f"giving up after {ATTEMPTS} attempts: {url}")


def pause():
    time.sleep(PAUSE)
    STATS["pause_wait"] += PAUSE


def path_title(title):
    return quote(title.replace(" ", "_"), safe="")


# ---------------------------------------------------------------- measures

def page_info(titles):
    """{input title: (resolved_title, page_id, is_redirect, length_bytes, item)}.

    item is the resolved page's Wikidata QID (pageprops wikibase_item), or None.
    """
    r = get(ACTION_API, {"action": "query", "prop": "info|pageprops",
                         "ppprop": "wikibase_item", "titles": "|".join(titles),
                         "redirects": 1, "format": "json", "formatversion": 2})
    q = r.json()["query"]
    normalized = {n["from"]: n["to"] for n in q.get("normalized", [])}
    redirects = {n["from"]: n["to"] for n in q.get("redirects", [])}
    pages = {p["title"]: p for p in q.get("pages", [])}
    out = {}
    for t in titles:
        resolved, hops = normalized.get(t, t), 0
        while resolved in redirects and hops < 10:
            resolved, hops = redirects[resolved], hops + 1
        p = pages.get(resolved, {})
        missing = p.get("missing", False) or "pageid" not in p
        out[t] = (resolved, None if missing else p["pageid"], hops > 0,
                  None if missing else p.get("length"),
                  p.get("pageprops", {}).get("wikibase_item"))
    return out


def wikipedia_counts(qids):
    """{qid: (n_wikipedias, n_sitelinks)}; follows Wikidata redirects."""
    r = get(WIKIDATA_API, {"action": "wbgetentities", "ids": "|".join(qids),
                           "props": "sitelinks", "format": "json"})
    out = {}
    for key, ent in r.json().get("entities", {}).items():
        sites = ent.get("sitelinks", {})
        n_wp = sum(1 for s in sites if s.endswith("wiki") and s not in NON_WIKIPEDIA_SITES)
        counts = None if "missing" in ent else (n_wp, len(sites))
        out[ent.get("redirects", {}).get("from", key)] = counts
        out.setdefault(key, counts)
    return out


def edit_counts(title):
    """{'edits': (count, capped), 'editors': (count, capped)}, or None on 404."""
    out = {}
    for kind in ["edits", "editors"]:
        pause()
        r = get(REST_COUNTS.format(title=path_title(title), kind=kind), ok=(200, 404))
        if r.status_code == 404:
            return None
        body = r.json()
        out[kind] = (body["count"], bool(body.get("limit", False)))
    return out


def pageviews(title):
    """(views over the 12 months, whether the API returned 404)."""
    pause()
    url = PAGEVIEWS.format(title=path_title(title), start=PV_START, end=PV_END)
    r = get(url, ok=(200, 404))
    if r.status_code == 404:
        return 0, True
    return sum(item["views"] for item in r.json().get("items", [])), False


def measure_chunk(chunk):
    """Fetch every measure for up to BATCH people; return one dict per person."""
    titles = [t for t in chunk["enwiki_title"] if t]
    info = page_info(titles) if titles else {}
    wp = wikipedia_counts(chunk["qid"].tolist())
    today = date.today().isoformat()
    rows = []
    for qid, title in zip(chunk["qid"], chunk["enwiki_title"]):
        notes = []
        resolved, page_id, is_redirect, length, item = info.get(
            title, (None, None, None, None, None))
        is_bio_redirect = (item == qid) if is_redirect and page_id is not None else None
        if not title:
            notes.append("no enwiki_title in sample")
        elif page_id is None:
            notes.append("article not found (Action API: missing)")
        counts = wp.get(qid)
        if counts is None:
            notes.append("Wikidata item not found")
        row = {"qid": qid, "enwiki_title": title, "resolved_title": resolved,
               "page_id": page_id, "is_redirect": is_redirect,
               "is_biography_redirect": is_bio_redirect, "length_bytes": length,
               "edit_count": None, "edit_count_capped": None, "n_editors": None,
               "n_editors_capped": None, "pageviews_12m": None,
               "pageviews_monthly_avg": None,
               "n_wikipedias": counts[0] if counts else None,
               "n_sitelinks": counts[1] if counts else None,
               "measured_at": today}
        if is_redirect:
            notes.append(f"enwiki_title redirects to {resolved}")
        if is_bio_redirect is False:
            notes.append(NON_BIO)
            row.update(page_id=None, length_bytes=None, n_wikipedias=None)
        elif page_id is not None:
            ec = edit_counts(resolved)
            if ec is None:
                notes.append("edit counts: REST 404")
            else:
                row["edit_count"], row["edit_count_capped"] = ec["edits"]
                row["n_editors"], row["n_editors_capped"] = ec["editors"]
                if ec["edits"][1]:
                    notes.append("edit_count capped by API limit")
                if ec["editors"][1]:
                    notes.append("n_editors capped by API limit")
            views, was_404 = pageviews(resolved)
            row["pageviews_12m"] = views
            row["pageviews_monthly_avg"] = round(views / PV_MONTHS, 1)
            if was_404:
                notes.append("pageviews: 404, counted as 0")
        row["notes"] = notes
        rows.append(row)
    return rows


# ---------------------------------------------------------------- run + report

def load_cache():
    if not CACHE_PATH.exists():
        return {}
    rows = [json.loads(line) for line in CACHE_PATH.read_text().splitlines() if line]
    return {r["qid"]: r for r in rows}


def fetch(sample):
    cache = load_cache()
    todo = sample[~sample["qid"].isin(cache)]
    print(f"{len(sample)} people; {len(sample) - len(todo)} cached, {len(todo)} to fetch")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(todo), BATCH):
        rows = measure_chunk(todo.iloc[i:i + BATCH])
        with CACHE_PATH.open("a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
                cache[row["qid"]] = row
        print(f"  {min(i + BATCH, len(todo))}/{len(todo)} done")
    return pd.DataFrame([cache[q] for q in sample["qid"]]), len(todo)


def report(cov, sample, sample_name):
    print("\n=== Timing (all fetch runs for this sample, from "
          f"{RUNS_PATH.relative_to(ROOT)})")
    runs = ([json.loads(x) for x in RUNS_PATH.read_text().splitlines() if x]
            if RUNS_PATH.exists() else [])
    runs = [r for r in runs if r.get("sample") == sample_name]
    if runs:
        t = {k: sum(r[k] for r in runs) for k in ["n_fetched", "elapsed_s", "requests",
             "rate_limit_wait", "retries", "deadline_retries", "pause_wait"]}
        print(f"  total run time: {t['elapsed_s'] / 60:.1f} min for {t['n_fetched']} "
              f"people ({t['requests']} requests, {len(runs)} run(s))")
        print(f"  waiting on rate limits / errors: {t['rate_limit_wait']:.0f} s "
              f"({t['retries']} retries, {t['deadline_retries']} over {DEADLINE}s); "
              f"fixed 0.5 s pauses: {t['pause_wait']:.0f} s")
        per = t["elapsed_s"] / t["n_fetched"]
        print(f"  {per:.2f} s per person -> about {per * 3000 / 60:.0f} min for 3,000 "
              f"people at this rate (more if rate limits kick in)")
    else:
        print("  no timing recorded")

    print("\n=== Missing or capped values")
    for col in ["length_bytes", "edit_count", "n_editors", "pageviews_12m",
                "n_wikipedias", "n_sitelinks"]:
        print(f"  {col}: {cov[col].isna().sum()} missing")
    print(f"  edit_count capped: {int(cov['edit_count_capped'].fillna(False).sum())}; "
          f"n_editors capped: {int(cov['n_editors_capped'].fillna(False).sum())}")
    print(f"  redirects: {int(cov['is_redirect'].fillna(False).sum())} (to a "
          f"non-biography: {int((cov['is_biography_redirect'] == False).sum())}); "
          f"pageviews 0: {int((cov['pageviews_12m'] == 0).sum())}")
    notes = pd.Series([n for ns in cov["notes"] for n in ns], dtype=str)
    if len(notes):
        print("  reasons (people):")
        reasons = notes.str.replace(r"^enwiki_title redirects to .*",
                                    "enwiki_title redirects to another title",
                                    regex=True).value_counts()
        for reason, n in reasons.items():
            print(f"    {n:4d}  {reason}")

    df = cov.merge(sample[["qid", "region", "gender"]], on="qid")
    print("\n=== Median [min - max] by region x gender")
    for col, name in MEASURES.items():
        g = df.groupby(["region", "gender"])[col]
        table = pd.DataFrame({"n": g.count(), "median": g.median(), "min": g.min(),
                              "max": g.max()})
        table[["median", "min", "max"]] = table[["median", "min", "max"]].round(0)
        print(f"\n  {name} ({col})")
        print("  " + table.astype({"median": "Int64", "min": "Int64", "max": "Int64"})
              .to_string().replace("\n", "\n  "))

    print("\n=== Spearman correlations (pairwise complete)")
    corr = df[list(MEASURES)].astype(float).corr(method="spearman").round(2)
    corr.index = corr.columns = list(MEASURES.values())
    print("  " + corr.to_string().replace("\n", "\n  "))


def spot_check(cov, sample, n, qids=None):
    df = cov.merge(sample[["qid", "label", "region", "gender", "coverage"]], on="qid")
    if qids:
        picks = df.set_index("qid").loc[qids].reset_index()
        print(f"\n=== Spot check ({len(picks)} chosen people)")
    else:
        picks = df.sample(n=n, random_state=np.random.default_rng())
        print(f"\n=== Spot check ({n} random people)")

    def fmt(v):
        return "missing" if pd.isna(v) else f"{v:,.0f}"

    for _, r in picks.iterrows():
        t = r["resolved_title"] or r["enwiki_title"]
        u = quote(t.replace(" ", "_"))
        print(f"\n  {r['label']} ({r['qid']}; {r['region']}, {r['gender']}, "
              f"{r['coverage']} coverage)")
        print(f"    length {fmt(r['length_bytes'])} bytes | edits {fmt(r['edit_count'])}"
              f"{' (capped)' if r['edit_count_capped'] is True else ''} | editors "
              f"{fmt(r['n_editors'])}{' (capped)' if r['n_editors_capped'] is True else ''}"
              f" | page views {fmt(r['pageviews_12m'])} (avg "
              f"{fmt(r['pageviews_monthly_avg'])}/month) | Wikipedias "
              f"{fmt(r['n_wikipedias'])} (all sitelinks {fmt(r['n_sitelinks'])})")
        print(f"    article:   https://en.wikipedia.org/wiki/{u}")
        print(f"    page info (length, edits, editors): "
              f"https://en.wikipedia.org/w/index.php?title={u}&action=info")
        print(f"    history:   https://en.wikipedia.org/w/index.php?title={u}&action=history")
        print(f"    pageviews: https://pageviews.wmcloud.org/?project=en.wikipedia.org"
              f"&platform=all-access&agent=user&start=2025-11-01&end=2026-10-31&pages={u}")
        print(f"    Wikidata:  https://www.wikidata.org/wiki/{r['qid']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--spot-check", type=int, default=0, metavar="N",
                        help="print N random people with links for manual checking")
    parser.add_argument("--spot-check-qids", default="", metavar="Q1,Q2",
                        help="print these people with links instead of random ones")
    args = parser.parse_args()

    sample = pd.read_csv(PROCESSED / f"study_sample_{args.sample}.csv", dtype=str,
                         keep_default_na=False)
    start = time.monotonic()
    cov, n_fetched = fetch(sample)
    elapsed = time.monotonic() - start
    if n_fetched:
        with RUNS_PATH.open("a") as f:
            f.write(json.dumps({"date": date.today().isoformat(), "sample": args.sample,
                                "n_fetched": n_fetched, "elapsed_s": round(elapsed, 1),
                                **STATS}) + "\n")

    out = PROCESSED / f"coverage_{args.sample}.csv"
    cov[COLUMNS].to_csv(out, index=False)
    print(f"Saved {len(cov)} rows to {out.relative_to(ROOT)}")

    report(cov, sample, args.sample)
    if args.spot_check or args.spot_check_qids:
        qids = args.spot_check_qids.split(",") if args.spot_check_qids else None
        spot_check(cov, sample, args.spot_check, qids)


if __name__ == "__main__":
    main()
