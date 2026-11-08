"""For 20 sampled people, fetch English Wikipedia article length and total edit count.

- Article length (bytes): MediaWiki Action API, action=query&prop=info
  (redirects followed, so we measure the real article, not a redirect stub).
- Edit count: REST API /page/{title}/history/counts/edits, using the resolved title.

Requests are sequential (one person at a time, 2 requests each) and timed, and the
timing is extrapolated to 3,000 people.
"""

import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "popqa.csv"
SAMPLE_PATH = ROOT / "data" / "processed" / "people_sample.csv"
OUT_PATH = ROOT / "data" / "processed" / "people_wiki_stats.csv"

ACTION_API = "https://en.wikipedia.org/w/api.php"
REST_COUNTS = "https://en.wikipedia.org/w/rest.php/v1/page/{}/history/counts/edits"
HEADERS = {"User-Agent": "COMP440-PopQA-Research/0.1 (Macalester College course project; "
                         "python-requests)"}
N_PEOPLE = 20
SEED = 440
TARGET_N = 3000

session = requests.Session()
session.headers.update(HEADERS)
stats = {"n_429": 0, "wait_s": 0.0}


def polite_get(url, params=None, max_retries=6):
    """GET that honors HTTP 429 Retry-After, recording how much time was spent waiting."""
    for attempt in range(max_retries):
        r = session.get(url, params=params, timeout=30)
        if r.status_code != 429:
            return r
        wait = float(r.headers.get("Retry-After", 2 ** attempt))
        stats["n_429"] += 1
        stats["wait_s"] += wait
        print(f"  429 on {url.split('/w/')[1][:60]}... waiting {wait:.0f}s")
        time.sleep(wait)
    r.raise_for_status()
    return r


def get_page_info(title):
    """Return (resolved_title, length_bytes) via the Action API, following redirects."""
    r = polite_get(ACTION_API, params={
        "action": "query", "prop": "info", "titles": title,
        "redirects": 1, "format": "json", "formatversion": 2,
    })
    r.raise_for_status()
    page = r.json()["query"]["pages"][0]
    if page.get("missing") or page.get("invalid"):
        return None, None
    return page["title"], page["length"]


def get_edit_count(title):
    """Return (count, capped) from the REST history counts endpoint."""
    url = REST_COUNTS.format(quote(title.replace(" ", "_"), safe=""))
    r = polite_get(url)
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    data = r.json()
    return data["count"], data.get("limit", False)


def main():
    titles = (pd.read_csv(RAW_PATH)[["subj_id", "s_wiki_title"]]
              .drop_duplicates("subj_id"))
    people = (pd.read_csv(SAMPLE_PATH)
              .sample(n=N_PEOPLE, random_state=SEED)
              .merge(titles, on="subj_id", how="left"))

    rows, per_person = [], []
    t_start = time.perf_counter()
    for title in people["s_wiki_title"]:
        t0 = time.perf_counter()
        resolved, length = get_page_info(title)
        count, capped = get_edit_count(resolved) if resolved else (None, None)
        per_person.append(time.perf_counter() - t0)
        rows.append({"resolved_title": resolved, "length_bytes": length,
                     "edit_count": count, "edit_count_capped": capped})
    total = time.perf_counter() - t_start

    out = pd.concat([people.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["length_bytes"] = out["length_bytes"].astype("Int64")
    out["edit_count"] = out["edit_count"].astype("Int64")
    out["n_sitelinks"] = out["n_sitelinks"].astype("Int64")
    out.to_csv(OUT_PATH, index=False)

    pd.set_option("display.width", 200)
    print(out[["subj", "s_wiki_title", "s_pop", "n_sitelinks", "length_bytes", "edit_count"]]
          .to_string(index=False))
    print(f"\nSaved {len(out)} rows to {OUT_PATH.relative_to(ROOT)}")
    redirected = (out["resolved_title"] != out["s_wiki_title"]).sum()
    print(f"Titles that redirected / were missing: {redirected}")
    print(f"Edit counts hitting the API cap: {int(out['edit_count_capped'].fillna(False).sum())}")

    # --- Timing ---
    n_req = 2 * N_PEOPLE
    s = pd.Series(per_person)
    print(f"\nTiming ({n_req} requests, {N_PEOPLE} people, sequential):")
    print(f"  total wall time: {total:.2f} s")
    print(f"  of which waiting on 429 Retry-After: {stats['wait_s']:.0f} s "
          f"({stats['n_429']} rate-limited responses)")
    active = total - stats["wait_s"]
    print(f"  excluding waits: {active:.2f} s")
    print(f"  per person: mean {s.mean():.3f} s, median {s.median():.3f} s, max {s.max():.3f} s")
    est = total / N_PEOPLE * TARGET_N
    est_active = active / N_PEOPLE * TARGET_N
    print(f"  estimate for {TARGET_N:,} people at observed rate (incl. waits): "
          f"{est:.0f} s ≈ {est / 60:.1f} min")
    print(f"  estimate for {TARGET_N:,} people if never rate-limited: "
          f"{est_active:.0f} s ≈ {est_active / 60:.1f} min")


if __name__ == "__main__":
    main()
