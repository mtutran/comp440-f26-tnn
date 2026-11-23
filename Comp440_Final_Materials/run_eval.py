#!/usr/bin/env python3
"""Ask the LLM every (person x question type), grade the answers, and write
results, a run summary (runtime, tokens, cost, outcomes) and hand-check sheets.

Pilot (step 9), 200 people x 2 question types:
    python run_eval.py --people data/sample.csv --limit 200 \
        --provider openai --model gpt-4o-mini \
        --price-in 0.15 --price-out 0.60 --out results/pilot_gpt-4o-mini

Dry run with no API key and no cost:
    python run_eval.py --people data/example_people.csv --provider mock --model mock

Because responses are cached, re-running the same command after fixing the
grader (step 11) re-grades everything without paying for new API calls.

Input CSV (one row per person). Default column names, change with --col-*:
    qid           Wikidata ID, e.g. Q7186
    name          full name
    description   Wikidata description (short, disambiguating)
    birth_year    gold year(s): 1867, an ISO date, or several joined with |
    citizenship   gold country label(s), several joined with |
    is_deceased   optional true/false; changes "is" -> "was" in the question
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from llm_eval.descriptions import DescriptionCleaner
from llm_eval.grader import Grader
from llm_eval.query import LLMClient
from llm_eval.templates import QUESTION_TYPES, TEMPLATE_VERSION, build_question

HANDCHECKERS = ["Tu", "Nadezhda", "Nayla"]


def to_bool(v):
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--people", required=True, help="CSV of sampled people")
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic", "mock"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", help="OpenAI-compatible base URL (Together, Groq, Ollama...)")
    ap.add_argument("--api-key-env", help="env var holding the API key")
    ap.add_argument("--question-types", nargs="+", default=list(QUESTION_TYPES),
                    choices=list(QUESTION_TYPES))
    ap.add_argument("--limit", type=int, help="only the first N people (e.g. 200 for the pilot)")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests")
    ap.add_argument("--max-tokens", type=int, default=150,
                    help="raise this (e.g. 1024) for reasoning models")
    ap.add_argument("--temperature", type=float, default=None,
                    help="leave unset for Claude Sonnet 5.5, which doesn't accept it")
    ap.add_argument("--cache", default="cache/llm_cache.sqlite")
    ap.add_argument("--price-in", type=float, default=0.0, help="USD per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=0.0, help="USD per 1M output tokens")
    ap.add_argument("--max-cost", type=float,
                    help="hard cap in USD for THIS run; stops making calls before exceeding it")
    ap.add_argument("--max-total-cost", type=float,
                    help="hard cap in USD across ALL runs of this model (counted from the cache)")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    ap.add_argument("--raw-descriptions", action="store_true",
                    help="don't remove answer leaks from descriptions (not recommended)")
    ap.add_argument("--aliases-csv", help="extra country aliases (canonical,alias)")
    ap.add_argument("--historical-csv", help="Tu's historical->modern lookup (historical,modern)")
    ap.add_argument("--year-tolerance", type=int, default=0)
    ap.add_argument("--handcheck-n", type=int, default=51,
                    help="answers to sample for hand-checking, split across the team")
    ap.add_argument("--seed", type=int, default=440)
    ap.add_argument("--out", default=None, help="output directory")
    for c in ["qid", "name", "description", "birth_year", "citizenship", "is_deceased"]:
        ap.add_argument(f"--col-{c.replace('_', '-')}", default=c, dest=f"col_{c}")
    args = ap.parse_args()

    out = args.out or os.path.join("results", f"{args.provider}_{args.model}".replace("/", "_"))
    os.makedirs(out, exist_ok=True)

    people = pd.read_csv(args.people, dtype=str)
    if args.limit:
        people = people.head(args.limit)
    gold_col = {"birth_year": args.col_birth_year, "citizenship": args.col_citizenship}
    for c in [args.col_qid, args.col_name] + [gold_col[q] for q in args.question_types]:
        if c not in people.columns:
            raise SystemExit(f"column {c!r} not in {args.people}; columns are {list(people.columns)}")

    try:
        client = make_client(args)
    except RuntimeError as e:
        raise SystemExit(f"ERROR: {e}")
    grader = Grader(args.aliases_csv, args.historical_csv, args.year_tolerance)
    _run(args, people, gold_col, client, grader, out)


def make_client(args):
    return LLMClient(args.provider, args.model, cache_path=args.cache,
                       base_url=args.base_url, api_key_env=args.api_key_env,
                       temperature=args.temperature, max_tokens=args.max_tokens,
                       price_in=args.price_in, price_out=args.price_out,
                       max_cost=args.max_cost, max_total_cost=args.max_total_cost)


def _run(args, people, gold_col, client, grader, out):

    cleaner = False if args.raw_descriptions else DescriptionCleaner(grader.resolver)
    jobs = []
    for _, p in people.iterrows():
        desc = p.get(args.col_description) if args.col_description in people.columns else None
        dec = to_bool(p.get(args.col_is_deceased)) if args.col_is_deceased in people.columns else None
        for qt in args.question_types:
            jobs.append({
                "qid": p[args.col_qid], "name": p[args.col_name],
                "description": desc, "question_type": qt,
                "description_used": (desc if cleaner is False or pd.isna(desc)
                                     else cleaner.clean(qt, desc)),
                "gold": p[gold_col[qt]],
                "prompt": build_question(qt, p[args.col_name], desc, dec, cleaner=cleaner),
            })

    print(f"{len(people)} people x {len(args.question_types)} question types = {len(jobs)} questions "
          f"-> {args.provider}:{args.model}  (cache has {len(client.cache)} entries)")

    # ----- cost estimate and confirmation before any money is spent
    new = [j for j in jobs if client.cache.get(client._key(j["prompt"])) is None]
    worst = sum(client._worst_case(j["prompt"]) for j in new)
    print(f"New API calls needed: {len(new)} (the other {len(jobs) - len(new)} are cached, free)")
    if args.price_in or args.price_out:
        print(f"Worst-case cost of this run: ${worst:.2f}   "
              f"(already spent on this model, from the cache: ${client.spent_before:.2f})")
        if args.max_cost is not None:
            print(f"Run cap (--max-cost): ${args.max_cost:.2f}")
        if args.max_total_cost is not None:
            print(f"Lifetime cap (--max-total-cost): ${args.max_total_cost:.2f}")
    elif args.provider != "mock":
        print("WARNING: no --price-in/--price-out given, so cost can't be estimated or capped.")
    if new and args.provider != "mock" and not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit("Refusing to spend without confirmation; rerun with --yes.")
        if input("Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
            raise SystemExit("Cancelled; nothing was sent.")

    t0 = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(client.ask, j["prompt"]): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            j, r = futs[f], f.result()
            g = grader.grade(j["question_type"], r.text, j["gold"])
            rows.append({**j, "model": r.model, "provider": r.provider,
                         "template_version": TEMPLATE_VERSION, "response": r.text,
                         **g.as_row(), "latency_s": r.latency_s,
                         "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                         "cached": r.cached, "error": r.error,
                         "stop_reason": r.stop_reason})
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.perf_counter() - t0:.1f}s)")
    wall = time.perf_counter() - t0

    df = pd.DataFrame(rows).sort_values(["qid", "question_type"]).reset_index(drop=True)
    # API errors are not answers: mark them so they're excluded, not counted as wrong.
    df.loc[df["error"].notna(), ["outcome", "outcome_strict"]] = "api_error"
    df.loc[df["error"] == "budget_exceeded", ["outcome", "outcome_strict"]] = "not_asked"
    df.loc[df["error"] == "truncated", ["outcome", "outcome_strict"]] = "truncated"
    n_skipped = int((df["error"] == "budget_exceeded").sum())
    if n_skipped:
        print(f"\nSPENDING CAP REACHED: {n_skipped} questions were not asked. "
              "Raise the cap and rerun; answered questions are cached and won't be re-billed.")
    df.to_csv(os.path.join(out, "responses.csv"), index=False)

    # ----- summary: runtime, cost, outcomes
    fresh = df[~df["cached"] & df["error"].isna()]
    billed = df[~df["cached"] & df["input_tokens"].notna()]   # includes cut-off replies
    tin = int(billed["input_tokens"].fillna(0).sum())
    tout = int(billed["output_tokens"].fillna(0).sum())
    cost = tin / 1e6 * args.price_in + tout / 1e6 * args.price_out
    n_fresh = len(fresh)
    answered = df[df["error"].isna()]   # format and flag rates only make sense for real answers
    flagged = ((~answered["format_ok"]) | answered["unrecognized"] | answered["historical_match"]
               | answered["imprecise"] | (answered["n_pred"] > 1))
    summary = {
        "provider": args.provider, "model": args.model, "template_version": TEMPLATE_VERSION,
        "n_people": len(people), "n_questions": len(df),
        "n_api_calls": n_fresh, "n_cached": int(df["cached"].sum()),
        "n_errors": int(df["error"].notna().sum()),
        "n_truncated": int((df["error"] == "truncated").sum()),
        "wall_time_s": round(wall, 1),
        "mean_latency_s": round(float(fresh["latency_s"].mean()), 3) if n_fresh else None,
        "input_tokens": tin, "output_tokens": tout, "est_cost_usd": round(cost, 4),
        "cost_per_question_usd": round(cost / len(billed), 6) if len(billed) else None,
        "projected_cost_per_1000_people_usd":
            round(cost / len(billed) * 1000 * len(args.question_types), 2) if len(billed) else None,
        "format_ok_rate": round(float(answered["format_ok"].mean()), 3) if len(answered) else None,
        "flagged_rate": round(float(flagged.mean()), 3) if len(answered) else None,
        "outcomes": {}, "outcomes_strict": {},
    }
    for qt, sub in df.groupby("question_type"):
        summary["outcomes"][qt] = sub["outcome"].value_counts(normalize=True).round(3).to_dict()
        summary["outcomes_strict"][qt] = sub["outcome_strict"].value_counts(normalize=True).round(3).to_dict()
        summary["outcomes"][qt]["n"] = len(sub)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    n_trunc = int((df["error"] == "truncated").sum())
    if n_trunc:
        print(f"\n{n_trunc} replies were CUT OFF at --max-tokens {args.max_tokens} and not graded. "
              "Rerun with a bigger --max-tokens (e.g. 300); they'll be asked again.")
    errs = df.loc[df["error"].notna() & ~df["error"].isin(["budget_exceeded", "truncated"]), "error"]
    if len(errs):
        print(f"\n{len(errs)} questions FAILED. Error message(s):")
        uniq = errs.astype(str).str.replace(r',?"request_id":"[^"]*"', "", regex=True).value_counts()
        for e, k in list(uniq.items())[:3]:
            print(f"  - ({k}x) {e[:400]}")
        print("Failed questions are not cached; fix the problem and rerun the same command.")
    elif n_fresh == 0 and len(df) and df["cached"].all():
        print("(Every answer came from the cache, so runtime and cost reflect no new API calls.)")

    # ----- hand-check sheets (step 10): stratified by question type and outcome,
    # over-sampling the flagged cases where the grader is most likely to be wrong.
    graded = df[~df["outcome"].isin(["api_error", "not_asked", "truncated"])].copy()
    graded["flagged"] = (~graded["format_ok"]) | graded["unrecognized"] | \
        graded["historical_match"] | graded["imprecise"] | (graded["n_pred"] > 1)
    n = min(args.handcheck_n, len(graded))
    if n:
        n_flag = min(int(graded["flagged"].sum()), n // 3)
        pick = graded[graded["flagged"]].sample(n=n_flag, random_state=args.seed)
        rest = graded.drop(pick.index)
        k = n - n_flag
        parts = [pick]
        if len(rest) and k:
            for _, s in rest.groupby(["question_type", "outcome"]):
                m = min(len(s), max(1, round(k * len(s) / len(rest))))
                parts.append(s.sample(n=m, random_state=args.seed))
        sample = pd.concat(parts)
        sample = sample[~sample.index.duplicated()]
        if len(sample) > n:
            sample = sample.sample(n=n, random_state=args.seed)
        elif len(sample) < n:
            more = rest.drop(sample.index, errors="ignore")
            sample = pd.concat([sample, more.sample(n=min(n - len(sample), len(more)),
                                                    random_state=args.seed)])
        sample = sample.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        cols = ["qid", "name", "description", "description_used", "question_type", "prompt", "gold",
                "response", "parsed", "outcome", "flagged"]
        hc = sample[cols].copy()
        hc["human_outcome"] = ""   # fill with: correct / wrong / refused
        hc["notes"] = ""
        for i, who in enumerate(HANDCHECKERS):
            hc.iloc[i::len(HANDCHECKERS)].to_csv(
                os.path.join(out, f"handcheck_{who}.csv"), index=False)
        print(f"Wrote {len(hc)} hand-check rows: "
              + ", ".join(f"handcheck_{w}.csv" for w in HANDCHECKERS))
    print(f"Results in {out}/")


if __name__ == "__main__":
    main()
