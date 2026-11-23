#!/usr/bin/env python3
"""Compute the grader's error rate from the filled-in hand-check sheets (step 10).

Each person fills the `human_outcome` column of their handcheck_<name>.csv with
correct / wrong / refused (and optional notes). Then:

    python score_handcheck.py results/pilot_gpt-4o-mini

prints the overall error rate with a 95% Wilson interval, the error rate by
question type, a confusion table, and writes grader_disagreements.csv: the
rows to look at when fixing the grader (step 11).
"""

import glob
import math
import os
import sys

import pandas as pd

VALID = {"correct", "wrong", "refused"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def main(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "handcheck_*.csv")))
    if not files:
        sys.exit(f"no handcheck_*.csv files in {out_dir}")
    frames = []
    for f in files:
        d = pd.read_csv(f, dtype=str).fillna("")
        d["checker"] = os.path.basename(f)[len("handcheck_"):-4]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["human_outcome"] = df["human_outcome"].str.strip().str.lower()
    todo = df[~df["human_outcome"].isin(VALID)]
    if len(todo):
        print(f"WARNING: {len(todo)} rows not filled in (or not correct/wrong/refused):")
        print(todo.groupby("checker").size().to_string(), "\n")
    df = df[df["human_outcome"].isin(VALID)]
    if df.empty:
        sys.exit("no filled-in rows yet")

    df["disagree"] = df["human_outcome"] != df["outcome"]
    k, n = int(df["disagree"].sum()), len(df)
    lo, hi = wilson(k, n)
    print(f"Grader error rate: {k}/{n} = {k / n:.1%}  (95% CI {lo:.1%} to {hi:.1%})\n")
    print("By question type:")
    for qt, s in df.groupby("question_type"):
        print(f"  {qt:12s} {int(s['disagree'].sum())}/{len(s)} = {s['disagree'].mean():.1%}")
    if "flagged" in df:
        print("\nBy grader flag (flagged rows were over-sampled):")
        err = {}
        for fl, s in df.groupby("flagged"):
            err[fl] = s["disagree"].mean()
            print(f"  flagged={fl:5s} {int(s['disagree'].sum())}/{len(s)} = {err[fl]:.1%}")
        summ = os.path.join(out_dir, "summary.json")
        if os.path.exists(summ) and {"True", "False"} <= set(err):
            import json
            fr = json.load(open(summ)).get("flagged_rate")
            if fr is not None:
                w = fr * err["True"] + (1 - fr) * err["False"]
                print(f"  Reweighted to the full run ({fr:.1%} of answers flagged): "
                      f"estimated grader error = {w:.1%}  <- report this one")
    print("\nConfusion (rows = grader, columns = human):")
    print(pd.crosstab(df["outcome"], df["human_outcome"]).to_string())

    dis = df[df["disagree"]]
    path = os.path.join(out_dir, "grader_disagreements.csv")
    dis.to_csv(path, index=False)
    print(f"\n{len(dis)} disagreements written to {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python score_handcheck.py <results dir>")
    main(sys.argv[1])
