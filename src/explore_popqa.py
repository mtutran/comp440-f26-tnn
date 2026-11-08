"""Download PopQA (test split), save it to CSV, and do a quick exploration."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "popqa.csv"
FIG_PATH = ROOT / "figures" / "popularity.png"


def main():
    # Load and save
    df = load_dataset("akariasai/PopQA", split="test").to_pandas()
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    print(f"Saved to {RAW_PATH.relative_to(ROOT)}\n")

    # Shape and columns
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")

    # Questions by relation type
    print("Questions by relation (prop):")
    print(df["prop"].value_counts().to_string(), "\n")

    # Popularity histogram on a log scale
    pop = df["s_pop"][df["s_pop"] > 0]
    bins = np.logspace(np.log10(pop.min()), np.log10(pop.max()), 50)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pop, bins=bins, color="#4C72B0", edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Subject popularity (s_pop, monthly Wikipedia page views)")
    ax.set_ylabel("Number of questions")
    ax.set_title("PopQA: distribution of subject popularity")
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"Saved histogram to {FIG_PATH.relative_to(ROOT)}")
    if (df["s_pop"] <= 0).any():
        print(f"(excluded {(df['s_pop'] <= 0).sum()} rows with s_pop <= 0 from log plot)")
    print()

    # Example rows
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 200)
    for rel in ["occupation", "place of birth"]:
        print(f"Examples for relation '{rel}':")
        print(df.loc[df["prop"] == rel, ["subj", "question", "possible_answers"]]
              .head(5).to_string(index=False), "\n")


if __name__ == "__main__":
    main()
