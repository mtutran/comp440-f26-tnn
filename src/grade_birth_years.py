"""Toy birth-year QA: grade free-text answers against Wikidata birth years.

The answers below are HAND-WRITTEN test cases (no LLM API key was available).
They are not real model outputs: they deliberately mix correct answers,
wrong answers, refusals and tricky phrasings to exercise the grader.
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "data" / "processed" / "people_sample.csv"

YEAR_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")  # exactly 4 digits, not part of a longer number

# qid -> hand-written answer
HAND_ANSWERS = {
    "Q2755894": "Darrell Hammond was born in 1955.",
    "Q5362161": "I'm not sure; I don't have reliable information about Eliza Lawrence.",
    "Q360823":  "Bruce Heezen was born on April 11, 1924, in Vinton, Iowa.",
    "Q1351218": "Eric Defoort (1943–2016) was a Belgian politician.",
    "Q147668":  "She was born in 1968.",
    "Q7328681": "Richard Rishi was born in 1981.",
    "Q7934296": "A 2005 profile notes that Virginia Haviland was born in 1911.",
    "Q4727053": "Alison Cockburn, the Scottish poet, was born in 1713.",
    "Q2732074": "1875",
    "Q3087132": "Frederick Mackenzie was born around 1840.",
}


def extract_year(text):
    """Return the first standalone 4-digit year in text as an int, or None."""
    m = YEAR_RE.search(text or "")
    return int(m.group(1)) if m else None


def wikidata_year(birth_date):
    """Year from a Wikidata xsd:dateTime like '1955-10-08T00:00:00Z' (or '-0500-...')."""
    m = re.match(r"^(-?\d+)-", str(birth_date))
    return int(m.group(1)) if m else None


def grade(answer, expected_year):
    """True iff the first 4-digit year in answer equals expected_year."""
    predicted = extract_year(answer)
    return predicted is not None and predicted == expected_year


def main():
    people = pd.read_csv(SAMPLE_PATH).set_index("qid")
    rows = []
    for qid, answer in HAND_ANSWERS.items():
        name = people.at[qid, "subj"]
        expected = wikidata_year(people.at[qid, "birth_date"])
        rows.append({
            "question": f"In what year was {name} born?",
            "answer": answer,
            "extracted": extract_year(answer),
            "expected": expected,
            "correct": grade(answer, expected),
        })

    for r in rows:
        mark = "✓" if r["correct"] else "✗"
        print(f"{mark} {r['question']}\n"
              f"    answer:    {r['answer']}\n"
              f"    extracted: {r['extracted']}   expected: {r['expected']}\n")
    n_correct = sum(r["correct"] for r in rows)
    print(f"Accuracy: {n_correct}/{len(rows)}")


if __name__ == "__main__":
    main()
