"""
templates.py - writes the questions we ask the LLM.
Author: Nadezhda Dominguez Salinas, AI Assistant (Claude Code)

COMP 440 Final Project 

We ask two questions about each person: their birth year and their country
of citizenship. Each question uses:
    - the person's full name
    - a short description from Wikidata, so the model knows which person
      we mean (there could be more than one "John Smith")
    - instructions to answer in a fixed format: "Answer: <value>"

The fixed format makes grading easier because the grader only has to read
the "Answer:" line. The model is told to say "Answer: unknown" if it doesn't
know, so we can count refusals separately from wrong answers.

Before the description goes into a question, descriptions.py removes
anything that would give away the answer (e.g. "Polish-French" for the
citizenship question, or "1867-1934" for the birth-year question).

Run this file directly to see example questions:
    python3 -m llm_eval.templates
"""

from __future__ import annotations

import re

# Change this whenever the wording changes. It's part of the cache key in
# query.py, so changing it makes the script ask the new questions instead
# of reusing old answers.
# v1: first version
# v2: descriptions are cleaned so they don't give away the answer
TEMPLATE_VERSION = "v2"

# Instructions sent with every question
SYSTEM_PROMPT = (
    "You answer short factual questions about people. "
    "Reply with exactly one line in the form 'Answer: <value>' and nothing else. "
    "If you do not know, reply 'Answer: unknown'. Do not guess wildly."
)

# The two question types. {name}, {description} and {is_was} get filled in
# by build_question(). The examples show the model the exact answer format.
TEMPLATES = {
    "birth_year": (
        "In what year was {name} ({description}) born?\n"
        "Give only the year as digits, e.g. 'Answer: 1952'. "
        "For years before the common era write e.g. 'Answer: 384 BC'. "
        "If you do not know, write 'Answer: unknown'."
    ),
    "citizenship": (
        "Of which country {is_was} {name} ({description}) a citizen?\n"
        "Give only the country name, e.g. 'Answer: Brazil'. "
        "If the person held more than one citizenship, list them separated by "
        "semicolons, e.g. 'Answer: France; Canada'. "
        "If you do not know, write 'Answer: unknown'."
    ),
}

QUESTION_TYPES = tuple(TEMPLATES)   # ("birth_year", "citizenship")


def _clean_description(desc):
    """
    Treats missing descriptions as None. pandas reads empty cells as NaN,
    so "nan" and "none" also count as missing.
    """
    if desc is None:
        return None
    desc = str(desc).strip()
    if not desc or desc.lower() in {"nan", "none"}:
        return None
    return desc


# The description cleaner is only created the first time it's needed,
# so importing this file stays fast.
_DEFAULT_CLEANER = None


def _default_cleaner():
    global _DEFAULT_CLEANER
    if _DEFAULT_CLEANER is None:
        from .descriptions import DescriptionCleaner
        _DEFAULT_CLEANER = DescriptionCleaner()
    return _DEFAULT_CLEANER


def build_question(question_type, name, description=None, is_deceased=None, cleaner=None):
    """
    Builds the question text for one person and one question type.

    Arguments:
        question_type: "birth_year" or "citizenship"
        name: the person's full name
        description: their Wikidata description (can be None)
        is_deceased: True -> "was", False -> "is", None -> "is or was"
            (only used in the citizenship question)
        cleaner: the DescriptionCleaner to use. None uses the default one,
            False skips cleaning (only for comparing with the raw version).

    Example:
        build_question("citizenship", "Marie Curie",
                       "Polish-French physicist and chemist (1867-1934)", True)
        -> "Of which country was Marie Curie (physicist and chemist,
            1867-1934) a citizen? ..."
    """
    if question_type not in TEMPLATES:
        raise ValueError(f"unknown question type {question_type!r}; "
                         f"choose from {QUESTION_TYPES}")

    # remove anything in the description that gives away the answer
    desc = _clean_description(description)
    if desc and cleaner is not False:
        desc = (cleaner or _default_cleaner()).clean(question_type, desc)

    # avoid brackets inside brackets: "physicist (1867-1934)" -> "physicist, 1867-1934"
    if desc:
        desc = re.sub(r"\s*\(([^()]*)\)", r", \1", desc).strip(" ,")

    # "was" for people who have died, "is" for living people
    if is_deceased is True:
        is_was = "was"
    elif is_deceased is False:
        is_was = "is"
    else:
        is_was = "is or was"

    text = TEMPLATES[question_type].format(
        name=name.strip(), description=desc or "", is_was=is_was
    )

    # if there's no description, remove the empty "()" after the name
    if desc is None:
        text = text.replace(f"{name.strip()} ()", name.strip())
    return text


if __name__ == "__main__":
    # print the example questions for Marie Curie
    print("SYSTEM:", SYSTEM_PROMPT, "\n")
    for qt in QUESTION_TYPES:
        print(f"[{qt}]")
        print(build_question(qt, "Marie Curie",
                             "Polish-French physicist and chemist (1867\u20131934)",
                             is_deceased=True))
        print()