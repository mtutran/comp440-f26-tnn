"""
grader.py - automatically grades the LLM's answers.
Author: Nadezhda Dominguez Salinas, AI Assistant (Claude Code)

COMP 440 Final Project, step 7: automatic grader.

Every answer gets one of three outcomes:
    correct  - the answer matches Wikidata
    wrong    - the model gave an answer but it doesn't match
    refused  - the model said it didn't know ("Answer: unknown")

How it works:
    1. Find the "Answer: ..." line in the model's reply.
    2. Check if it's a refusal.
    3. Turn the answer into something we can compare (a year as a number,
       or a country as one standard name).
    4. Compare it with the correct answer(s) from Wikidata.

Things it handles:
    - Date formats: "1952", "c. 1952", "1952 AD", "384 BC", "born in 1952"
    - Country aliases: "USA", "U.S." and "American" all mean United States
      (the list is in countries.py)
    - Historical countries: "Italy" matches "Kingdom of Italy"
    - Multiple valid answers: some people have two citizenships or a
      disputed birth year, so matching any of them counts

Two scores:
    When the model gives more than one value ("France; Canada"), we record
        outcome         - correct if ANY of its values is right (lenient)
        outcome_strict  - correct only if ALL of them are right
    so Nayla can check if this choice changes the results.

Flags for the hand-check (cases where the grader is most likely wrong):
    format_ok         False if there was no "Answer:" line
    n_pred            how many values the model gave
    unrecognized      a country name we don't have in our list
    historical_match  only matched after mapping an old country to a new one
    imprecise         a vague answer like "1950s" (graded wrong)

Note: BC years are stored as negative numbers (384 BC -> -384). Wikidata can
be off by one year for BC dates because there's no year 0, so BC cases should
be hand-checked.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field

from .countries import CountryResolver, normalize

# the three outcomes
CORRECT, WRONG, REFUSED = "correct", "wrong", "refused"

# Phrases that mean the model didn't know. \b means "word boundary", so
# "unknown" matches but not inside another word.
REFUSAL_PATTERNS = [
    r"\bunknown\b", r"\bi don'?t know\b", r"\bi do not know\b", r"\bnot sure\b",
    r"\bunsure\b", r"\bcannot (?:determine|find|confirm|answer|provide)\b",
    r"\bcan'?t (?:determine|find|confirm|answer|provide)\b", r"\bno (?:reliable )?information\b",
    r"\bnot (?:enough|sufficient) information\b", r"\bunable to\b", r"\bunclear\b",
    r"\bi'?m not (?:aware|familiar)\b", r"\bno record\b", r"^n/?a$", r"^none$", r"^\?+$",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)

# finds "Answer: 1952" (also accepts "answer - 1952", any capitalization)
_ANSWER_RE = re.compile(r"answer\s*[:\-]\s*(.+)", re.IGNORECASE)


@dataclass
class Grade:
    """
    The result of grading one answer. Each field becomes a column in
    responses.csv.

    Fields:
        outcome: "correct", "wrong" or "refused" (lenient score)
        outcome_strict: same, but every value the model gave must be right
        parsed: the answer text we graded, e.g. "1867"
        predictions: the values we read from it, e.g. [1867] or ["France", "Canada"]
        format_ok, n_pred, unrecognized, historical_match, imprecise:
            the hand-check flags described at the top of the file
    """
    outcome: str
    outcome_strict: str
    parsed: str
    predictions: list = field(default_factory=list)
    format_ok: bool = True
    n_pred: int = 0
    unrecognized: bool = False
    historical_match: bool = False
    imprecise: bool = False

    def as_row(self):
        """Turns the grade into a dict for one CSV row (the list becomes a JSON string)."""
        d = asdict(self)
        d["predictions"] = json.dumps(d["predictions"], ensure_ascii=False)
        return d


# ---------------------------------------------------------------------------
# Helpers used by both question types
# ---------------------------------------------------------------------------

def extract_answer(text):
    """
    Gets the answer from the model's reply.

    Uses the last "Answer:" line, and strips markdown like **bold** and
    trailing periods. If there's no "Answer:" line, it uses the whole reply
    and returns format_ok = False so we can flag it.

    Example: "Answer: **1867**." -> ("1867", True)
    """
    text = (text or "").strip()
    matches = _ANSWER_RE.findall(text)
    if matches:
        ans = matches[-1].strip()
        return ans.splitlines()[0].strip().strip("*_`\"'").strip(" ."), True
    return text.strip(), False


def is_refusal(ans):
    """True if the answer is empty or contains a refusal phrase like "unknown"."""
    return not ans.strip() or bool(_REFUSAL_RE.search(ans))


def parse_gold_list(value):
    """
    Reads the correct answer(s) from Tu's sample file into a list.

    Accepts a Python list, a JSON list like '["Poland", "France"]', or a
    string separated by | like "Poland|France". Empty cells give [].
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s or s.lower() == "nan":   # pandas reads empty cells as "nan"
        return []
    if s.startswith("["):
        try:
            return [str(v).strip() for v in json.loads(s) if str(v).strip()]
        except json.JSONDecodeError:
            pass
    return [p.strip() for p in s.split("|") if p.strip()]


# ---------------------------------------------------------------------------
# Birth year
# ---------------------------------------------------------------------------

# Matches a number with an optional "s" (for decades) and an optional era.
# The lookarounds stop it from matching part of a longer number or a date
# like "1952-03-11".
_YEAR_RE = re.compile(
    r"(?<![\d-])(\d{1,4})(s)?\s*(bce|bc|b\.c\.e?\.?|ce|ad|a\.d\.)?(?!\d)", re.IGNORECASE
)
_CENTURY_RE = re.compile(r"\b\d{1,2}(st|nd|rd|th)\s+century\b", re.IGNORECASE)


def parse_years(ans):
    """
    Finds the year(s) in an answer. Returns (years, imprecise).

    BC years come back negative. Decades ("1950s") and centuries
    ("19th century") are too vague to grade, so they aren't returned as
    years and imprecise is set to True.

    Examples:
        "1952"        -> ([1952], False)
        "384 BC"      -> ([-384], False)
        "1951 or 1952" -> ([1951, 1952], False)
        "1950s"       -> ([], True)
    """
    if _CENTURY_RE.search(ans):
        return [], True
    years, imprecise = [], False
    for num, decade, era in _YEAR_RE.findall(ans):
        y = int(num)
        if decade:
            imprecise = True
            continue
        if era and era.lower().startswith("b"):
            y = -y
        elif not era and y < 100:
            continue   # a bare 1-2 digit number probably isn't a year
        years.append(y)
    return list(dict.fromkeys(years)), imprecise   # remove duplicates, keep order


def parse_gold_year(v):
    """
    Reads a year from Wikidata's format.
    1952, "1952", "1952-03-11T00:00:00Z" -> 1952
    "-0384-01-01" -> -384 (BC)
    Returns None if there's no year.
    """
    s = str(v).strip()
    m = re.match(r"^([+-]?)0*(\d{1,4})", s)
    if not m:
        return None
    y = int(m.group(2))
    return -y if m.group(1) == "-" else y


def grade_birth_year(response_text, gold, year_tolerance=0):
    """
    Grades a birth-year answer.

    Arguments:
        response_text: the model's full reply
        gold: the correct year(s) from Wikidata
        year_tolerance: how many years off still counts as correct
            (0 normally, 1 for the +/-1 year robustness check)
    """
    ans, fmt = extract_answer(response_text)
    preds, imprecise = parse_years(ans)

    # Only count it as a refusal if there's no year at all, so
    # "1952, but I'm not sure" is still graded on the 1952.
    if not preds and not imprecise and is_refusal(ans):
        return Grade(REFUSED, REFUSED, ans, format_ok=fmt)

    gold_years = [y for y in (parse_gold_year(g) for g in parse_gold_list(gold))
                  if y is not None]
    if not preds:
        return Grade(WRONG, WRONG, ans, format_ok=fmt, imprecise=imprecise)

    def ok(p):
        # a year is right if it's close enough to ANY of the valid years
        return any(abs(p - g) <= year_tolerance for g in gold_years)

    hits = [ok(p) for p in preds]
    return Grade(
        outcome=CORRECT if any(hits) else WRONG,
        outcome_strict=CORRECT if all(hits) else WRONG,
        parsed=ans, predictions=preds, format_ok=fmt, n_pred=len(preds),
        imprecise=imprecise,
    )


# ---------------------------------------------------------------------------
# Citizenship
# ---------------------------------------------------------------------------

def split_countries(ans):
    """
    Splits an answer that names more than one country.

    Splits on semicolons (the format we asked for). If there are none, it
    splits on commas, slashes, "and", "or" and "&". It also removes extra
    words like "citizen of", "probably" and anything in brackets.

    Examples:
        "France; Canada"            -> ["France", "Canada"]
        "probably a citizen of Peru" -> ["Peru"]
    """
    ans = re.sub(r"\(.*?\)", "", ans)   # drop things like "(formerly ...)"
    if ";" in ans:
        parts = ans.split(";")
    else:
        parts = re.split(r",|/|\band\b|\bor\b|&", ans)
    out = []
    for p in parts:
        p = re.sub(r"^\s*(?:a |an )?(?:citizen of|citizenship:?|national of)\s+", "",
                   p.strip(), flags=re.IGNORECASE)
        p = re.sub(r"^\s*(?:possibly|probably|likely|maybe|perhaps|most likely)\s+", "",
                   p, flags=re.IGNORECASE)
        p = re.sub(r"\s+(?:citizen|national|citizenship)$", "", p, flags=re.IGNORECASE)
        p = p.strip(" .")
        if p:
            out.append(p)
    return out


def grade_citizenship(response_text, gold, resolver):
    """
    Grades a citizenship answer.

    Arguments:
        response_text: the model's full reply
        gold: the correct country/countries from Wikidata
        resolver: the CountryResolver from countries.py, which knows the
            aliases and historical countries
    """
    ans, fmt = extract_answer(response_text)
    if not ans.strip():
        return Grade(REFUSED, REFUSED, ans, format_ok=fmt)

    # Check the whole answer first so country names with "and" in them
    # ("Bosnia and Herzegovina") don't get split into two.
    preds = [ans] if resolver.canonical(ans) else split_countries(ans)

    # If there's a refusal phrase, only keep values that are real countries:
    # "not sure, possibly France" -> France. If none are left, it's a refusal.
    if is_refusal(ans):
        preds = [p for p in preds if resolver.canonical(p)]
        if not preds:
            return Grade(REFUSED, REFUSED, ans, format_ok=fmt)

    # Turn the correct answers into standard names, modern names, and plain
    # lowercase text, so we can compare in all three ways.
    gold_list = parse_gold_list(gold)
    gold_canon = {resolver.canonical(g) or g.strip() for g in gold_list}
    gold_modern = {resolver.modern(g) or g.strip() for g in gold_list}
    gold_norm = {normalize(g) for g in gold_list} | {normalize(g) for g in gold_canon}

    hits, unrecognized, historical = [], False, False
    for p in preds:
        c = resolver.canonical(p)
        if c is None:
            # not in our country list: fall back to comparing the text
            unrecognized = True
            hits.append(normalize(p) in gold_norm)
            continue
        if c in gold_canon:
            hits.append(True)                       # same country
        elif (resolver.modern(p) or c) in gold_modern:
            hits.append(True)                       # match after historical -> modern
            historical = True
        else:
            hits.append(False)                      # different country

    if not preds:
        return Grade(WRONG, WRONG, ans, format_ok=fmt)
    return Grade(
        outcome=CORRECT if any(hits) else WRONG,
        outcome_strict=CORRECT if all(hits) else WRONG,
        parsed=ans, predictions=preds, format_ok=fmt, n_pred=len(preds),
        unrecognized=unrecognized, historical_match=historical,
    )


# ---------------------------------------------------------------------------
# Main grader
# ---------------------------------------------------------------------------

class Grader:
    """
    The grader run_eval.py uses. It loads the country list once and then
    sends each answer to the right grading function.

    Arguments:
        aliases_csv: optional extra country names from Wikidata
            (made by scripts/fetch_country_aliases.py)
        historical_csv: optional historical -> modern country table from Tu
        year_tolerance: years off that still count as correct (default 0)
    """

    def __init__(self, aliases_csv=None, historical_csv=None, year_tolerance=0):
        self.resolver = CountryResolver(aliases_csv, historical_csv)
        self.year_tolerance = year_tolerance

    def grade(self, question_type, response_text, gold):
        """Grades one answer. question_type is "birth_year" or "citizenship"."""
        if question_type == "birth_year":
            return grade_birth_year(response_text, gold, self.year_tolerance)
        if question_type == "citizenship":
            return grade_citizenship(response_text, gold, self.resolver)
        raise ValueError(f"unknown question type {question_type!r}")