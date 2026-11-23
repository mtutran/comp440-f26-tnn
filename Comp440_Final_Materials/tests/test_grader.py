import pytest

from llm_eval.grader import Grader, CORRECT, WRONG, REFUSED, parse_years
from llm_eval.templates import build_question

G = Grader()


@pytest.mark.parametrize("resp,gold,expected", [
    ("Answer: 1952", "1952", CORRECT),
    ("Answer: 1953", "1952", WRONG),
    ("Answer: c. 1952", "1952-03-11T00:00:00Z", CORRECT),
    ("Answer: 1952 AD", 1952, CORRECT),
    ("Answer: 384 BC", "-384", CORRECT),
    ("Answer: 384 BCE", "-0384-01-01T00:00:00Z", CORRECT),
    ("Answer: unknown", "1952", REFUSED),
    ("I don't know.", "1952", REFUSED),
    ("", "1952", REFUSED),
    ("Answer: 1950s", "1952", WRONG),
    ("Answer: 19th century", "1852", WRONG),
    ("He was born in 1952.", "1952", CORRECT),           # no Answer: line
    ("Answer: 1952, but I'm not sure", "1952", CORRECT), # hedge with a value
    ("Answer: 1951", "1951|1952", CORRECT),              # multiple valid gold years
    ("Answer: **1952**", "1952", CORRECT),               # markdown
])
def test_birth_year(resp, gold, expected):
    assert G.grade("birth_year", resp, gold).outcome == expected


def test_birth_year_tolerance():
    assert Grader(year_tolerance=1).grade("birth_year", "Answer: 1953", "1952").outcome == CORRECT
    assert Grader(year_tolerance=1).grade("birth_year", "Answer: 1954", "1952").outcome == WRONG


def test_birth_year_hedging_strict_vs_lenient():
    g = G.grade("birth_year", "Answer: 1951 or 1952", "1952")
    assert g.outcome == CORRECT and g.outcome_strict == WRONG and g.n_pred == 2


def test_decade_flagged_imprecise():
    years, imprecise = parse_years("1950s")
    assert years == [] and imprecise


@pytest.mark.parametrize("resp,gold,expected", [
    ("Answer: United States", "United States of America", CORRECT),
    ("Answer: USA", "United States of America", CORRECT),
    ("Answer: U.S.", "United States of America", CORRECT),
    ("Answer: American", "United States of America", CORRECT),
    ("Answer: the United Kingdom", "United Kingdom", CORRECT),
    ("Answer: Britain", "United Kingdom", CORRECT),
    ("Answer: England", "United Kingdom", CORRECT),
    ("Answer: France", "United Kingdom", WRONG),
    ("Answer: Côte d'Ivoire", "Ivory Coast", CORRECT),
    ("Answer: Cote d'Ivoire", "Ivory Coast", CORRECT),
    ("Answer: Bosnia and Herzegovina", "Bosnia and Herzegovina", CORRECT),
    ("Answer: Italy", "Kingdom of Italy", CORRECT),           # historical -> modern
    ("Answer: Soviet Union", "Russia", CORRECT),             # historical -> modern
    ("Answer: unknown", "France", REFUSED),
    ("I'm not sure who this is.", "France", REFUSED),
    ("Answer: not sure, possibly France", "France", CORRECT),
    ("Answer: France; Canada", "Canada", CORRECT),           # one of several
    ("Answer: Kenya", "Kenya|United Kingdom", CORRECT),      # multiple valid gold
    ("Answer: Czechia", "Czech Republic", CORRECT),
    ("Answer: Türkiye", "Turkey", CORRECT),
])
def test_citizenship(resp, gold, expected):
    assert G.grade("citizenship", resp, gold).outcome == expected


def test_citizenship_strict():
    g = G.grade("citizenship", "Answer: France; Canada", "Canada")
    assert g.outcome == CORRECT and g.outcome_strict == WRONG


def test_unrecognized_country_falls_back_to_string_match():
    g = G.grade("citizenship", "Answer: Freedonia", "Freedonia")
    assert g.outcome == CORRECT and g.unrecognized


def test_historical_flag():
    g = G.grade("citizenship", "Answer: Germany", "West Germany")
    assert g.outcome == CORRECT and g.historical_match


def test_template_includes_name_and_description():
    q = build_question("birth_year", "Ada Lovelace", "English mathematician")
    assert "Ada Lovelace (English mathematician)" in q and "Answer:" in q
    q2 = build_question("citizenship", "Ada Lovelace", None)
    assert "()" not in q2 and "is or was" in q2
