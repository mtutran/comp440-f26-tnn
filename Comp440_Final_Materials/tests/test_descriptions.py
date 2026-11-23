import pytest

from llm_eval.descriptions import DescriptionCleaner
from llm_eval.templates import build_question

C = DescriptionCleaner()


@pytest.mark.parametrize("raw,expected", [
    ("Polish-French physicist and chemist (1867–1934)", "physicist and chemist (1867–1934)"),
    ("Kenyan environmental and political activist", "environmental and political activist"),
    ("German-born theoretical physicist", "theoretical physicist"),
    ("American actor", "actor"),
    ("Nigerian-American writer", "writer"),
    ("Franco-British actor", "actor"),
    ("Soviet and Russian chess player", "chess player"),
    ("President of South Africa from 1994 to 1999", "President from 1994 to 1999"),
    ("US politician", "politician"),
    ("footballer", "footballer"),
    ("Roman Catholic bishop", "Roman Catholic bishop"),   # not a nationality
])
def test_citizenship_cleaning(raw, expected):
    assert C.for_citizenship(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Polish-French physicist and chemist (1867–1934)", "Polish-French physicist and chemist"),
    ("Classical Greek philosopher (384–322 BC)", "Classical Greek philosopher"),
    ("President of South Africa from 1994 to 1999", "President of South Africa"),
    ("19th-century British novelist", "British novelist"),
    ("Indian cricketer born 1988", "Indian cricketer"),
    ("Japanese politician in the 1960s", "Japanese politician"),
    ("singer", "singer"),
])
def test_birth_year_cleaning(raw, expected):
    assert C.for_birth_year(raw) == expected


def test_questions_no_longer_leak():
    d = "Polish-French physicist and chemist (1867–1934)"
    cit = build_question("citizenship", "Marie Curie", d, True)
    by = build_question("birth_year", "Marie Curie", d, True)
    assert "Polish" not in cit and "French" not in cit.split("?")[0]
    assert "1867" not in by
    assert "Marie Curie (physicist and chemist, 1867–1934)" in cit


def test_fully_removed_description_drops_brackets():
    q = build_question("citizenship", "Jane Doe", "American", None)
    assert "()" not in q and "Jane Doe a citizen" in q
