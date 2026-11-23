"""
descriptions.py - removes answer giveaways from Wikidata descriptions.
Author: Nadezhda Dominguez Salinas, AI Assistant (Claude Code)

COMP 440 Final Project used by templates.py.

Why we need this:
    We add each person's Wikidata description to the question so the model
    knows who we mean. But descriptions often contain the answer, e.g.
    "Polish-French physicist and chemist (1867-1934)" has both Marie Curie's
    citizenship and her birth year. If we left that in, the model could just
    copy the answer, and accuracy would look too high. It would also be
    uneven, because well-known people tend to have more detailed
    descriptions, which messes up the coverage comparison we're testing.

What it does:
    Each question type gets its own cleaned description:
        citizenship -> remove country names and nationality words
            "Polish-French physicist and chemist" -> "physicist and chemist"
        birth_year  -> remove years, date ranges, centuries and decades
            "physicist (1867-1934)" -> "physicist"
    Everything else is kept so the person can still be identified.

Limitations (for our write-up):
    - Cities and institutions can still hint at a country ("mayor of Chicago").
    - Job titles can still hint at an era.
    - Rare nationality words are only caught if they're in the country list,
      so for the real run we should load the full Wikidata list with
      scripts/fetch_country_aliases.py.
"""

from __future__ import annotations

import re

from .countries import BUILTIN_ALIASES, BUILTIN_HISTORICAL, CountryResolver

# Wikidata uses several kinds of dashes (-, en dash, em dash...), so this
# matches a normal hyphen plus the Unicode dash characters U+2010 to U+2015.
DASH = r"[-\u2010-\u2015]"

# Nationality words that aren't in the country list in countries.py:
# adjectives for historical states, and prefixes like "Franco-" in
# "Franco-British" or "Sino-" in "Sino-Japanese".
EXTRA_TERMS = [
    "Soviet", "Prussian", "Ottoman", "Yugoslav", "Yugoslavian", "Czechoslovak",
    "Czechoslovakian", "Austro-Hungarian", "Austria-Hungary", "Byzantine",
    "Rhodesian", "Siamese", "Ceylonese", "Bohemian", "Bavarian", "Venetian",
    "Franco", "Anglo", "Sino", "Russo", "Austro", "Italo", "Hispano", "Luso",
    "Greco", "Indo",
]

# Endings that go with a nationality word and should be removed with it,
# e.g. "German-born" or "London-based"
_NAT_SUFFIX = rf"(?:{DASH}born|{DASH}based|{DASH}descended)?"


class DescriptionCleaner:
    """
    Cleans descriptions for each question type.

    When it's created, it builds one big regex out of every country name,
    alias and nationality word we know about, so removing them is a single
    regex substitution.

    Arguments:
        resolver: optional CountryResolver from countries.py. If it was loaded
            with the full Wikidata alias list, those names get removed too.
    """

    def __init__(self, resolver=None):
        # collect every country word: names, aliases, demonyms, historical states
        terms = set()
        for canon, aliases in BUILTIN_ALIASES.items():
            terms.add(canon)
            terms.update(aliases)
        terms.update(BUILTIN_HISTORICAL)
        terms.update(EXTRA_TERMS)
        if resolver is not None:
            terms.update(resolver.all_names())

        # Short all-caps forms (US, UK, PRC) are matched case-sensitively so
        # the normal word "us" doesn't get removed. Everything else ignores case.
        acronyms = sorted({t for t in terms if t.isupper() and len(t) <= 4},
                          key=len, reverse=True)
        words = sorted({t for t in terms if t not in acronyms and len(t) >= 3},
                       key=len, reverse=True)

        # Longest names go first in the regex so "South Africa" is matched
        # before "Africa" could be. The lookarounds make sure we only match
        # whole words ("Indian" shouldn't match inside "Indiana").
        alt = "|".join(re.escape(w) for w in words)
        self._country_re = re.compile(
            rf"(?<![\w])(?:{alt}){_NAT_SUFFIX}(?![\w])", re.IGNORECASE)
        self._acronym_re = re.compile(
            rf"(?<![\w.])(?:{'|'.join(re.escape(a) for a in acronyms)})(?![\w])")

    def for_citizenship(self, desc):
        """
        Removes country names and nationality words.
        "Polish-French physicist and chemist" -> "physicist and chemist"
        """
        s = self._country_re.sub(" ", desc)
        s = self._acronym_re.sub(" ", s)
        return _tidy(s)

    def for_birth_year(self, desc):
        """
        Removes years and dates.
        "Polish-French physicist and chemist (1867-1934)"
            -> "Polish-French physicist and chemist"
        """
        s = desc
        # y matches a year with an optional era, e.g. "1952", "384 BC"
        y = r"\d{1,4}(?:\s*(?:BCE|BC|CE|AD))?"

        # brackets with a year inside: "(1867-1934)", "(born 1952)"
        s = re.sub(r"\([^()]*\d{3,4}[^()]*\)", " ", s)
        # brackets with a BC year: "(c. 384 BC)"
        s = re.sub(r"\(\s*(?:born|b\.|c\.|circa|fl\.)?[^()]*\b\d{1,4}\s*(?:BCE|BC)\b[^()]*\)",
                   " ", s, flags=re.IGNORECASE)

        # centuries: "19th-century", "19th century"
        s = re.sub(rf"\b\d{{1,2}}(?:st|nd|rd|th){DASH}?\s*century\b", " ", s,
                   flags=re.IGNORECASE)
        # decades: "1960s"
        s = re.sub(r"\b\d{3,4}s\b", " ", s)

        # ranges: "from 1994 to 1999", "between 1990 and 2000", "1994-1999"
        s = re.sub(rf"\bfrom\s+{y}\s+(?:to|until|till)\s+{y}\b", " ", s, flags=re.IGNORECASE)
        s = re.sub(rf"\bbetween\s+{y}\s+and\s+{y}\b", " ", s, flags=re.IGNORECASE)
        s = re.sub(rf"\b{y}\s*{DASH}\s*{y}\b", " ", s, flags=re.IGNORECASE)

        # a word followed by a year: "born 1952", "since 2001", "in 1988"
        s = re.sub(rf"\b(?:born|b\.|since|in|from|until|after|before|circa|c\.)\s+{y}\b",
                   " ", s, flags=re.IGNORECASE)

        # anything left over: a BC year, or any 3-4 digit number on its own
        s = re.sub(r"\b\d{1,4}\s*(?:BCE|BC)\b", " ", s, flags=re.IGNORECASE)
        s = re.sub(r"(?<![\w.])\d{3,4}(?![\w.])", " ", s)
        return _tidy(s)

    def clean(self, question_type, desc):
        """
        Picks the right cleaning for the question type. This is the function
        templates.py calls. Returns None if nothing is left after cleaning
        (e.g. a description that was just "American").
        """
        if not desc:
            return desc
        if question_type == "citizenship":
            out = self.for_citizenship(desc)
        elif question_type == "birth_year":
            out = self.for_birth_year(desc)
        else:
            out = desc
        return out or None


def _tidy(s):
    """
    Fixes the grammar left behind after removing words, e.g.
    "President of South Africa from 1994" -> "President of  from 1994"
    -> "President from 1994".

    Runs 3 times because fixing one thing can leave something else to fix
    (removing a comma can leave a dangling "and").
    """
    for _ in range(3):
        # leftover dashes: "Polish-French" -> " - " -> " "
        s = re.sub(rf"\s*{DASH}\s*(?={DASH}|\s|$|[,;)])", " ", s)
        s = re.sub(rf"(?:(?<=\s)|^){DASH}\s*", " ", s)
        # empty brackets: "( )" or "(, )"
        s = re.sub(r"\(\s*[,;\-–]*\s*\)", " ", s)
        # spaces before punctuation: "physicist ," -> "physicist,"
        s = re.sub(r"\s+([,;)])", r"\1", s)
        # commas with nothing after them
        s = re.sub(r"([,;])\s*(?=[,;)]|$)", "", s)
        s = re.sub(r"\(\s+", "(", s)
        # dangling small words: "President of from" -> "President from"
        s = re.sub(r"\b(?:of|in|from|for|and|or|to|the)\s+(?=(?:from|to|in|and|of|since|"
                   r"between|until)\b|[,;)]|$)", "", s, flags=re.IGNORECASE)
        # small words left at the start or end: "and chess player" -> "chess player"
        s = re.sub(r"^\s*(?:and|or|of|in|from|the|born|,|;)\s+", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+(?:and|or|of|in|from|the|to)\s*$", "", s, flags=re.IGNORECASE)
        # extra spaces
        s = re.sub(r"\s{2,}", " ", s).strip(" ,;")
    return s