"""Country names, aliases, demonyms and historical-state mappings for grading.

The built-in table covers common names and the forms models actually produce
("USA", "the U.S.", "American", "Britain", "UK"...). It is a starting point:

* `scripts/fetch_country_aliases.py` pulls every country's label, aliases and
  demonyms from Wikidata into `data/country_aliases.csv`; pass that file to the
  grader to extend this table.
* Tu's historical-state lookup (historical -> modern country) should be passed
  as `historical_csv`; its rows override the small built-in list below. A state
  that split into several countries lists them all, joined with "|"
  ("Soviet Union" -> "Russia|Ukraine|Lithuania|...").
"""

from __future__ import annotations

import csv
import re
import unicodedata


def normalize(s: str) -> str:
    """Lowercase, strip accents and punctuation, drop a leading 'the'."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold().replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)       # punctuation -> space (U.S.A. -> u s a)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^the ", "", s)
    # collapse dotted initialisms: "u s a" -> "usa", "u k" -> "uk"
    if re.fullmatch(r"(\w )+\w", s) and all(len(t) == 1 for t in s.split()):
        s = s.replace(" ", "")
    return s


# canonical name -> aliases and demonyms (all compared after normalize()).
BUILTIN_ALIASES: dict[str, list[str]] = {
    "United States": ["United States of America", "USA", "US", "America",
                      "American", "U.S.", "U.S.A.", "United States citizen"],
    "United Kingdom": ["UK", "Britain", "Great Britain", "British",
                       "United Kingdom of Great Britain and Northern Ireland",
                       "England", "English", "Scotland", "Scottish", "Wales",
                       "Welsh", "Northern Ireland"],
    "Russia": ["Russian Federation", "Russian"],
    "China": ["People's Republic of China", "PRC", "Chinese", "Mainland China"],
    "Taiwan": ["Republic of China", "ROC", "Taiwanese"],
    "South Korea": ["Republic of Korea", "Korea", "Korean", "South Korean"],
    "North Korea": ["Democratic People's Republic of Korea", "DPRK", "North Korean"],
    "Germany": ["Federal Republic of Germany", "German"],
    "France": ["French", "French Republic"],
    "Italy": ["Italian", "Italian Republic"],
    "Spain": ["Spanish", "Kingdom of Spain"],
    "Portugal": ["Portuguese"],
    "Netherlands": ["Holland", "Dutch", "Kingdom of the Netherlands"],
    "Belgium": ["Belgian"],
    "Switzerland": ["Swiss"],
    "Austria": ["Austrian"],
    "Sweden": ["Swedish"],
    "Norway": ["Norwegian"],
    "Denmark": ["Danish"],
    "Finland": ["Finnish"],
    "Iceland": ["Icelandic"],
    "Ireland": ["Republic of Ireland", "Irish", "Eire"],
    "Poland": ["Polish"],
    "Czech Republic": ["Czechia", "Czech"],
    "Slovakia": ["Slovak"],
    "Hungary": ["Hungarian"],
    "Romania": ["Romanian"],
    "Bulgaria": ["Bulgarian"],
    "Greece": ["Greek", "Hellenic Republic"],
    "Turkey": ["Turkiye", "Türkiye", "Turkish", "Republic of Turkey"],
    "Ukraine": ["Ukrainian"],
    "Belarus": ["Belarusian"],
    "Serbia": ["Serbian"],
    "Croatia": ["Croatian"],
    "Slovenia": ["Slovenian", "Slovene"],
    "Bosnia and Herzegovina": ["Bosnia", "Bosnian"],
    "North Macedonia": ["Macedonia", "Macedonian"],
    "Canada": ["Canadian"],
    "Mexico": ["Mexican", "United Mexican States"],
    "Brazil": ["Brazilian"],
    "Argentina": ["Argentine", "Argentinian"],
    "Chile": ["Chilean"],
    "Colombia": ["Colombian"],
    "Peru": ["Peruvian"],
    "Venezuela": ["Venezuelan"],
    "Cuba": ["Cuban"],
    "Jamaica": ["Jamaican"],
    "Australia": ["Australian"],
    "New Zealand": ["New Zealander", "Kiwi"],
    "Japan": ["Japanese"],
    "India": ["Indian", "Republic of India"],
    "Pakistan": ["Pakistani"],
    "Bangladesh": ["Bangladeshi"],
    "Sri Lanka": ["Sri Lankan", "Ceylon"],
    "Nepal": ["Nepali", "Nepalese"],
    "Indonesia": ["Indonesian"],
    "Philippines": ["Filipino", "Philippine"],
    "Vietnam": ["Viet Nam", "Vietnamese"],
    "Thailand": ["Thai", "Siam"],
    "Malaysia": ["Malaysian"],
    "Singapore": ["Singaporean"],
    "Myanmar": ["Burma", "Burmese"],
    "Iran": ["Islamic Republic of Iran", "Persia", "Iranian"],
    "Iraq": ["Iraqi"],
    "Israel": ["Israeli"],
    "Palestine": ["State of Palestine", "Palestinian"],
    "Saudi Arabia": ["Saudi"],
    "Egypt": ["Egyptian"],
    "Morocco": ["Moroccan"],
    "Algeria": ["Algerian"],
    "Tunisia": ["Tunisian"],
    "Nigeria": ["Nigerian"],
    "Ghana": ["Ghanaian"],
    "Kenya": ["Kenyan"],
    "Ethiopia": ["Ethiopian"],
    "South Africa": ["South African"],
    "Democratic Republic of the Congo": ["DR Congo", "DRC", "Congo-Kinshasa", "Zaire"],
    "Republic of the Congo": ["Congo-Brazzaville"],
    "Ivory Coast": ["Côte d'Ivoire", "Cote d'Ivoire", "Ivorian"],
    "Senegal": ["Senegalese"],
    "Uganda": ["Ugandan"],
    "Tanzania": ["Tanzanian"],
    "Zimbabwe": ["Zimbabwean", "Rhodesia"],
    "Cape Verde": ["Cabo Verde", "Cape Verdean", "Cabo Verdean"],
}

# historical state -> modern country. Only unambiguous cases are built in;
# Tu's lookup table should be loaded on top of this.
BUILTIN_HISTORICAL: dict[str, str] = {
    "Soviet Union": "Russia", "USSR": "Russia", "Russian Empire": "Russia",
    "Russian Soviet Federative Socialist Republic": "Russia",
    "West Germany": "Germany", "East Germany": "Germany",
    "German Democratic Republic": "Germany", "German Empire": "Germany",
    "Weimar Republic": "Germany", "Nazi Germany": "Germany",
    "Kingdom of Prussia": "Germany",
    "Kingdom of Italy": "Italy",
    "Ottoman Empire": "Turkey",
    "Kingdom of Great Britain": "United Kingdom",
    "United Kingdom of Great Britain and Ireland": "United Kingdom",
    "Kingdom of England": "United Kingdom", "Kingdom of Scotland": "United Kingdom",
    "British Raj": "India", "Dominion of India": "India",
    "Empire of Japan": "Japan", "Qing dynasty": "China",
    "Republic of China (1912-1949)": "China",
    "Kingdom of France": "France", "French Third Republic": "France",
    "Spanish Empire": "Spain", "Kingdom of Portugal": "Portugal",
    "Empire of Brazil": "Brazil", "Union of South Africa": "South Africa",
    "Kingdom of Hungary": "Hungary", "Second Polish Republic": "Poland",
    "Polish People's Republic": "Poland", "Kingdom of Romania": "Romania",
    "Kingdom of Greece": "Greece", "Kingdom of Egypt": "Egypt",
    "Mandatory Palestine": "Israel",
}


class CountryResolver:
    def __init__(self, aliases_csv: str | None = None, historical_csv: str | None = None):
        self._alias_to_canon: dict[str, str] = {}
        self._historical: dict[str, list[str]] = {}
        self._raw_names: set[str] = set()
        for canon, aliases in BUILTIN_ALIASES.items():
            self._add(canon, canon)
            for a in aliases:
                self._add(a, canon)
        for hist, modern in BUILTIN_HISTORICAL.items():
            self._historical[normalize(hist)] = [modern]
        if aliases_csv:
            self.load_aliases(aliases_csv)
        if historical_csv:
            self.load_historical(historical_csv)

    def _add(self, alias: str, canon: str, overwrite: bool = False):
        self._raw_names.add(str(alias).strip())
        k = normalize(alias)
        if k and (overwrite or k not in self._alias_to_canon):
            self._alias_to_canon[k] = canon

    def load_aliases(self, path: str):
        """CSV with columns: canonical, alias  (one alias per row)."""
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self._add(row["canonical"], row["canonical"])
                self._add(row["alias"], row["canonical"])

    def load_historical(self, path: str):
        """CSV with columns: historical, modern. `modern` may list several
        successor countries joined with "|" (e.g. for the Soviet Union)."""
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                moderns = [m.strip() for m in row["modern"].split("|") if m.strip()]
                if not moderns:
                    continue
                self._historical[normalize(row["historical"])] = moderns
                self._raw_names.update(moderns)
                self._add(row["historical"], row["historical"])

    def all_names(self) -> set[str]:
        """Every country name, alias and demonym loaded (original spelling)."""
        return set(self._raw_names)

    def canonical(self, name: str) -> str | None:
        """Canonical country for a name/alias/demonym, or None if unrecognized."""
        n = normalize(name)
        if n in self._alias_to_canon:
            return self._alias_to_canon[n]
        if n in self._historical:
            return name.strip()  # a known historical state is its own canonical
        return None

    def moderns(self, name: str) -> set[str]:
        """Modern-country equivalents: every successor of a historical state,
        else the canonical name itself; empty if unrecognized."""
        n = normalize(name)
        if n in self._historical:
            return {self.canonical(m) or m for m in self._historical[n]}
        c = self.canonical(name)
        return {c} if c else set()

    def modern(self, name: str) -> str | None:
        """One modern-country equivalent (the first successor listed)."""
        n = normalize(name)
        if n in self._historical:
            m = self._historical[n][0]
            return self.canonical(m) or m
        return self.canonical(name)
