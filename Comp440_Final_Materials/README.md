# Comp440_Final_Materials: LLM query + grading pipeline (Nadezhda's part)

COMP 440 final project: does LLM accuracy about people track Wikipedia coverage,
and does that differ by gender and region? This folder covers steps 3, 6, 7, 9–11
of the work division: question templates, the cached query script, the
automatic grader, the pilot runner and the hand-check tooling.

```
llm_eval/templates.py     step 3  question templates (name + Wikidata description, fixed "Answer:" format)
llm_eval/descriptions.py          removes answer leaks (nationality, years) from descriptions
llm_eval/query.py         step 6  LLM client with SQLite cache, retries, token/latency logging
llm_eval/grader.py        step 7  correct / wrong / refused; aliases, date formats, multiple valid answers
llm_eval/countries.py             country aliases, demonyms, historical -> modern mapping
run_eval.py               step 9  ask + grade + summary (runtime, cost) + hand-check sheets
score_handcheck.py        step 10 grader error rate from the filled hand-check sheets
scripts/fetch_country_aliases.py  full alias/demonym list from Wikidata (run locally)
tests/                            62 tests (grader + description cleaning):  python3 -m pytest -q
data/example_people.csv           6 illustrative rows showing the input format
```

## Setup

```bash
pip install -r requirements.txt
python scripts/fetch_country_aliases.py   # optional: writes data/country_aliases.csv
```

### The API key: on your computer only, never in the repo

The key goes in your shell settings file, outside the repo, under a
**project-only name**, `COMP440_ANTHROPIC_KEY`. Don't use the standard
`ANTHROPIC_API_KEY`: Claude Code and other tools read that name automatically and
would bill their usage to the project's key. The script refuses to use
`ANTHROPIC_API_KEY`, and it warns you if that variable is set.

1. `nano ~/.zshrc` (or `~/.bashrc` on a Linux server), and add at the bottom:
   `export COMP440_ANTHROPIC_KEY="sk-ant-..."`
2. Save (`Control+O`, `Enter`, `Control+X`), run `chmod 600 ~/.zshrc`, then open a new Terminal window.
3. Check it without printing the key:
   `python3 -c "import os; print(bool(os.environ.get('COMP440_ANTHROPIC_KEY')), bool(os.environ.get('ANTHROPIC_API_KEY')))"`
   should print `True False`.

The code reads the key with `os.environ.get("COMP440_ANTHROPIC_KEY")` in
`llm_eval/query.py` and sends it only in the request header to the API. It is never
printed, logged, cached or written to any output file. Only the person running the
queries needs the key; teammates can use the cache and results without it.
Never paste the key into code, notebooks or chats. Run `git diff --staged` before
committing. `git grep "sk-ant-" $(git rev-list --all)` should print nothing.

Add `cache/` and `results/` to `.gitignore` if you don't want them in the repo
(the cache is worth keeping somewhere shared, since it saves money on reruns).

## Input: the sample from Tu

One row per person. Default column names (rename with `--col-name`, `--col-birth-year`, etc.):

| column | example | notes |
|---|---|---|
| `qid` | Q7186 | |
| `name` | Marie Curie | full name |
| `description` | Polish-French physicist and chemist | Wikidata description, used to disambiguate |
| `birth_year` | 1867 | a year or ISO date; several valid values joined with `\|` |
| `citizenship` | Poland\|France | country labels joined with `\|` |
| `is_deceased` | true | optional; picks "is" vs "was" |

## The questions

System prompt: answer with one line `Answer: <value>`, or `Answer: unknown`.

- **birth_year**: "In what year was {name} ({description}) born? Give only the year as digits, e.g. 'Answer: 1952'. For years before the common era write e.g. 'Answer: 384 BC'. If you do not know, write 'Answer: unknown'."
- **citizenship**: "Of which country {is/was} {name} ({description}) a citizen? Give only the country name… If the person held more than one citizenship, list them separated by semicolons…"

### Descriptions are cleaned so they don't give away the answer

Wikidata descriptions often contain the answer: "**Polish-French** physicist
(**1867**–1934)". `llm_eval/descriptions.py` removes it per question:

| question | removed | "Polish-French physicist and chemist (1867–1934)" becomes |
|---|---|---|
| citizenship | country names, nationality words, "-born" | physicist and chemist, 1867–1934 |
| birth_year | years, date ranges, centuries, decades | Polish-French physicist and chemist |

The description actually sent is saved in the `description_used` column of
`responses.csv` and the hand-check sheets, so checkers can flag leaks that slipped
through. `--raw-descriptions` turns cleaning off (only for comparison).

**Limitations for the write-up:** cities, regions and institutions can still hint at
a country ("mayor of Chicago"). Roles still hint at an era. Uncommon nationality words
are only caught if they're in the alias table, so run
`scripts/fetch_country_aliases.py` and pass `--aliases-csv` for the real run.

Changing the wording? Bump `TEMPLATE_VERSION` in `templates.py` so cached answers to the old wording aren't reused.

## Running the pilot (step 9)

```bash
# dry run, no key needed
python run_eval.py --people data/example_people.csv --provider mock --model mock

# real pilot: 200 people x 2 question types
python run_eval.py --people data/sample.csv --limit 200 \
  --provider anthropic --model claude-sonnet-5-5 --price-in 2 --price-out 10 \
  --max-cost 1 --max-total-cost 10 \
  --historical-csv data/historical_to_modern.csv \
  --aliases-csv data/country_aliases.csv \
  --out results/pilot_sonnet
```

Check the model's current per-million-token prices before you fill in
`--price-in`/`--price-out`. For other providers, `--provider openai --base-url ...`
works with any OpenAI-compatible endpoint (Together, Groq, OpenRouter, a local
Ollama server at `http://localhost:11434/v1`). Use `--provider anthropic` for
Claude models. The proposal calls for two models of different sizes, so run
the command once per model with a different `--out`. For reasoning models,
raise `--max-tokens` (e.g. 1024), or they may be cut off before answering.

### Spending limits

Before sending anything, the script prints how many new calls are needed and the
worst-case cost, then asks `Proceed? [y/N]` (skip with `--yes`). Two hard caps:

- `--max-cost 1.00`: this run never spends more than $1.00
- `--max-total-cost 10.00`: all runs of this model together never pass $10.00
  (past spending is counted from the cache, so keep using the same cache file)

Both need `--price-in` and `--price-out`. The script reserves each call's worst case
(`--max-tokens` of output) before sending it, so it stops just under the cap. Questions
it didn't ask are marked `not_asked`. Raise the cap and rerun to finish them, and
answered questions won't be re-billed. The caps count what this script spends; they
are not enforced by Anthropic. For that, ask whoever owns the key to set a spend
limit in the Claude Console.

Outputs in `--out`:
- `responses.csv`: one row per question, with the prompt, raw response, parsed answer, `outcome`, `outcome_strict`, flags, latency and tokens
- `summary.json`: wall time, mean latency, tokens, estimated cost, projected cost per 1,000 people, outcome rates by question type
- `handcheck_Tu.csv`, `handcheck_Nadezhda.csv`, `handcheck_Nayla.csv`: 17 answers each

Rerunning the same command costs nothing: every answer comes from the cache. That's
how you re-grade after fixing the grader (step 11). API errors are not cached and
are marked `api_error`, not `wrong`, so a rerun retries them.

## Hand-check and grader error (steps 10–11)

Each person fills `human_outcome` in their sheet with `correct`, `wrong` or `refused`
(plus `notes`), judging against the gold value **and** their own knowledge or a quick
lookup (Wikidata itself can be wrong or incomplete; note those cases). Then:

```bash
python score_handcheck.py results/pilot_sonnet
```

It reports the raw error rate with a 95% CI, error by question type, a confusion
table, and a **reweighted** error rate. The sheets over-sample "flagged" answers
(bad format, unrecognized country, historical-state match, decade-only year,
multiple values), where the grader is most likely to slip, so the raw rate overstates
the error. Report the reweighted number. Disagreements go to `grader_disagreements.csv`
and point you to what to fix.

## Grading rules

| situation | rule |
|---|---|
| no `Answer:` line | grade the whole reply; `format_ok = False` |
| "unknown", "I don't know", "not sure"… with no value | `refused` |
| a hedge plus a value ("1952, but I'm not sure") | graded on the value |
| several valid gold values | correct if the answer matches any |
| model gives several values | `outcome`: correct if any is valid; `outcome_strict`: only if all are |
| year formats | 1952, c. 1952, 1952 AD, 384 BC/BCE, ISO dates in gold |
| "1950s", "19th century" | `wrong`, flagged `imprecise` |
| `--year-tolerance 1` | ±1 year (robustness check, step 19) |
| aliases and demonyms | USA / U.S. / American → United States; England / British → United Kingdom |
| historical states | match their modern successor either way (Kingdom of Italy ↔ Italy) |
| unknown country name | plain string match against gold; flagged `unrecognized` |

**Known limitations to mention in the write-up:**
- England, Scotland and Wales count as the United Kingdom.
- The built-in historical map only covers unambiguous successors. Splits like
  Austria-Hungary, Yugoslavia or Czechoslovakia, and ancient states like
  "Classical Athens", need Tu's lookup table (`historical,modern` columns).
- Wikidata BC dates can be off by one year because there is no year zero; hand-check BC cases.
- The lenient `outcome` gives credit to a model that lists several countries if one
  is right. `outcome_strict` is there to check whether that matters.
