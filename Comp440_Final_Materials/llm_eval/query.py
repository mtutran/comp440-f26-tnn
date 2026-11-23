"""
query.py - sends our questions to the LLM and saves the answers.

Authors: Nadezhda Dominguez Salinas, AI Assistant (Claude Code)

COMP 440 Final Project, step 6: LLM query script with caching.

How it works:
    1. Before asking a question, check the cache (a SQLite file). If we've
       already asked this exact question, return the saved answer instead.
       This way no question is ever asked twice and reruns are free.
    2. Check the spending caps so we never go over budget.
    3. Send the question to the API. If we get rate limited or the server
       has an error, wait and try again (2s, 4s, 8s, ... up to 60s).
    4. Save the answer to the cache.

The API key is read from the COMP440_ANTHROPIC_KEY environment variable.
We don't use the usual ANTHROPIC_API_KEY name because other tools (like
Claude Code) automatically use that one and would spend the class's credits.
The key is never written to any file.

Example:
    client = LLMClient("anthropic", "claude-sonnet-5-5",
                       price_in=2, price_out=10, max_total_cost=8)
    resp = client.ask("In what year was Marie Curie (...) born? ...")
    print(resp.text)   # Answer: 1867
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict

import requests

from .templates import SYSTEM_PROMPT, TEMPLATE_VERSION


@dataclass
class LLMResponse:
    """
    Stores the result of one question sent to the model.

    Fields:
        text: what the model said, e.g. "Answer: 1867" ("" if the call failed)
        model, provider: which model answered
        input_tokens, output_tokens: token counts from the API (used for cost)
        latency_s: how long the API call took in seconds
        cached: True if the answer came from the cache (no API call, no cost)
        error: None if it worked, otherwise what went wrong.
               "budget_exceeded" means a spending cap skipped this question.
               "truncated" means the reply hit max_tokens and got cut off.
        stop_reason: why the model stopped, as the API reports it:
               "end_turn" (finished normally), "max_tokens" (cut off) or
               "refusal" (the model declined). None for old cached answers.
    """
    text: str
    model: str
    provider: str
    input_tokens: int | None
    output_tokens: int | None
    latency_s: float
    cached: bool
    error: str | None = None
    stop_reason: str | None = None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class ResponseCache:
    """
    Saves model answers in a SQLite database so we never ask the same
    question twice.

    Each row is one answer, looked up by a key (a hash of the request).
    I picked SQLite because it's built into Python, it's just one file,
    and if a run crashes halfway the answers so far are already saved.

    The worker threads in run_eval.py share this cache, so every database
    access goes through a lock to avoid two threads writing at once.
    """

    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # check_same_thread=False lets the threads share one connection;
        # the lock makes sure only one uses it at a time
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                " key TEXT PRIMARY KEY, provider TEXT, model TEXT,"
                " prompt TEXT, response_json TEXT, created_at REAL)"
            )
            self._conn.commit()

    @staticmethod
    def make_key(**parts):
        """
        Turns the request settings into a unique ID (a SHA-256 hash).

        The same settings always give the same key, and changing anything
        (model, question wording, temperature...) gives a different key.
        sort_keys=True makes sure the order of the arguments doesn't matter.
        """
        blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key):
        """Returns the saved answer as a dict, or None if we haven't asked this yet."""
        with self._lock:
            row = self._conn.execute(
                "SELECT response_json FROM responses WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, provider, model, prompt, resp):
        """Saves an answer. Commits right away so nothing is lost if the run crashes."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?)",
                (key, provider, model, prompt, json.dumps(resp), time.time()),
            )
            self._conn.commit()

    def responses_for(self, provider, model):
        """Returns all saved answers for one model (used to add up past spending)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT response_json FROM responses WHERE provider = ? AND model = ?",
                (provider, model),
            ).fetchall()
        return [json.loads(j) for (j,) in rows]

    def __len__(self):
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]


# ---------------------------------------------------------------------------
# API calls
#
# Every provider function takes the same arguments and returns
# (reply_text, input_tokens, output_tokens, stop_reason), so LLMClient can call any of
# them the same way. To add another provider, write another function like
# these and add it to PROVIDERS.
# ---------------------------------------------------------------------------

class RetryableError(Exception):
    """
    An API error that's worth retrying: 429 (rate limited) or 5xx (server
    problem). Other errors like a wrong key or bad request won't fix
    themselves, so those aren't retried.
    """
    pass


def _post(url, headers, payload, timeout):
    """Sends a POST request and returns the JSON response, or raises an error."""
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code == 429 or r.status_code >= 500:
        raise RetryableError(f"HTTP {r.status_code}: {r.text[:200]}")
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


def _call_anthropic(model, system, user, temperature, max_tokens,
                    base_url, api_key, timeout):
    """
    Calls Claude through Anthropic's Messages API.

    The system prompt goes in its own "system" field, and the reply comes
    back as a list of content blocks, so we join the text blocks together.
    """
    payload = {
        "model": model,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": max_tokens,
    }
    # Newer Claude models (like Sonnet 5.5) reject a temperature setting,
    # so we only send it if one was chosen on purpose.
    if temperature is not None:
        payload["temperature"] = temperature
    data = _post(
        f"{(base_url or 'https://api.anthropic.com/v1').rstrip('/')}/messages",
        {"x-api-key": api_key, "anthropic-version": "2023-06-01",
         "content-type": "application/json"},
        payload,
        timeout,
    )
    usage = data.get("usage") or {}
    text = "".join(b.get("text", "") for b in data.get("content", [])
                   if b.get("type") == "text")
    return text, usage.get("input_tokens"), usage.get("output_tokens"), data.get("stop_reason")


def _call_openai_compatible(model, system, user, temperature, max_tokens,
                            base_url, api_key, timeout):
    """
    Calls any API that uses OpenAI's chat format (OpenAI, Together, Groq,
    a local Ollama server...). We're only using Claude right now, but this
    lets us add a second model later for a robustness check.
    """
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    data = _post(
        f"{base_url.rstrip('/')}/chat/completions",
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload,
        timeout,
    )
    usage = data.get("usage") or {}
    choice = data["choices"][0]
    text = choice["message"].get("content") or ""
    # OpenAI calls a cut-off reply "length"; use the same name as Anthropic
    stop = "max_tokens" if choice.get("finish_reason") == "length" else choice.get("finish_reason")
    return text, usage.get("prompt_tokens"), usage.get("completion_tokens"), stop


def _call_mock(model, system, user, temperature, max_tokens, *_, **__):
    """
    Fake model for testing without an API key or spending money.

    The answer is based on a hash of the question, so it's the same every
    time. About 20% are refusals and the rest are random years/countries
    (in different formats like "USA" vs "United States") so the grader
    gets tested too. The answers aren't supposed to be correct.
    """
    h = int(hashlib.md5(user.encode()).hexdigest(), 16)
    time.sleep(0.01)
    kind = h % 10
    if "born" in user:
        if kind < 2:
            return "Answer: unknown", 60, 4, "end_turn"
        return f"Answer: {1850 + h % 150}", 60, 5, "end_turn"
    if kind < 2:
        return "I'm not sure who this is.", 60, 8, "end_turn"
    countries = ["United States", "France", "Kenya", "India", "USA", "Brazil",
                 "Japan", "United Kingdom", "Germany", "Nigeria"]
    return f"Answer: {countries[h % len(countries)]}", 60, 5, "end_turn"


PROVIDERS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai_compatible,
    "mock": _call_mock,
}

DEFAULT_BASE_URLS = {
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
    "mock": None,
}

# project-only name for the key so no other program on the computer uses it
PROJECT_ANTHROPIC_KEY_ENV = "COMP440_ANTHROPIC_KEY"

DEFAULT_KEY_ENV = {
    "anthropic": PROJECT_ANTHROPIC_KEY_ENV,
    "openai": "OPENAI_API_KEY",
    "mock": None,
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Asks the model questions, using the cache and the spending caps.

    run_eval.py makes one of these per run and shares it across threads.

    Main arguments:
        provider: "anthropic", "openai", or "mock" (free fake answers)
        model: e.g. "claude-sonnet-5-5"
        cache_path: the SQLite file. Keep using the same one so answers get
            reused and past spending is counted.
        temperature: None by default, so the model's own default is used.
            Newer Claude models like Sonnet 5.5 don't allow setting it, so
            only set it for a model that accepts it (e.g. 0.0 for the most
            likely answer every time).
        max_tokens: the longest reply allowed. 150 leaves room for Sonnet 5.5,
            which sometimes explains before answering (32 was too short and
            cut answers off), while still keeping each call cheap
        price_in, price_out: dollars per million tokens (Sonnet 5.5: 2 and 10)
        max_cost: max dollars for this run
        max_total_cost: max dollars for all runs together
    """

    def __init__(self, provider, model, cache_path="cache/llm_cache.sqlite",
                 base_url=None, api_key_env=None,
                 temperature=None, max_tokens=150,
                 max_retries=6, timeout=60.0,
                 system_prompt=SYSTEM_PROMPT,
                 price_in=0.0, price_out=0.0,
                 max_cost=None, max_total_cost=None):
        if provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {list(PROVIDERS)}")
        self.provider, self.model = provider, model
        self.base_url = base_url or DEFAULT_BASE_URLS[provider]

        # get the API key from the environment (never hardcoded)
        key_env = api_key_env or DEFAULT_KEY_ENV[provider]
        if provider == "anthropic" and key_env == "ANTHROPIC_API_KEY":
            raise RuntimeError(
                "Don't store the project key as ANTHROPIC_API_KEY: other tools (e.g. Claude "
                f"Code) use that name automatically. Use {PROJECT_ANTHROPIC_KEY_ENV} instead.")
        self.api_key = os.environ.get(key_env) if key_env else None
        if provider != "mock" and not self.api_key:
            raise RuntimeError(
                f"The {key_env} environment variable is not set. Add this line to ~/.zshrc "
                f"and open a new Terminal window:\n    export {key_env}=\"your-key\"")
        if provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            print("WARNING: ANTHROPIC_API_KEY is also set in this terminal. This script "
                  f"ignores it and uses {key_env}, but other tools (e.g. Claude Code) may "
                  "bill to it. Remove it from ~/.zshrc if it holds the project key.")

        self.temperature, self.max_tokens = temperature, max_tokens
        self.max_retries, self.timeout = max_retries, timeout
        self.system_prompt = system_prompt
        self.cache = ResponseCache(cache_path)

        # spending caps
        self.price_in, self.price_out = price_in, price_out
        self.max_cost, self.max_total_cost = max_cost, max_total_cost
        if (max_cost is not None or max_total_cost is not None) and provider != "mock" \
                and not (price_in or price_out):
            raise RuntimeError("A spending cap needs --price-in and --price-out to be set.")
        self._budget_lock = threading.Lock()
        self.spent_this_run = 0.0               # cost of finished calls this run
        self._reserved = 0.0                    # worst-case cost of calls in progress
        self.spent_before = self.cache_spend()  # cost of all earlier runs
        self.budget_hit = False

    # ----- cost tracking -----

    def cost(self, tin, tout):
        """Dollar cost of a call: tokens / 1 million * price per million."""
        return (tin or 0) / 1e6 * self.price_in + (tout or 0) / 1e6 * self.price_out

    def cache_spend(self):
        """
        How much we've already spent on this model in past runs.
        Every answer in the cache was paid for once, so we just add them up.
        """
        return sum(self.cost(d.get("input_tokens"), d.get("output_tokens"))
                   for d in self.cache.responses_for(self.provider, self.model))

    def _worst_case(self, user):
        """
        Overestimates the cost of a call before making it, since we only
        get the real token count after. Assumes ~1 token per 3 characters
        (it's usually closer to 4) and the max possible output length.
        """
        tin = (len(self.system_prompt) + len(user)) // 3 + 20
        return self.cost(tin, self.max_tokens)

    def _reserve(self, user):
        """
        Sets aside the worst-case cost before a call. Returns None if the
        call would go over a cap.

        This is needed because several threads run at once: without
        reserving, two threads could both see "$0.01 left" and both make a
        call, going over the cap.
        """
        w = self._worst_case(user)
        with self._budget_lock:
            run_total = self.spent_this_run + self._reserved + w
            if self.max_cost is not None and run_total > self.max_cost:
                self.budget_hit = True
                return None
            if self.max_total_cost is not None and \
                    self.spent_before + run_total > self.max_total_cost:
                self.budget_hit = True
                return None
            self._reserved += w
            return w

    def _settle(self, reserved, actual):
        """After a call finishes, swaps the reserved amount for the real cost."""
        with self._budget_lock:
            self._reserved -= reserved
            self.spent_this_run += actual

    # ----- asking questions -----

    def _key(self, user):
        """
        Cache key for a question. Includes everything that could change
        the answer, so if we change the model, the wording, the
        TEMPLATE_VERSION, etc., the question gets asked again.
        """
        return ResponseCache.make_key(
            provider=self.provider, model=self.model, system=self.system_prompt,
            user=user, temperature=self.temperature, max_tokens=self.max_tokens,
            template_version=TEMPLATE_VERSION,
        )

    def ask(self, user):
        """
        Asks one question and returns an LLMResponse.

        This doesn't raise on API errors. It returns them in resp.error
        instead, so one bad question doesn't crash a run of thousands.
        """
        # step 1: check the cache first
        key = self._key(user)
        hit = self.cache.get(key)
        if hit is not None:
            hit["cached"] = True
            return LLMResponse(**hit)

        # step 2: make sure we have budget left
        fn = PROVIDERS[self.provider]
        reserved = self._reserve(user)
        if reserved is None:
            return LLMResponse(text="", model=self.model, provider=self.provider,
                               input_tokens=None, output_tokens=None, latency_s=0.0,
                               cached=False, error="budget_exceeded")

        # step 3: call the API, retrying temporary errors with exponential backoff.
        # The random extra wait (jitter) keeps the threads from all retrying at once.
        delay = 2.0
        last_err = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                text, tin, tout, stop = fn(self.model, self.system_prompt, user,
                                           self.temperature, self.max_tokens,
                                           self.base_url, self.api_key, self.timeout)
                resp = LLMResponse(text=text, model=self.model, provider=self.provider,
                                   input_tokens=tin, output_tokens=tout,
                                   latency_s=round(time.perf_counter() - t0, 3),
                                   cached=False, stop_reason=stop)
                # step 4: record the real cost (we're charged even if it got cut off)
                self._settle(reserved, self.cost(tin, tout))
                # A cut-off reply isn't a real answer, so we don't grade it or
                # save it. Rerunning with a bigger --max-tokens asks it again.
                if stop == "max_tokens":
                    resp.error = "truncated"
                    return resp
                self.cache.put(key, self.provider, self.model, user, asdict(resp))
                return resp
            except (RetryableError, requests.ConnectionError, requests.Timeout) as e:
                last_err = str(e)
                if attempt == self.max_retries:
                    break
                time.sleep(delay + random.uniform(0, 1))
                delay = min(delay * 2, 60)
            except Exception as e:
                # wrong key, bad request, etc. - retrying won't help
                last_err = str(e)
                break

        # the call failed, so we didn't get charged - release the reservation.
        # errors aren't cached, so running the command again will retry them.
        self._settle(reserved, 0.0)
        return LLMResponse(text="", model=self.model, provider=self.provider,
                           input_tokens=None, output_tokens=None, latency_s=0.0,
                           cached=False, error=last_err)
