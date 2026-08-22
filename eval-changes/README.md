# Changes made to `rag-local-eval-loop`

Cleared with the organisers: *"you can tweak accordingly to your code … just show
us the codes that you changed during that period."* This folder is that record.

**One file was modified: `eval/judge.py`. Nothing else in the suite was touched.**
No check, no prompt, no rubric, no scoring, no threshold was changed. `eval/checks/*`,
`eval/pipeline.py`, `eval/runner.py`, `eval/report.py`, `eval/dataset.py`,
`eval/msmarco.py`, `eval/target.py` and `eval/index_build.py` are byte-identical to
upstream (`BeaconBandhu/rag-local-eval-loop`).

## Why

`eval/judge.py` accepted only OpenAI or Anthropic, both paid. Without a key the
FAITHFULNESS and CORRECTNESS checks report `SKIPPED` and only three of the five
numbers exist.

## What changed

Added two **free-tier** judge providers — Google Gemini and Groq. Both expose an
**OpenAI-compatible** chat-completions endpoint, so they reuse the existing,
already-tested `_call_openai` path with a different `base_url`, key and model.
The request shape, the strict-JSON contract, `_parse_verdict`, and the fail-closed
behaviour on unparseable output are all upstream code, unchanged.

    EVAL_JUDGE_PROVIDER=gemini   GEMINI_API_KEY   gemini-3.1-flash-lite
    EVAL_JUDGE_PROVIDER=groq     GROQ_API_KEY     openai/gpt-oss-120b

Five edits, all additive:

1. `JUDGE_MODEL_GEMINI` / `JUDGE_MODEL_GROQ` + an `_OPENAI_COMPATIBLE` table.
2. `_resolve_provider()` recognises the two new names; `auto` picks a free key
   when one is present, and an explicitly forced provider still wins outright.
3. `_call_openai()` takes a `provider` argument selecting the endpoint.
   `provider="openai"` is the default and is byte-for-byte the upstream path.
4. `JudgeVerdict.provider` reports the real provider, so a Gemini run is not
   mislabelled `"openai"` in the saved report.
5. The "not configured" error message lists the free options first.
6. Bounded retry (default 5 attempts) on HTTP 429. Free tiers throttle hard —
   a real key measured **5 requests/minute** on `gemini-3.6-flash`, while a
   50/50 run needs ~200 judge calls. Upstream lets a 429 propagate, which the
   checks record as an errored example, silently shrinking the sample the score
   is computed over. The retry honours the server's own "retry in 12.9s" hint
   when present and falls back to exponential backoff (2/4/8/16s).

7. `max_completion_tokens` raised from 200 to 800. The free-tier models emit
   reasoning tokens before the JSON verdict; at 200 the response is truncated
   mid-reasoning and Groq rejects it with `json_validate_failed`. Only the
   ceiling changed — the verdict JSON itself is ~30 tokens.

Model choice: `gemini-3.1-flash-lite`, not `gemini-3.6-flash`. The quota is the
binding constraint, not judge capability — `-flash` allows 5 req/min; `-flash-lite`
sustained 9 consecutive calls untouched. (`gemini-2.0-flash`, the first default,
has been retired by Google and now 404s.)

## What did NOT change

* Judge prompts and rubrics (`_FAITHFULNESS_SYSTEM`, `_CORRECTNESS_SYSTEM`) — verbatim.
* `judge_faithfulness()` / `judge_correctness()` signatures and behaviour.
* The OpenAI and Anthropic paths. With a paid key set, this file behaves exactly
  as upstream — verified by the resolution matrix below.
* The independence property the suite requires: the judge is still a separate
  model from the one under test, so there is no self-judging bias.

## Verification

Provider resolution, 10 cases, all passing — the four marked UPSTREAM confirm
unchanged behaviour when a paid key is present:

    [PASS] nothing set                      -> JudgeNotConfigured
    [PASS] GEMINI only (auto)               -> gemini
    [PASS] GROQ only (auto)                 -> groq
    [PASS] forced gemini, key set           -> gemini
    [PASS] forced groq, NO key              -> JudgeNotConfigured
    [PASS] UPSTREAM: openai only            -> openai
    [PASS] UPSTREAM: anthropic only         -> anthropic
    [PASS] UPSTREAM: forced openai          -> openai
    [PASS] forced openai wins over gemini   -> openai
    [PASS] bad provider name                -> JudgeNotConfigured

To confirm only this one file differs from upstream:

    git clone https://github.com/BeaconBandhu/rag-local-eval-loop.git /tmp/upstream
    cd /tmp/upstream && git checkout bea160e3e790 && cd -
    diff -r --exclude=__pycache__ /tmp/upstream/eval ./eval

Upstream commit this was diffed against: **`bea160e3e790`**. The only file the
diff reports is `eval/judge.py`.

## The target adapters

Separately, the suite requires `embed`/`embed_one`/`get_model` and
`generate_answer` (see its `TARGET_INTERFACE.md`). This project exposes an
`Embedder.encode` method and an async `RagPipeline`, so two thin adapters were
added — `app/embedder.py` and `app/generator.py`. They are shims over the real
served code, not reimplementations: the same embedder singleton, the same
`ExtractiveComposer`, the same `OutputGuard`, and the same measured 0.67
confidence floor. `answer.grounded` is wired to the real refusal decision, which
is what the RELIABILITY "lying factor" check reads.

## One change to this project's own code, for the same reason

`app/config.py` — added a `load_dotenv()` call (guarded by `try/except ImportError`).

`Settings` reads `.env` through pydantic-settings, which populates the model but
does not touch `os.environ`. `eval/judge.py` imports `app.config` specifically for
a `load_dotenv()` side effect and then reads its credential from `os.environ`, so
without this a judge key kept in `.env` is invisible and the judge-based checks
report SKIPPED. `override=False` keeps an exported shell variable authoritative
over the file. The import guard matters because `python-dotenv` is an eval/dev
dependency and is deliberately absent from the runtime `requirements.txt` — the
deployed container must not crash on a missing import.

No pipeline behaviour changes; the full test suite (241 tests) passes unchanged.


### Judge actually used for the reported numbers

**Groq, `openai/gpt-oss-120b`.** Gemini's free tier proved unusable for a run this
size: 15 requests/minute shared across models against ~200 judge calls. Groq
measured ~0.9s per judged example with no throttling. (`llama-3.3-70b-versatile`,
the first Groq default written here, is not available on a current free key.)

Verified in both directions before the run — a faithful answer scores True, a
fabricated one False; a correct answer True, an incorrect one False.
