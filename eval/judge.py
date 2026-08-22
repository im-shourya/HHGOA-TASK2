"""LLM-as-a-judge: the technique from the CampusX "LLM Eval Methods" video
this suite follows -- prompting an LLM to score another model's output
against a stated rubric, rather than exact/fuzzy string matching.

Two judge calls, matching that video's reference-based vs. reference-free
split exactly:

  judge_faithfulness()  -- REFERENCE-FREE. No ground-truth answer is given
                            to the judge at all -- only the retrieved
                            context and the generated answer. Scores
                            whether every claim in the answer is actually
                            supported by that context. This is the
                            hallucination check: a reference-free judge is
                            required here specifically because hallucination
                            is a property of the answer's relationship to
                            its *own* context, not to some external ground
                            truth -- an answer can be faithful to bad
                            context, or unfaithful even when the context
                            happens to be the same topic as a correct
                            reference answer.

  judge_correctness()   -- REFERENCE-BASED. Given the MSMARCO-XI ground-
                            truth answer (Eng_Answer) as the reference, and
                            the target system's generated answer, scores
                            whether they convey the same information. This
                            is what "correctness" means here -- e.g. is the
                            model right, not just non-hallucinatory (a
                            model can be faithful to its context and still
                            wrong, if the retrieved context itself doesn't
                            contain the correct answer).

Deliberately a *separate* call from whatever GENERATION_BACKEND produced
the answer under test (see eval/target.py) -- judging a model with itself,
using the same call that produced the answer, is a known bias risk (a
model is more likely to rate its own output favorably).

PROVIDER-AGNOSTIC ON PURPOSE: this suite is public, and whoever runs it
against their own RAG project won't necessarily have an OpenAI key --
they might have an Anthropic key instead, or a local-only setup with no
hosted API key at all. The judge picks whichever real, working credential
is actually present rather than assuming OpenAI:

  EVAL_JUDGE_PROVIDER=openai      force OpenAI (needs OPENAI_API_KEY)
  EVAL_JUDGE_PROVIDER=anthropic   force Anthropic (needs ANTHROPIC_API_KEY,
                                   or any credential `ant auth status` reports --
                                   see the Anthropic SDK's own auth resolution)
  EVAL_JUDGE_PROVIDER=auto        (default) OPENAI_API_KEY if present, else
                                   ANTHROPIC_API_KEY, else raise JudgeNotConfigured
                                   naming both env vars so the fix is obvious

Both providers are called with a strict JSON output contract (OpenAI:
`response_format={"type": "json_object"}`, verified working against
JUDGE_MODEL_OPENAI before use; Anthropic: `output_config.format` with an
explicit json_schema, which per Anthropic's own docs guarantees the first
content block is valid JSON matching the schema). The same tolerant
fallback parser backs both anyway, in case a provider ever returns
something unexpected -- fail closed (verdict=False) rather than crash the
whole run over one bad example.

No live Anthropic key was available in the environment this suite was
built and tested in -- the OpenAI path has been run end-to-end repeatedly
(see this repo's README for real output); the Anthropic path is written
directly from Anthropic's own current API documentation (verified
`output_config` schema shape, current exception classes), not guessed, but
has NOT been exercised against a live response here. If you're the first
to run it with an Anthropic key, and something's off, that's the part to
check first.
"""
import json
import os
import re  # LOCAL ADDITION: parsing retry-after hints out of 429 messages
import time
from dataclasses import dataclass

from eval import target

JUDGE_MODEL_OPENAI = os.environ.get("EVAL_JUDGE_MODEL_OPENAI", "gpt-5.4-mini")
JUDGE_MODEL_ANTHROPIC = os.environ.get("EVAL_JUDGE_MODEL_ANTHROPIC", "claude-opus-5")

# --- LOCAL ADDITION (not upstream): free-tier judge providers -------------
# Neither an OpenAI nor an Anthropic key was available for this submission, and
# both are paid. Gemini and Groq each expose an OpenAI-COMPATIBLE endpoint, so
# they reuse `_call_openai` verbatim with a different base_url and key rather
# than getting a hand-written client — the request/response shape, the JSON
# contract, the parser and the fail-closed behaviour are all the already-tested
# upstream path. Only the transport target changes.
#
#   EVAL_JUDGE_PROVIDER=gemini   needs GEMINI_API_KEY  (aistudio.google.com)
#   EVAL_JUDGE_PROVIDER=groq     needs GROQ_API_KEY    (console.groq.com)
#
# Both are still real, independent LLM judges — separate from anything the
# target system itself calls, which is the property eval/judge.py's docstring
# actually requires (no self-judging bias).
# Pinned to an explicit version rather than the floating `gemini-flash-latest`
# alias: an eval number is only reproducible if the judge that produced it can be
# named. (`gemini-2.0-flash` was retired and now 404s — override with
# EVAL_JUDGE_MODEL_GEMINI if this one is retired in turn.)
#
# `-flash-lite` rather than `-flash` because of the FREE-TIER QUOTA, measured
# against a real key: `gemini-3.6-flash` allows 5 requests/minute, which a
# 50/50 run (~200 judge calls) exhausts almost immediately — every call past the
# 5th in a minute 429s. `gemini-3.1-flash-lite` sustained 9 consecutive calls
# with no throttling. A grading judge scoring short answers against an explicit
# rubric is well within a lite model's ability; the quota is the binding
# constraint here, not judge capability.
JUDGE_MODEL_GEMINI = os.environ.get("EVAL_JUDGE_MODEL_GEMINI", "gemini-3.1-flash-lite")
# `llama-3.3-70b-versatile` was the first choice here but is not available on a
# current free Groq key (404 model_not_found). `openai/gpt-oss-120b` is, and is
# the strongest general model on the free tier — measured 0.8s per judged
# example against a real key.
JUDGE_MODEL_GROQ = os.environ.get("EVAL_JUDGE_MODEL_GROQ", "openai/gpt-oss-120b")

# provider -> (env var holding the key, OpenAI-compatible base URL, model)
_OPENAI_COMPATIBLE: dict[str, tuple[str, str, str]] = {
    "gemini": (
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        JUDGE_MODEL_GEMINI,
    ),
    "groq": (
        "GROQ_API_KEY",
        "https://api.groq.com/openai/v1",
        JUDGE_MODEL_GROQ,
    ),
}

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}

_openai_client = None
_anthropic_client = None
# LOCAL ADDITION: cached clients for the free-tier OpenAI-compatible providers.
_compat_clients: dict = {}


class JudgeNotConfigured(RuntimeError):
    """No usable judge credential available."""


@dataclass
class JudgeVerdict:
    verdict: bool          # True = faithful / correct, False = hallucinated / incorrect
    reason: str
    judge_ms: float
    provider: str
    raw: str                # raw judge output, kept for debugging/audit


def _resolve_provider() -> str:
    # Best-effort: if the target has an app.config that loads a .env (this
    # suite's original target project does, via python-dotenv), importing
    # it here guarantees that's happened before the env var checks below --
    # relying on some other module having imported it first is fragile to
    # call order. Not every target does this, or even has an app.config at
    # all (it's OPTIONAL per eval/target.py's interface contract), so this
    # is silently skipped rather than required -- either way, the actual
    # judge credential can just be set in the shell environment directly.
    target.load_target()
    try:
        import app.config  # noqa: F401 -- imported for its load_dotenv() side effect, if any
    except ImportError:
        pass

    forced = os.environ.get("EVAL_JUDGE_PROVIDER", "auto").lower()
    _valid = ("openai", "anthropic", "auto", *_OPENAI_COMPATIBLE)
    if forced not in _valid:
        raise JudgeNotConfigured(
            f"EVAL_JUDGE_PROVIDER={forced!r} is not one of: {', '.join(_valid)}."
        )

    # LOCAL ADDITION: free-tier OpenAI-compatible providers. Checked before the
    # paid ones under "auto" so that a machine with only a free key configured
    # resolves to it, and an explicitly forced provider still wins outright.
    for name, (env_var, _base_url, _model) in _OPENAI_COMPATIBLE.items():
        if forced == name:
            if not os.environ.get(env_var):
                raise JudgeNotConfigured(
                    f"EVAL_JUDGE_PROVIDER={name} needs {env_var} set in the environment."
                )
            return name
        if forced == "auto" and os.environ.get(env_var):
            return name

    if forced in ("openai", "auto") and os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if forced in ("anthropic", "auto") and (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return "anthropic"
    if forced == "anthropic":
        # Don't hard-fail here: the Anthropic SDK also resolves credentials from
        # an `ant auth login` profile or Workload Identity Federation env vars,
        # neither of which show up as a plain env var check -- let the actual
        # client call in _call_anthropic() be the final word (it has its own
        # broad failure handling for the "truly nothing configured" case).
        return "anthropic"
    raise JudgeNotConfigured(
        "The judge needs a real LLM credential and found none. Set one of:\n"
        "  GEMINI_API_KEY      (free tier -- aistudio.google.com, EVAL_JUDGE_PROVIDER=gemini)\n"
        "  GROQ_API_KEY        (free tier -- console.groq.com, EVAL_JUDGE_PROVIDER=groq)\n"
        "  OPENAI_API_KEY      (loaded via the target project's .env, see eval/target.py)\n"
        "  ANTHROPIC_API_KEY   (or ANTHROPIC_AUTH_TOKEN, or `ant auth login` -- see `ant auth status`)\n"
        "...or set EVAL_JUDGE_PROVIDER=anthropic explicitly if you're using a credential source "
        "that doesn't show up as one of the env vars above.\n"
        "Judge-based checks (faithfulness, correctness) can't run without one; retrieval, "
        "reliability, and latency checks don't need it."
    )


# LOCAL ADDITION: bounded retry for free-tier rate limits (see _call_openai).
_RATE_LIMIT_ATTEMPTS = int(os.environ.get("EVAL_JUDGE_RATE_LIMIT_ATTEMPTS", 5))
# Headroom for reasoning tokens ahead of the JSON verdict (see _call_openai).
_MAX_COMPLETION_TOKENS = int(os.environ.get("EVAL_JUDGE_MAX_TOKENS", 800))
# Providers whose models accept OpenAI's `reasoning_effort` knob (see _call_openai).
_LOW_EFFORT_PROVIDERS = frozenset({"groq"})
_EXTRA_BODY = {"reasoning_effort": os.environ.get("EVAL_JUDGE_REASONING_EFFORT", "low")}


def _retry_delay_s(error: Exception, attempt: int) -> float:
    """Seconds to wait before retrying a 429.

    Prefers the server's own "Please retry in 12.9s" hint when present — it
    knows when the quota window actually resets, which a fixed backoff can only
    guess at — and falls back to exponential backoff (2, 4, 8, 16s) otherwise.
    """
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(error))
    if match:
        return min(float(match.group(1)) + 1.0, 65.0)
    return float(2 ** (attempt + 1))


def _parse_verdict(raw: str) -> tuple[bool, str]:
    try:
        parsed = json.loads(raw)
        return bool(parsed["verdict"]), str(parsed.get("reason", ""))
    except (json.JSONDecodeError, KeyError, TypeError):
        # Judge didn't follow the JSON contract -- fail closed (treat as a
        # negative verdict) rather than silently dropping the example, and
        # keep the raw text so it's auditable in the saved report.
        return False, f"[judge output did not parse as expected JSON: {raw[:200]!r}]"


def _call_openai(system_prompt: str, user_content: str, provider: str = "openai") -> JudgeVerdict:
    global _openai_client
    import openai

    # LOCAL ADDITION: `provider` selects which OpenAI-compatible endpoint to
    # talk to. "openai" keeps the upstream behaviour exactly (default base_url,
    # OPENAI_API_KEY from the environment); a free-tier provider swaps the
    # base_url, key and model. Clients are cached per provider so a run does not
    # rebuild one per judged example.
    if provider == "openai":
        if _openai_client is None:
            _openai_client = openai.OpenAI()
        client, model = _openai_client, JUDGE_MODEL_OPENAI
    else:
        env_var, base_url, model = _OPENAI_COMPATIBLE[provider]
        if provider not in _compat_clients:
            _compat_clients[provider] = openai.OpenAI(
                api_key=os.environ[env_var], base_url=base_url
            )
        client = _compat_clients[provider]

    t0 = time.perf_counter()
    # LOCAL ADDITION: retry on rate limits. The free tiers this path exists to
    # support throttle aggressively (a real key measured 5 req/min on
    # gemini-3.6-flash), and upstream lets a 429 propagate — which the checks
    # count as an errored example, silently shrinking the sample the reported
    # score is computed over. Honouring the server's own retry-after and backing
    # off keeps the judged sample size equal to the requested one.
    # Bounded: after _RATE_LIMIT_ATTEMPTS the error propagates as before.
    last_error = None
    for attempt in range(_RATE_LIMIT_ATTEMPTS):
        try:
            # LOCAL ADDITION: free tiers meter TOKENS per minute, not just
            # requests — a real Groq key allows 1000 req/min but only 8000
            # tok/min, so ~8 judged examples exhaust the minute while 99% of
            # the request quota is still unused. Reasoning models spend most of
            # those tokens thinking before emitting a ~30-token verdict, so
            # `reasoning_effort="low"` is the single highest-leverage saving
            # here: measured 35 completion tokens instead of several hundred,
            # with the verdict unchanged on the same probes. Grading a short
            # answer against an explicit rubric is not a task that needs
            # extended reasoning. Sent only to providers that accept it.
            extra = dict(_EXTRA_BODY) if provider in _LOW_EFFORT_PROVIDERS else {}
            response = client.chat.completions.create(
                model=model,
                **extra,
                # LOCAL ADDITION: upstream's 200 is enough for a non-reasoning
                # model, but the free-tier models this path targets emit thinking
                # tokens before the JSON object. At 200 the response is truncated
                # mid-reasoning and Groq rejects it with json_validate_failed —
                # which looked like a prompt problem and was really a budget one.
                # Only the ceiling changes; the verdict JSON itself is ~30 tokens.
                max_completion_tokens=_MAX_COMPLETION_TOKENS,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            break
        except openai.RateLimitError as e:
            last_error = e
            if attempt == _RATE_LIMIT_ATTEMPTS - 1:
                raise
            time.sleep(_retry_delay_s(e, attempt))
    else:  # pragma: no cover - loop always breaks or raises
        raise last_error
    judge_ms = (time.perf_counter() - t0) * 1000
    raw = (response.choices[0].message.content or "").strip()
    verdict, reason = _parse_verdict(raw)
    # LOCAL ADDITION: report the actual provider, so the saved report says
    # "gemini"/"groq" rather than mislabelling a free-tier run as "openai".
    return JudgeVerdict(verdict=verdict, reason=reason, judge_ms=judge_ms, provider=provider, raw=raw)


def _call_anthropic(system_prompt: str, user_content: str) -> JudgeVerdict:
    global _anthropic_client
    try:
        import anthropic
    except ImportError as e:
        raise JudgeNotConfigured(
            "EVAL_JUDGE_PROVIDER=anthropic needs the `anthropic` package: pip install anthropic"
        ) from e

    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / ant auth profile

    t0 = time.perf_counter()
    try:
        response = _anthropic_client.messages.create(
            model=JUDGE_MODEL_ANTHROPIC,
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
            output_config={"format": {"type": "json_schema", "schema": _VERDICT_SCHEMA}},
        )
    except anthropic.AuthenticationError as e:
        raise JudgeNotConfigured(f"Invalid Anthropic credentials: {e}") from e
    except TypeError as e:
        # Verified in the target RAG project's own earlier history (see its
        # app/generator.py git log): when NO Anthropic credentials exist at
        # all -- not even a resolvable `ant auth login` profile -- the SDK
        # fails locally in request construction with a bare TypeError, not
        # AuthenticationError (that one only fires for a real 401 from the
        # API). Without this, "truly unconfigured" surfaces as a confusing
        # crash instead of the same clear JudgeNotConfigured message the
        # OpenAI path already gives.
        raise JudgeNotConfigured(
            f"Anthropic credentials could not be resolved (`ant auth status` may help diagnose): {e}"
        ) from e
    judge_ms = (time.perf_counter() - t0) * 1000
    raw = next((b.text for b in response.content if b.type == "text"), "").strip()
    verdict, reason = _parse_verdict(raw)
    return JudgeVerdict(verdict=verdict, reason=reason, judge_ms=judge_ms, provider="anthropic", raw=raw)


def _call_judge(system_prompt: str, user_content: str) -> JudgeVerdict:
    provider = _resolve_provider()
    if provider == "anthropic":
        return _call_anthropic(system_prompt, user_content)
    # LOCAL ADDITION: gemini/groq share the OpenAI-compatible call path.
    return _call_openai(system_prompt, user_content, provider=provider)


_FAITHFULNESS_SYSTEM = """You are a strict fact-checking judge for a retrieval-augmented \
generation system. You will be given CONTEXT (retrieved document chunks) and an ANSWER a \
model produced from that context. Judge ONLY whether every factual claim in the ANSWER is \
directly supported by the CONTEXT -- do not judge whether the answer is true in general, \
only whether the CONTEXT supports it. An answer that correctly says the context doesn't \
cover the question is faithful (verdict: true). An answer that states anything not \
present in or directly implied by the CONTEXT is unfaithful (verdict: false), even if that \
claim happens to be true in reality.

Respond ONLY with a JSON object: {"verdict": true or false, "reason": "one short sentence"}"""


def judge_faithfulness(answer: str, context: str) -> JudgeVerdict:
    user_content = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    return _call_judge(_FAITHFULNESS_SYSTEM, user_content)


_CORRECTNESS_SYSTEM = """You are a grading judge comparing a model's ANSWER to a QUESTION \
against a REFERENCE ANSWER known to be correct. Judge whether the ANSWER conveys the same \
core information as the REFERENCE ANSWER -- wording, length, and extra (correct) detail \
don't matter, only whether the key fact(s) match. If the ANSWER says the documents don't \
contain the information, or refuses to answer, that is INCORRECT (verdict: false) -- the \
REFERENCE ANSWER proves the information was answerable.

Respond ONLY with a JSON object: {"verdict": true or false, "reason": "one short sentence"}"""


def judge_correctness(query: str, answer: str, reference_answer: str) -> JudgeVerdict:
    user_content = f"QUESTION:\n{query}\n\nREFERENCE ANSWER:\n{reference_answer}\n\nANSWER:\n{answer}"
    return _call_judge(_CORRECTNESS_SYSTEM, user_content)
