"""REFERENCE-FREE check: for every example the pipeline produced an answer
for (answerable AND unanswerable both -- hallucination can happen on
either), asks the judge whether the answer is actually supported by the
context that was retrieved for it. No ground-truth answer is shown to the
judge -- see eval/judge.py's docstring for why that's the correct design
for this specific question ("is this grounded", not "is this correct").

This is the suite's direct hallucination measurement. It also cross-checks
the target system's OWN self-reported `grounded` flag (from
app/generator.py's _is_grounded()) against the judge's independent read --
but only over examples where the system claims grounded=True (it believes
it gave a real, substantiated answer). Refusals are deliberately excluded
from this specific comparison: a refusal text ("the documents don't cover
this") is, by definition, never unfaithful to any context -- there's no
claim in it to check -- so including refusals would make the two signals
disagree on every single correct refusal (grounded=False from the system,
but trivially "faithful"=True from the judge) for reasons that have
nothing to do with self-report accuracy. Restricting to grounded=True
isolates the actually interesting question: when the system claims
confidence, how often does the judge back that up? A gap here is a
sharper, more specific finding than the aggregate hallucination_rate alone
(which already includes this same signal, diluted across refusals too).

Runs judge calls concurrently -- this is always a real speedup regardless
of GENERATION_BACKEND, since the judge is always a direct OpenAI call
(see eval/judge.py), independent of whichever backend generated the
answer being judged.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

from eval import judge
from eval.pipeline import ExampleResult


def _judge_one(r: ExampleResult):
    v = judge.judge_faithfulness(answer=r.answer_text, context=r.context_text_en)
    return r, v


def run(results: list[ExampleResult], workers: int) -> dict:
    candidates = [r for r in results if r.error is None and r.answer_text]

    verdicts: list[tuple[ExampleResult, judge.JudgeVerdict]] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_judge_one, r) for r in candidates]
        for future in as_completed(futures):
            try:
                verdicts.append(future.result())
            except judge.JudgeNotConfigured:
                errors += 1

    if errors:
        return {
            "check": "faithfulness / hallucination (reference-free, LLM-as-judge)",
            "num_evaluated": 0,
            "error": "OPENAI_API_KEY not configured -- judge-based checks skipped.",
        }
    if not verdicts:
        return {"check": "faithfulness / hallucination (reference-free, LLM-as-judge)", "num_evaluated": 0}

    faithful = sum(1 for _, v in verdicts if v.verdict)
    n = len(verdicts)

    # Self-report precision: restricted to grounded=True (system claims a
    # real answer) -- see the module docstring for why refusals are excluded.
    confident_verdicts = [(r, v) for r, v in verdicts if r.answer_grounded]
    n_confident = len(confident_verdicts) or 1
    confident_disagreements = [
        {"query": r.example.query_en, "answer": r.answer_text[:200], "judge_reason": v.reason}
        for r, v in confident_verdicts
        if not v.verdict
    ]
    hallucinated_examples = [
        {"query": r.example.query_en, "answer": r.answer_text[:200], "judge_reason": v.reason}
        for r, v in verdicts
        if not v.verdict
    ]

    return {
        "check": "faithfulness / hallucination (reference-free, LLM-as-judge)",
        "num_evaluated": n,
        "faithful_rate": round(faithful / n, 4),
        "hallucination_rate": round(1 - faithful / n, 4),
        "num_confident_answers": len(confident_verdicts),
        "self_report_precision": round(1 - len(confident_disagreements) / n_confident, 4),
        "self_report_disagreements": confident_disagreements[:5],
        "hallucinated_examples": hallucinated_examples[:5],
    }
