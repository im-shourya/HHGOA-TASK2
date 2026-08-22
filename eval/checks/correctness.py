"""REFERENCE-BASED check: for answerable examples only, does the generated
answer actually convey MSMARCO-XI's ground-truth Eng_Answer? Uses the
judge with the reference answer shown to it (see eval/judge.py's
judge_correctness docstring for why that's a different, and easier,
question than faithfulness -- and why an answer can be faithful to its
context yet still be judged incorrect here, if the retrieved context
itself never contained the right information: that combination is a
retrieval failure wearing a generation-shaped costume, and cross-
referencing this check against eval/checks/retrieval.py's numbers is how
the report tells the two apart).
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

from eval import judge
from eval.pipeline import ExampleResult


def _judge_one(r: ExampleResult):
    v = judge.judge_correctness(query=r.example.query_en, answer=r.answer_text, reference_answer=r.example.gt_answer_en)
    return r, v


def run(results: list[ExampleResult], workers: int) -> dict:
    candidates = [r for r in results if r.error is None and r.example.is_answerable and r.answer_text]

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
            "check": "correctness (reference-based, LLM-as-judge)",
            "num_evaluated": 0,
            "error": "OPENAI_API_KEY not configured -- judge-based checks skipped.",
        }
    if not verdicts:
        return {"check": "correctness (reference-based, LLM-as-judge)", "num_evaluated": 0}

    correct = sum(1 for _, v in verdicts if v.verdict)
    n = len(verdicts)
    incorrect_examples = [
        {
            "query": r.example.query_en,
            "reference_answer": r.example.gt_answer_en[:200],
            "answer": r.answer_text[:200],
            "judge_reason": v.reason,
        }
        for r, v in verdicts
        if not v.verdict
    ]

    return {
        "check": "correctness (reference-based, LLM-as-judge)",
        "num_evaluated": n,
        "correct_rate": round(correct / n, 4),
        "incorrect_examples": incorrect_examples[:5],
    }
