"""The "lying factor" check: a 2x2 of ground truth (is this query actually
answerable, per MSMARCO-XI's own labels) against system behavior (did the
target system attempt an answer, or decline?).

  should answer / did answer    -> correct attempt (content graded separately
                                    by eval/checks/correctness.py)
  should answer / declined      -> FALSE REFUSAL. The system had a real,
                                    retrievable answer available and
                                    claimed otherwise. This is the exact
                                    failure mode this project's own
                                    app/local_generator.py docstring
                                    documents for its Qwen3-0.6B backend
                                    (verified there against a real corpus
                                    example) -- this check measures its
                                    real rate against a much larger,
                                    dataset-drawn sample instead of one
                                    anecdote.
  unanswerable / declined       -> correct abstention
  unanswerable / did answer     -> FALSE CONFIDENCE. None of the 10
                                    candidate passages for this query
                                    actually answer it (verified by
                                    MSMARCO-XI's own is_selected labels,
                                    all zero) -- yet the system produced a
                                    confident-looking answer anyway. This
                                    is "lying" in the sharpest sense this
                                    suite can measure: an assertion with no
                                    basis in the retrieved evidence, on a
                                    query the dataset guarantees has no
                                    answer among the candidates. It is a
                                    worse failure than false refusal --
                                    false refusal loses an answer the user
                                    could have gotten; false confidence
                                    hands the user a fabrication.

"System attempted an answer" is read from the target project's own
answer.grounded flag (app/generator.py's _is_grounded()) -- the same
signal the dashboard itself uses to decide whether to show an answer or a
"not in the documents" message.
"""
from eval.pipeline import ExampleResult


def run(results: list[ExampleResult]) -> dict:
    usable = [r for r in results if r.error is None]
    answerable = [r for r in usable if r.example.is_answerable]
    unanswerable = [r for r in usable if not r.example.is_answerable]

    false_refusals = [r for r in answerable if not r.answer_grounded]
    false_confidences = [r for r in unanswerable if r.answer_grounded]

    n_ans = len(answerable) or 1
    n_unans = len(unanswerable) or 1

    return {
        "check": 'reliability / "lying factor" (answerable-vs-answered 2x2)',
        "num_answerable_evaluated": len(answerable),
        "num_unanswerable_evaluated": len(unanswerable),
        "false_refusal_rate": round(len(false_refusals) / n_ans, 4),
        "false_refusal_examples": [
            {"query": r.example.query_en, "expected_answer": r.example.gt_answer_en[:200]} for r in false_refusals[:5]
        ],
        "false_confidence_rate": round(len(false_confidences) / n_unans, 4),
        "false_confidence_examples": [
            {"query": r.example.query_en, "fabricated_answer": r.answer_text[:200]} for r in false_confidences[:5]
        ],
    }
