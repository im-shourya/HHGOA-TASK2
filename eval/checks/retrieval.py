"""REFERENCE-BASED check: does retrieval actually find the MSMARCO-XI
passage marked is_selected=1 for each answerable query? Ground truth
(which passage is correct) comes directly from the dataset -- this is a
reference-based eval in the CampusX video's sense: we're comparing against
a known-correct label, not asking a judge to guess.

Two variants computed together:
  "cross_lingual" -- a hit counts regardless of which language chunk was
                      retrieved (English query finding the Hindi version of
                      the right passage still counts). This is the metric
                      that actually reflects what the target project's
                      embedding model was fine-tuned for -- cross-lingual
                      Hindi/English alignment on this exact dataset -- so
                      it's the headline number.
  "same_language"  -- a hit only counts if the retrieved chunk's language
                      matches the query's language. Included for
                      comparison; a large gap between the two numbers
                      would itself be a useful diagnostic (e.g. the model
                      leaning on lexical overlap rather than true
                      cross-lingual alignment).

Unanswerable examples are excluded from the denominator here (by
definition, no passage is correct to find) -- they're graded by
eval/checks/reliability.py instead.
"""
import statistics

from eval.pipeline import ExampleResult

K_LEVELS = (1, 3, 5)


def _rank(hits: list, query_id: int, want_lang: str | None) -> int | None:
    for i, hit in enumerate(hits, 1):
        if hit.query_id == query_id and hit.is_selected and (want_lang is None or hit.lang == want_lang):
            return i
    return None


def run(results: list[ExampleResult], top_k: int) -> dict:
    k_levels = [k for k in K_LEVELS if k <= top_k]
    answerable = [r for r in results if r.example.is_answerable and r.error is None]

    metrics = {}
    for variant, en_lang, hi_lang in (("cross_lingual", None, None), ("same_language", "en", "hi")):
        hits_at_k = {k: 0 for k in k_levels}
        reciprocal_ranks = []
        for r in answerable:
            rank_en = _rank(r.retrieved_en, r.example.query_id, en_lang)
            rank_hi = _rank(r.retrieved_hi, r.example.query_id, hi_lang)
            best_rank = min((x for x in (rank_en, rank_hi) if x is not None), default=None)
            reciprocal_ranks.append(1.0 / best_rank if best_rank else 0.0)
            for k in k_levels:
                if best_rank and best_rank <= k:
                    hits_at_k[k] += 1
        n = len(answerable) or 1
        metrics[variant] = {
            "recall_at_k": {k: round(hits_at_k[k] / n, 4) for k in k_levels},
            "mrr": round(statistics.mean(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
        }

    return {
        "check": "retrieval (reference-based)",
        "num_evaluated": len(answerable),
        "top_k": top_k,
        **metrics,
    }
