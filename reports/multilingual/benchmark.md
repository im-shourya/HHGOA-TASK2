# Benchmark report

Generated 2026-08-17T18:56:06Z · Windows 11 (AMD64) · Python 3.12.0

## Headline latency (core RAG path)

Core path = input guardrail → query embedding → hybrid retrieval → retrieval guardrail → answer generation → grounding verification. Speech-to-text is a separate network call and is reported on its own below.

| Metric | Value |
| --- | --- |
| **P50** | **7.12 ms** |
| **P70** | **10.31 ms** |
| **P100 (worst case)** | **261.87 ms** |
| P90 | 18.7 ms |
| P95 | 37.0 ms |
| P99 | 75.89 ms |
| Mean | 11.77 ms |
| Budget | 200 ms |
| Within budget | 99.68% of 618 runs |
| Cold first call (excluded, see note) | 266.11 ms |

Measured over 618 runs (309 distinct queries × 2 repeats), query cache disabled, after warm-up.

## Per-stage breakdown (ms)

| Stage | P50 | P70 | P95 | P100 | Mean |
| --- | --- | --- | --- | --- | --- |
| classify | 0.04 | 0.04 | 0.07 | 3.91 | 0.05 |
| embed_query | 0.32 | 0.44 | 1.61 | 186.57 | 0.82 |
| generate | 2.81 | 5.78 | 24.98 | 217.44 | 6.76 |
| guard_input | 0.09 | 0.1 | 0.17 | 38.04 | 0.16 |
| retrieval_guard | 0.0 | 0.0 | 0.01 | 0.03 | 0.0 |
| retrieve | 2.73 | 3.24 | 7.03 | 61.25 | 3.55 |
| verify | 0.67 | 0.8 | 1.39 | 3.24 | 0.74 |

## By query language

| Language | P50 | P70 | P100 | n |
| --- | --- | --- | --- | --- |
| english | 6.69 | 10.83 | 261.87 | 310 |
| hin_Deva | 7.3 | 9.73 | 82.52 | 308 |

## Retrieval quality (dataset `is_selected` labels)

| Query language | Recall@1 | Recall@4 | Recall@10 | MRR@10 | Mean confidence | n |
| --- | --- | --- | --- | --- | --- | --- |
| english | 0.2452 | 0.6387 | 0.8065 | 0.4224 | 0.7911 | 155 |
| translated | 0.2143 | 0.4545 | 0.5909 | 0.337 | 0.7828 | 154 |

## Guardrails (100% of 19 cases correct)

| Category | Passed | Total | Rate |
| --- | --- | --- | --- |
| benign_lookalike | 4 | 4 | 1.0 |
| illicit | 2 | 2 | 1.0 |
| injection | 3 | 3 | 1.0 |
| malformed | 3 | 3 | 1.0 |
| malware | 2 | 2 | 1.0 |
| self_harm | 2 | 2 | 1.0 |
| violence | 1 | 1 | 1.0 |
| weapons | 2 | 2 | 1.0 |

## Retrieval-confidence calibration

Answerable queries (n=155): median 0.8066, p10 0.6434.  
Unanswerable holdout (n=56, evidence never indexed): median 0.5782, p90 0.7133.

**Recommended `MIN_RETRIEVAL_SCORE` = 0.66** (answers 88% of answerable queries, wrongly answers 20% of unanswerable ones).

| Threshold | Answer rate | False-answer rate | Youden J |
| --- | --- | --- | --- |
| 0.1 | 1.0 | 1.0 | 0.0 |
| 0.14 | 1.0 | 1.0 | 0.0 |
| 0.18 | 1.0 | 1.0 | 0.0 |
| 0.22 | 1.0 | 1.0 | 0.0 |
| 0.26 | 1.0 | 1.0 | 0.0 |
| 0.3 | 1.0 | 1.0 | 0.0 |
| 0.34 | 1.0 | 0.9821 | 0.0179 |
| 0.38 | 0.9871 | 0.9643 | 0.0228 |
| 0.42 | 0.9871 | 0.9286 | 0.0585 |
| 0.46 | 0.9871 | 0.8929 | 0.0942 |
| 0.5 | 0.9871 | 0.7679 | 0.2192 |
| 0.54 | 0.9742 | 0.6607 | 0.3135 |
| 0.58 | 0.9677 | 0.5 | 0.4677 |
| 0.62 | 0.9226 | 0.2857 | 0.6369 |
| 0.66 | 0.8839 | 0.1964 | 0.6874 |
| 0.7 | 0.8258 | 0.1429 | 0.6829 |
| 0.74 | 0.7097 | 0.0714 | 0.6382 |

## Index

```json
{
  "chunks": 18376,
  "passages": 5979,
  "bm25_vocab": 17126,
  "chunking": {
    "documents": 5979,
    "chunks_before_dedup": 39882,
    "chunks": 18376,
    "merged_duplicates": 21506,
    "dedup_ratio": 0.5392,
    "per_strategy": {
      "passage": 5992,
      "sentence_window": 6207,
      "semantic": 4272,
      "recursive_char": 1169,
      "fixed_window": 736
    },
    "words": {
      "mean": 45.6,
      "p50": 43.0,
      "p95": 90.0,
      "max": 220
    },
    "parents": 5992,
    "multi_strategy_chunks": 7611
  },
  "corpus": {
    "dataset": "ai4bharat/MSMARCO-XI",
    "split": "validation",
    "languages": [
      "hin_Deva"
    ],
    "queries_per_language": 300,
    "include_translations": true,
    "passages": 5979,
    "passages_by_lang": {
      "eng_Latn": 2990,
      "hin_Deva": 2989
    },
    "eval_queries": 155,
    "ingested_at": "2026-08-17T18:24:13Z",
    "ingest_seconds": 62.1
  }
}
```
