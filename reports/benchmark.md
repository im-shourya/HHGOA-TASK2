# Benchmark report

Generated 2026-08-19T12:45:38Z · Windows 11 (AMD64) · Python 3.12.0

## Headline latency (core RAG path)

Core path = input guardrail → query embedding → hybrid retrieval → retrieval guardrail → answer generation → grounding verification. Speech-to-text is a separate network call and is reported on its own below.

| Metric | Value |
| --- | --- |
| **P50** | **7.87 ms** |
| **P70** | **9.62 ms** |
| **P100 (worst case)** | **40.71 ms** |
| P90 | 13.18 ms |
| P95 | 15.26 ms |
| P99 | 20.37 ms |
| Mean | 8.6 ms |
| Budget | 200 ms |
| Within budget | 100.0% of 927 runs |
| Cold first call (excluded, see note) | 28.57 ms |

Measured over 927 runs (309 distinct queries × 3 repeats), query cache disabled, after warm-up.

## Per-stage breakdown (ms)

| Stage | P50 | P70 | P95 | P100 | Mean |
| --- | --- | --- | --- | --- | --- |
| classify | 0.05 | 0.07 | 0.11 | 2.84 | 0.06 |
| embed_query | 0.31 | 0.42 | 0.81 | 3.43 | 0.39 |
| generate | 2.43 | 3.18 | 5.91 | 15.31 | 2.86 |
| guard_input | 0.15 | 0.2 | 0.37 | 2.79 | 0.18 |
| retrieval_guard | 0.0 | 0.01 | 0.01 | 0.08 | 0.0 |
| retrieve | 4.3 | 4.96 | 7.46 | 18.71 | 4.71 |
| verify | 0.79 | 1.03 | 1.95 | 15.86 | 0.93 |

## By query language

| Language | P50 | P70 | P100 | n |
| --- | --- | --- | --- | --- |
| english | 6.99 | 8.24 | 40.71 | 465 |
| hin_Deva | 8.95 | 10.97 | 35.83 | 462 |

## Retrieval quality (dataset `is_selected` labels)

| Query language | Recall@1 | Recall@4 | Recall@10 | MRR@10 | Mean confidence | n |
| --- | --- | --- | --- | --- | --- | --- |
| english | 0.2968 | 0.7226 | 0.8645 | 0.4876 | 0.8189 | 155 |
| translated | 0.0844 | 0.2338 | 0.4675 | 0.1708 | 0.8488 | 154 |

## Guardrails (100% of 43 cases correct)

| Category | Passed | Total | Rate |
| --- | --- | --- | --- |
| benign_lookalike | 14 | 14 | 1.0 |
| illicit | 6 | 6 | 1.0 |
| injection | 3 | 3 | 1.0 |
| malformed | 3 | 3 | 1.0 |
| malware | 4 | 4 | 1.0 |
| self_harm | 3 | 3 | 1.0 |
| violence | 1 | 1 | 1.0 |
| weapons | 9 | 9 | 1.0 |

## Retrieval-confidence calibration

Answerable queries (n=155): median 0.8402, p10 0.6812.  
Unanswerable holdout (n=56, evidence never indexed): median 0.5494, p90 0.6824.

**Recommended `MIN_RETRIEVAL_SCORE` = 0.67** (answers 92% of answerable queries, wrongly answers 12% of unanswerable ones).

| Threshold | Answer rate | False-answer rate | Youden J |
| --- | --- | --- | --- |
| 0.1 | 1.0 | 1.0 | 0.0 |
| 0.14 | 1.0 | 1.0 | 0.0 |
| 0.18 | 1.0 | 1.0 | 0.0 |
| 0.22 | 1.0 | 1.0 | 0.0 |
| 0.26 | 1.0 | 0.9821 | 0.0179 |
| 0.3 | 1.0 | 0.9464 | 0.0536 |
| 0.34 | 1.0 | 0.9286 | 0.0714 |
| 0.38 | 1.0 | 0.9286 | 0.0714 |
| 0.42 | 0.9935 | 0.875 | 0.1185 |
| 0.46 | 0.9935 | 0.7679 | 0.2257 |
| 0.5 | 0.9871 | 0.6607 | 0.3264 |
| 0.54 | 0.9742 | 0.5357 | 0.4385 |
| 0.58 | 0.9677 | 0.4464 | 0.5213 |
| 0.62 | 0.9548 | 0.3036 | 0.6513 |
| 0.66 | 0.929 | 0.1429 | 0.7862 |
| 0.7 | 0.8581 | 0.0893 | 0.7688 |
| 0.74 | 0.8065 | 0.0714 | 0.735 |

## Index

```json
{
  "chunks": 18416,
  "passages": 5979,
  "bm25_vocab": 15303,
  "chunking": {
    "documents": 5979,
    "chunks_before_dedup": 39906,
    "chunks": 18416,
    "merged_duplicates": 21490,
    "dedup_ratio": 0.5385,
    "per_strategy": {
      "passage": 5992,
      "sentence_window": 6207,
      "semantic": 4298,
      "recursive_char": 1178,
      "fixed_window": 741
    },
    "words": {
      "mean": 45.6,
      "p50": 43.0,
      "p95": 90.0,
      "max": 220
    },
    "parents": 5992,
    "multi_strategy_chunks": 7623
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
