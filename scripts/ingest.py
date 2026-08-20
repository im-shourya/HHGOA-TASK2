#!/usr/bin/env python
"""Pull a slice of ai4bharat/MSMARCO-XI into a local corpus.

The dataset is 11.4 M rows / 55 GB, and its rows API is currently broken
(ArrowNotImplementedError on the nested `passages` column), so this script goes
straight at the Parquet files over HTTP range requests and projects only the
columns it needs. Each validation shard holds exactly one target language, so a
single shard yields the English passages *and* their translations.

Outputs (all UTF-8 JSONL, safe to commit):

    data/corpus/passages.jsonl      {passage_id, text, lang, query_id, is_selected, ...}
    data/corpus/eval_queries.jsonl  {query_id, query, eng_query, gold_passage_ids, ...}
    data/corpus/manifest.json       provenance + counts

Example:
    python scripts/ingest.py --languages hin_Deva --queries 500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATASET = "ai4bharat/MSMARCO-XI"
PARQUET_API = f"https://datasets-server.huggingface.co/parquet?dataset={DATASET.replace('/', '%2F')}"
COLUMNS = [
    "target_lang",
    "query_id",
    "query_type",
    "Eng_Query",
    "query",
    "Eng_Answer",
    "Answer",
    "passages",
]
# Shard -> language map for the validation split, discovered by probing the
# `target_lang` column (see --refresh-shard-map). Cached here so a normal run
# needs one HTTP request instead of fourteen.
VALIDATION_SHARDS: dict[str, str] = {
    "asm_Beng": "0000.parquet", "ben_Beng": "0001.parquet", "guj_Gujr": "0002.parquet",
    "hin_Deva": "0003.parquet", "kan_Knda": "0004.parquet", "mal_Mlym": "0005.parquet",
    "mar_Deva": "0006.parquet", "npi_Deva": "0007.parquet", "ory_Orya": "0008.parquet",
    "pan_Guru": "0009.parquet", "san_Deva": "0010.parquet", "tam_Taml": "0011.parquet",
    "tel_Telu": "0012.parquet", "urd_Arab": "0013.parquet",
}


def passage_id(text: str) -> str:
    """Content-addressed id so the same passage never enters the corpus twice."""
    return "p" + hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def shard_url(split: str, filename: str) -> str:
    return (
        f"https://huggingface.co/datasets/{DATASET}/resolve/"
        f"refs%2Fconvert%2Fparquet/default/{split}/{filename}"
    )


def open_shard(url: str):
    import fsspec
    import pyarrow.parquet as pq

    return pq.ParquetFile(fsspec.filesystem("http").open(url, "rb"))


def refresh_shard_map(split: str, count: int = 14) -> dict[str, str]:
    """Re-probe which shard holds which language (cheap: one column, one batch)."""
    import concurrent.futures as futures

    def probe(i: int) -> tuple[str, str] | None:
        filename = f"{i:04d}.parquet"
        try:
            handle = open_shard(shard_url(split, filename))
            batch = next(handle.iter_batches(batch_size=8, columns=["target_lang"]))
            return batch.to_pylist()[0]["target_lang"], filename
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {filename}: {exc}", file=sys.stderr)
            return None

    with futures.ThreadPoolExecutor(max_workers=count) as pool:
        found = [r for r in pool.map(probe, range(count)) if r]
    return dict(sorted(found))


def stream_rows(
    split: str, filename: str, limit: int, skip: int = 0, batch_size: int = 256
) -> Iterator[dict[str, Any]]:
    handle = open_shard(shard_url(split, filename))
    seen = 0
    yielded = 0
    for batch in handle.iter_batches(batch_size=batch_size, columns=COLUMNS):
        for row in batch.to_pylist():
            seen += 1
            if seen <= skip:
                continue
            yield row
            yielded += 1
            if yielded >= limit:
                return


def build_corpus(
    languages: list[str],
    queries_per_language: int,
    split: str,
    include_translations: bool,
    shard_map: dict[str, str],
    skip: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    passages: dict[str, dict[str, Any]] = {}
    eval_queries: list[dict[str, Any]] = []

    for language in languages:
        filename = shard_map.get(language)
        if not filename:
            raise SystemExit(f"unknown language '{language}'. known: {sorted(shard_map)}")
        print(f"[ingest] {language} <- {split}/{filename} ({queries_per_language} queries, skip={skip})")
        started = time.perf_counter()
        seen_queries = 0

        for row in stream_rows(split, filename, queries_per_language, skip=skip):
            payload = row.get("passages") or {}
            english = payload.get("English_passages") or []
            translated = payload.get("Translated_passages") or []
            selected = payload.get("is_selected") or []
            query_id = int(row["query_id"])
            query_type = (row.get("query_type") or "UNKNOWN").upper()
            gold: list[str] = []

            variants: list[tuple[list[str], str]] = [(english, "eng_Latn")]
            if include_translations and translated:
                variants.append((translated, language))

            for texts, lang in variants:
                for position, text in enumerate(texts):
                    text = (text or "").strip()
                    if len(text) < 40:
                        continue
                    pid = passage_id(text)
                    is_selected = bool(selected[position]) if position < len(selected) else False
                    record = passages.setdefault(
                        pid,
                        {
                            "passage_id": pid,
                            "text": text,
                            "lang": lang,
                            "query_id": query_id,
                            "query_type": query_type,
                            "position": position,
                            "is_selected": is_selected,
                            "n_queries": 0,
                        },
                    )
                    record["n_queries"] += 1
                    record["is_selected"] = record["is_selected"] or is_selected
                    if is_selected:
                        gold.append(pid)

            if not gold:
                continue  # no labelled answer passage: useless for retrieval eval
            eng_query = (row.get("Eng_Query") or "").strip().lstrip(". ")
            translated_query = (row.get("query") or "").strip()
            # A handful of translations degenerate into repetition loops (one 15-char
            # English question came back 7.7k chars long). Drop the translation, keep
            # the English question rather than discarding the row.
            if len(translated_query) > max(400, 6 * max(len(eng_query), 1)):
                translated_query = ""
            eval_queries.append(
                {
                    "query_id": query_id,
                    "query": translated_query,
                    "eng_query": eng_query,
                    "query_type": query_type,
                    "target_lang": language,
                    "gold_answer": (row.get("Eng_Answer") or "").strip(),
                    "gold_answer_translated": (row.get("Answer") or "").strip(),
                    "gold_passage_ids": sorted(set(gold)),
                }
            )
            seen_queries += 1

        elapsed = time.perf_counter() - started
        print(
            f"[ingest] {language}: {seen_queries} usable queries, "
            f"{len(passages)} passages so far ({elapsed:.1f}s)"
        )
    return list(passages.values()), eval_queries


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--languages", nargs="+", default=["hin_Deva"],
        help=f"target languages to pull. known: {', '.join(sorted(VALIDATION_SHARDS))}",
    )
    parser.add_argument("--queries", type=int, default=500, help="queries per language")
    parser.add_argument("--split", default="validation", choices=["validation", "train"])
    parser.add_argument(
        "--no-translations", action="store_true",
        help="index only the English passages (skips the translated copies)",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "corpus")
    parser.add_argument(
        "--skip", type=int, default=0,
        help="skip this many leading rows (use with --holdout for a disjoint slice)",
    )
    parser.add_argument(
        "--holdout", action="store_true",
        help="write queries only to holdout_queries.jsonl and index none of their "
             "passages — these become the provably-unanswerable set the retrieval "
             "guardrail is evaluated against",
    )
    parser.add_argument(
        "--refresh-shard-map", action="store_true",
        help="re-probe which parquet shard holds which language",
    )
    args = parser.parse_args()

    shard_map = VALIDATION_SHARDS
    if args.refresh_shard_map or args.split != "validation":
        print("[ingest] probing shard -> language map …")
        shard_map = refresh_shard_map(args.split)
        print(json.dumps(shard_map, indent=2))

    started = time.perf_counter()
    passages, eval_queries = build_corpus(
        languages=args.languages,
        queries_per_language=args.queries,
        split=args.split,
        include_translations=not args.no_translations,
        shard_map=shard_map,
        skip=args.skip,
    )
    if not passages:
        print("[ingest] nothing ingested", file=sys.stderr)
        return 1

    if args.holdout:
        # Passages are intentionally discarded: an unanswerable query set is only
        # unanswerable if its evidence never reaches the index.
        write_jsonl(args.out / "holdout_queries.jsonl", eval_queries)
        print(
            f"[ingest] wrote {len(eval_queries)} holdout queries to "
            f"{args.out / 'holdout_queries.jsonl'} (passages discarded)"
        )
        return 0

    write_jsonl(args.out / "passages.jsonl", passages)
    write_jsonl(args.out / "eval_queries.jsonl", eval_queries)
    manifest = {
        "dataset": DATASET,
        "split": args.split,
        "languages": args.languages,
        "queries_per_language": args.queries,
        "include_translations": not args.no_translations,
        "passages": len(passages),
        "passages_by_lang": {
            lang: sum(1 for p in passages if p["lang"] == lang)
            for lang in sorted({p["lang"] for p in passages})
        },
        "eval_queries": len(eval_queries),
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ingest_seconds": round(time.perf_counter() - started, 1),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[ingest] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
