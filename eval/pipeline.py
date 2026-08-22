"""Phase A: runs the target project's REAL retrieval and REAL generation
for every sampled example -- this is the only place either gets called.
Every check in eval/checks/ grades the output collected here; none of them
call app.retriever or app.generator directly.

Retrieval is bilingual (English query vs. the shared mixed-language index,
and Hindi query vs. the same index) since that's what the retrieval check
needs (see eval/checks/retrieval.py). Generation is English-only: the
system prompt, judge prompts, and MSMARCO-XI's Eng_Answer ground truth are
all English, so an English-only generation pass is what those checks can
actually grade correctly -- see README.md's "Scope" section for why Hindi
generation grading isn't included.

Parallelism: examples are processed concurrently via a thread pool. This
is a real speedup for any target whose generation call is a blocking
network request (network waits release the GIL, so multiple in-flight
requests genuinely overlap). It is NOT a real speedup -- and is actively
risky -- for a target holding one model on one local GPU: concurrent
threads calling into it would contend for the same CUDA device with no
throughput gain and a real risk of GPU memory pressure from multiple
simultaneous KV caches. This module reads the target's OPTIONAL
app.config.GENERATION_BACKEND (see eval/target.py's interface contract)
and clamps to 1 worker automatically when it's exactly "local" -- that
specific value is this suite's original target project's own convention,
not a general standard, so if your target uses local-GPU generation under
a different config name or value, the auto-clamp won't see it -- pass
--workers 1 yourself in that case.
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from eval import target
from eval.dataset import EvalExample
from eval.index_build import ChunkRecord


@dataclass
class RetrievedHit:
    query_id: int
    lang: str
    is_selected: bool
    score: float


@dataclass
class _Context:
    """Duck-typed context object handed to the target's generate_answer()
    -- deliberately NOT the target's own SearchResult class (see
    eval/target.py's interface contract: only `.text` / `.source` need to
    exist, under whatever attribute-access pattern the target's own
    generator uses)."""
    text: str
    source: str
    score: float


@dataclass
class ExampleResult:
    example: EvalExample
    retrieved_en: list[RetrievedHit] = field(default_factory=list)
    retrieved_hi: list[RetrievedHit] = field(default_factory=list)
    embed_ms_en: float = 0.0
    search_ms_en: float = 0.0
    embed_ms_hi: float = 0.0
    search_ms_hi: float = 0.0
    context_text_en: str = ""          # what was actually handed to the generator
    answer_text: str = ""
    answer_grounded: bool = False
    generation_ms: float = 0.0
    generation_model: str = ""
    error: str | None = None


def _search(query: str, index, records: list[ChunkRecord], top_k: int, embed_one):
    t0 = time.perf_counter()
    qvec = embed_one(query).reshape(1, -1)
    t1 = time.perf_counter()
    scores, indices = index.search(qvec, top_k)
    t2 = time.perf_counter()
    hits = [
        RetrievedHit(query_id=records[i].query_id, lang=records[i].lang, is_selected=records[i].is_selected, score=float(s))
        for s, i in zip(scores[0], indices[0])
        if i != -1
    ]
    return hits, (t1 - t0) * 1000, (t2 - t1) * 1000, [records[i] for i in indices[0] if i != -1]


def _process_one(ex: EvalExample, index, records: list[ChunkRecord], top_k: int) -> ExampleResult:
    embed_one = target.get_embedder().embed_one
    generate_answer = target.get_generator().generate_answer

    result = ExampleResult(example=ex)
    try:
        hits_en, embed_ms_en, search_ms_en, chunk_records_en = _search(ex.query_en, index, records, top_k, embed_one)
        hits_hi, embed_ms_hi, search_ms_hi, _ = _search(ex.query_hi, index, records, top_k, embed_one)
        result.retrieved_en = hits_en
        result.retrieved_hi = hits_hi
        result.embed_ms_en, result.search_ms_en = embed_ms_en, search_ms_en
        result.embed_ms_hi, result.search_ms_hi = embed_ms_hi, search_ms_hi

        search_results = [
            _Context(text=rec.text, source=f"msmarco-xi/q{rec.query_id}/{rec.lang}", score=hit.score)
            for rec, hit in zip(chunk_records_en, hits_en)
        ]
        result.context_text_en = "\n\n".join(sr.text for sr in search_results)

        answer = generate_answer(ex.query_en, search_results)
        result.answer_text = answer.text
        result.answer_grounded = answer.grounded
        result.generation_ms = answer.generation_ms
        result.generation_model = answer.model
    except Exception as e:  # noqa: BLE001 -- one bad example must not kill the whole run
        result.error = f"{type(e).__name__}: {e}"
    return result


def run(examples: list[EvalExample], index, records: list[ChunkRecord], top_k: int, workers: int) -> list[ExampleResult]:
    generation_backend = target.optional_config("GENERATION_BACKEND", default=None)
    effective_workers = 1 if generation_backend == "local" else max(1, workers)
    if effective_workers != workers:
        print(
            f"[pipeline] GENERATION_BACKEND=\"local\" -- clamping workers {workers} -> 1 "
            f"(single shared GPU model, see this module's docstring)."
        )
    elif generation_backend is None and workers > 1:
        print(
            f"[pipeline] target doesn't declare app.config.GENERATION_BACKEND -- running "
            f"{workers} workers as requested. If your target holds one shared model on one "
            f"local GPU, pass --workers 1 yourself (see this module's docstring)."
        )

    results: list[ExampleResult] = [None] * len(examples)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        # All examples are submitted up front, so up to `effective_workers`
        # run concurrently regardless of iteration order below -- as_completed
        # just lets progress printing reflect real completion order instead
        # of blocking on submission order.
        future_to_index = {pool.submit(_process_one, ex, index, records, top_k): i for i, ex in enumerate(examples)}
        done = 0
        for future in as_completed(future_to_index):
            results[future_to_index[future]] = future.result()
            done += 1
            if done % 10 == 0 or done == len(examples):
                print(f"[pipeline] {done}/{len(examples)} examples processed")
    return results
