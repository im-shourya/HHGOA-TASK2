# Voice RAG over `ai4bharat/MSMARCO-XI`

Voice → speech-to-text → hybrid retrieval → grounded answer, with the whole core path
measured at **P50 7.87 ms / P70 9.62 ms / P100 40.71 ms** against a 200 ms budget, and a
system that declines rather than guesses when the corpus does not contain the answer.

Built for HH Goa 2026 Shortlisting Task 2 · `#RAGInGoa`

| | |
| --- | --- |
| **Live demo** | _add the deploy URL here_ |
| **Corpus** | `ai4bharat/MSMARCO-XI` validation split — 5,979 passages (2,990 `eng_Latn` / 2,989 `hin_Deva`), 155 labelled eval queries |
| **Index** | 18,416 chunks from five chunking strategies, 512-d static embeddings + BM25+ |
| **Latency** | in-process P50 7.87 / P70 9.62 / P100 40.71 ms · **over real HTTP** P50 6.04 / P70 7.00 / P100 58.30 ms · both 100% inside 200 ms |
| **STT** | Sarvam (`saaras:v3`) and ElevenLabs (`scribe_v1`), pluggable, with browser-side fallback |
| **Answering** | extractive by default (cannot hallucinate by construction); optional Claude path |
| **Tests** | 241, `pytest tests/ -q` |
| **Full measurements** | [`reports/benchmark.md`](reports/benchmark.md) |

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate     # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt                 # runtime + ingest + benchmark + tests

python scripts/ingest.py --queries 300              # download and shape the corpus (~60 s)
python scripts/build_index.py                       # chunk, embed, build BM25 (~40 s)
uvicorn app.main:app --reload                       # http://127.0.0.1:8000
```

`data/corpus/` is committed, so `scripts/build_index.py` alone is enough for a fresh
checkout. `data/index/` is **not** committed — it is derived data, and shipping an index
that silently disagrees with the corpus beside it is a class of bug worth designing out.

No API keys are required. With none set, the pipeline runs fully offline: the browser's
Web Speech API handles dictation and the extractive composer writes the answer. Copy
`.env.example` to `.env` to add Sarvam, ElevenLabs, or Anthropic keys.

The frontend dev server (optional — the container serves a static export):

```bash
cd web && npm install && npm run dev                # http://localhost:3000
```

---

## The pipeline

```
audio ──► transcribe ──► guard_input ──► classify ──► embed_query ──► retrieve
          (Sarvam /      (unsafe /       (intent,     (static         (dense + BM25+
           ElevenLabs)    injection /     script)      512-d)          → RRF → MMR)
                          malformed)                                       │
                                                                           ▼
   answer ◄── verify ◄────────── generate ◄────────────────── retrieval_guard
             (grounding,        (extractive, or Claude       (confidence ≥ 0.67,
              numeric,           when the budget allows)      else decline)
              citations,
              PII)
```

Every box is a named, timed span. Four of them can end the request with a refusal, each
carrying its own verdict and recorded reason. `transcribe` is the only stage excluded from
the 200 ms core budget — it is a network round-trip to a third party, and including it
would measure someone else's infrastructure.

---

## Chunking

The brief asks for "vast chunking" rather than one naive fixed-size pass. Five strategies
run over every passage, each making a *different* promise about what it will not break:

| Strategy | Parameters | The promise it makes | Chunks |
| --- | --- | --- | --- |
| `passage` | ≤ 220 words | Never dropped. The parent every child expands into, so a match on a 12-word window returns the whole passage as context. | 5,992 |
| `sentence_window` | 3 sentences, stride 1 | Boundaries fall on sentences, so an extracted quote is a whole sentence rather than a fragment. Records its centre sentence for precise attribution. | 6,207 |
| `semantic` | 78th-percentile distance, 20–130 words | Boundaries fall where meaning shifts, not where the word counter runs out. | 4,298 |
| `recursive_char` | 420 chars, 80 overlap | A hard character cap that holds even for scripts the separator cascade never saw — the cascade ends at the empty separator precisely so Devanagari cannot escape it. | 1,178 |
| `fixed_window` | 90 words, 24 overlap (26.7%) | Overlap guarantees no span is only ever seen cut in half. | 741 |

**Reconciliation is what makes running all five affordable.** 39,906 raw chunks collapse to
18,416 — a 53.9% dedup ratio — because five views of the same text become *one* chunk that
remembers every strategy that found it. 7,623 chunks carry more than one strategy. Nothing
is lost: provenance survives into the citation, and a strategy whose output merely restates
its parent is folded into that parent rather than kept as a second copy.

Two deliberate asymmetries:

- **Contextual headers go on `embed_text` only, never on `text`.** A chunk cut from the
  middle of a passage still retrieves on document-level context, but quoting the header
  back would put keyword soup in the answer. The two fields are different on purpose.
- **No gold query ever enters chunk metadata.** Headers are derived from the passage alone.
  Using the dataset's own query as metadata would inflate retrieval scores on exactly the
  queries used to evaluate it.

Chunk size after reconciliation: mean 45.6 words, P50 43, P95 90, max 220.

---

## Retrieval

Dense cosine over static embeddings, BM25+ over the same chunks, fused by reciprocal rank,
then diversified by MMR with one chunk per parent passage.

**Static embeddings are what make the budget reachable.** `minishlab/potion-retrieval-32M`
is a token-lookup model — a query is encoded by averaging rows out of a table, with no
transformer forward pass. Measured: **0.31 ms P50** to embed a query, against tens of
milliseconds for a small sentence-transformer on CPU. The 200 ms end-to-end figure is not
reachable on commodity CPU any other way.

**BM25+** (k1 = 1.4, b = 0.72, δ = 0.5) with plural folding applied symmetrically to the
postings and the query. The δ term matters here: MS MARCO passages are short, and plain
BM25 lets long-document normalisation push a single-occurrence match to zero.

**RRF fuses ranks, not scores** — which is the point and also the trap. A passage that is
mediocre on both lists outranks a passage that is dense #0 and absent from the sparse list.
That behaviour is correct (agreement between two differently-fallible retrievers is real
evidence) and it is pinned as an explicit test, because it once cost an eagle query its
right answer and the mechanism was not obvious from the output.

**MMR reorders.** The returned scores are *not* monotonically descending, and asserting
that they were would be asserting that diversification does nothing. The first slot is the
exception and has to be: with nothing yet selected the diversity penalty is zero, so slot 1
is plain argmax relevance.

### Measured quality

Against the dataset's own `is_selected` labels, 155 English / 154 translated queries:

| Query language | Recall@1 | Recall@4 | Recall@10 | MRR@10 | Mean confidence |
| --- | --- | --- | --- | --- | --- |
| English | 0.2968 | **0.7226** | 0.8645 | 0.4876 | 0.8189 |
| Hindi (translated) | 0.0844 | **0.2338** | 0.4675 | 0.1708 | 0.8488 |

The Hindi gap is real and is a property of the embedding profile, not a bug — see
[Embedding profiles](#embedding-profiles) below, where swapping the model nearly doubles
Hindi recall@4 and costs English eight points.

---

## Latency

**Core path: P50 7.87 ms · P70 9.62 ms · P100 40.71 ms · 100% of 927 runs within 200 ms.**

927 runs = 309 distinct queries × 3 repeats, query cache disabled, after warm-up.
P90 13.18 · P95 15.26 · P99 20.37 · mean 8.6 ms.

| Stage | P50 | P70 | P95 | P100 |
| --- | --- | --- | --- | --- |
| `guard_input` | 0.15 | 0.20 | 0.37 | 2.79 |
| `classify` | 0.05 | 0.07 | 0.11 | 2.84 |
| `embed_query` | 0.31 | 0.42 | 0.81 | 3.43 |
| `retrieve` | 4.30 | 4.96 | 7.46 | 18.71 |
| `retrieval_guard` | 0.00 | 0.01 | 0.01 | 0.08 |
| `generate` | 2.43 | 3.18 | 5.91 | 15.31 |
| `verify` | 0.79 | 1.03 | 1.95 | 15.86 |

| Language | P50 | P70 | P100 | n |
| --- | --- | --- | --- | --- |
| English | 6.99 | 8.24 | 40.71 | 465 |
| `hin_Deva` | 8.95 | 10.97 | 35.83 | 462 |

Four things about how this is measured, because each one is a way to report a number the
system does not actually have:

- **Warm-up is mandatory and is excluded.** A cold process answers its first query in
  ~29 ms and every one after it in single digits: BLAS selects kernels, numpy allocates
  scratch buffers, the tokeniser fills caches. Averaging one process-start into a P100
  describes the interpreter, not the pipeline. The cold figure is reported separately
  rather than hidden.
- **The query cache is off.** It is a real feature (`ENABLE_QUERY_CACHE`) and it would make
  the repeats free. Benchmarking with it on measures a dictionary.
- **The trace has to add up.** `core_latency_ms` equals the sum of its named spans, with
  `transcribe` as the single documented exclusion. This is asserted in the test suite, so
  time cannot quietly accumulate between stages.
- **Those numbers are in-process, not served.** `app/benchmark.py` constructs `RagPipeline`
  and awaits it directly, so it excludes uvicorn's accept loop, ASGI dispatch and response
  serialisation. That is the right way to measure the pipeline and the wrong way to answer
  "what does the live link give me". The served path is measured separately, below.

### The served path

What a grader clicking the live URL actually gets, over real HTTP
(`scripts/bench_http.py`, 300 sequential requests, one kept-alive connection):

| metric | P50 | P70 | P90 | P95 | P99 | P100 |
| --- | --- | --- | --- | --- | --- | --- |
| `core_latency_ms` (server-measured) | 6.04 | 7.00 | 7.93 | 8.55 | 9.85 | 58.30 |
| wall (client-observed) | 7.58 | 8.56 | 9.66 | 10.16 | 13.23 | 60.49 |

**Cost of being a web service: 1.56 ms mean.** 100% of requests inside the 200 ms budget;
`core` exceeded 20 ms once in 300 requests.

Getting that 1.56 ms right required discarding a first attempt. A client built on
`urllib.request` opens a **fresh TCP connection per request**, and it reported wall P50
30.9 ms against a 6.6 ms core — 23 ms of apparent "HTTP overhead". No browser and no
`fetch()` loop pays a connect handshake per call, so publishing that would have attributed
my client's artifact to the service. `--no-keepalive` still reproduces it on demand:

| client | wall P50 | mean overhead |
| --- | --- | --- |
| one kept-alive connection | 7.58 | **1.56 ms** |
| fresh connection per request | 30.89 | 23.28 ms |

### The one budget violation I measured, and why it is not the pipeline

Probing the running server after deliberate idle gaps, the first request after a 300 s idle
came back in **459 ms** once and **394 ms** on a later sample — over the 200 ms budget. That
is worth stating plainly rather than leaving inside a P100 that never sampled it.

It is a host wake-up artifact, not a pipeline property, and the evidence is specific:

- **Both processes stalled together.** On one sample the client waited 1799 ms while the
  server measured 197 ms of wall time for the same request. The ~1.6 s difference was lost
  inside the *client* — a separate Python process that had also been idle for 300 s. No
  server-side code path can slow down its own caller's interpreter.
- **The affected stage moved between samples** — spread across `embed_query`/`retrieve`/
  `generate` on one, concentrated in `generate` (375 ms against a 2–7 ms norm) on another.
  A code defect reproduces in the same place; a stall lands wherever the clock happens to be.
- **It was not an LLM call**, which is what a ~375 ms `generate` first suggests. The LLM path
  needs `mode == "llm"` or `"auto"` with budget headroom; these requests ran the default
  extractive mode and the responses confirm `generation_mode: extractive`. That span was
  pure local `composer.compose`.
- **Recovery is immediate** — the very next request was 7.19 ms.
- **The idle relationship is not monotonic** (20 s → no penalty, 50 s → 2.6×, 180 s → none,
  300 s → 394 ms), which is what host power management looks like and not what a warm-up
  curve looks like.

Two useful things came out of it anyway. The trace accounting held exactly through both
spikes (`span_sum == core_latency_ms` at 85.12 and 394.07), which is the invariant above
tested under stress rather than in the quiet case. And it names the real deployment risk
honestly: on a free tier the instance is *stopped* after idle, and no in-process keep-warm
can prevent that — it needs an external pinger, which is a hosting decision, not code. I did
not add a speculative keep-warm task, because the measurement did not support one.

### The GC finding

Gen-2 collections were a measured source of P100, not a guess. The index is ~112k live
container objects that never become garbage; CPython's generational collector walks them
anyway, and a walk landing inside a 6 ms request is a visible outlier.

`app/harness/gc_tuning.py` calls `gc.freeze()` after warm-up — moving **111,954 objects**
into the permanent generation — and raises the thresholds from `(700, 10, 10)` to
`(20000, 25, 25)`. The collector still runs; it just stops re-scanning an index that will
never be collected.

---

## The harness

`RagPipeline` is a harness rather than a function call in five specific senses:

1. **Structured I/O throughout.** Every stage boundary is a pydantic model — `AskRequest`
   in, `AskResponse` out, `RetrievedChunk`, `GroundingReport`, `GuardFinding`, `Span`. The
   LLM path returns a validated draft, not a string to be parsed.
2. **The deadline is an input, not a hope.** `TraceRecorder.remaining_ms()` decides whether
   the Claude path is affordable at all (floor: 1,200 ms). Below it the extractive composer
   runs instead and the span is marked `degraded` — visibly, in the response the UI renders.
3. **External calls are wrapped.** STT and the LLM get bounded retries with exponentially
   backed-off, jittered delays behind a circuit breaker (closed → open → half-open). The
   retry set is *narrow* on purpose: retrying a 401 spends the user's whole budget on a call
   that cannot succeed, so `retry_if` abandons it after one attempt. A fully failed call
   counts as **one** breaker failure, not one per attempt — otherwise a single flaky call
   trips a threshold-3 breaker by itself.
4. **Refusal is a first-class outcome.** Four exit points, each with its own verdict:
   `declined_unsafe`, `declined_injection`, `declined_malformed`, `declined_no_context`,
   `declined_ungrounded`.
5. **Errors return structure.** An unexpected exception becomes an `ERROR` verdict with a
   trace and a request id attached — never a bare 500, and never an internal message in
   user-facing text. Asserted in the test suite by breaking the retriever on purpose.

Both STT providers degrade to each other and then to the browser. `/api/config` tells the
frontend which providers are actually live, so the UI never offers a button that cannot work.

---

## Guardrails: knowing when *not* to answer

Three independent layers, because they fail differently.

### 1. Input guard — before the index is touched

Unsafe requests, prompt injection, and malformed queries are refused without retrieval
running at all (asserted: `retrieve` does not appear in the trace of a blocked request).
PII is redacted before anything is logged.

**43/43 red-team cases correct**, spanning weapons (9), benign lookalikes (14), illicit (6),
malware (4), injection (3), malformed (3), self-harm (3), violence (1).

The `benign_lookalike` count carries the actual difficulty. A guard that blocks "how do I
make a pipe bomb" and also blocks "penalties for laundering money under Indian law" has not
solved the problem — money laundering is a compliance industry, counterfeit detection is a
bank function, and `kill` is something you do to a hung process. Before an asking phrase was
made mandatory for that category, **5 of 8 real legal/compliance questions were refused as
unsafe.** The phrasing fragment `_ASK` is written once and shared across rules, because when
it was spelled inline per rule, "how to make a pipe bomb" was blocked while "how do I make a
pipe bomb" sailed through — and a behavioural sweep only caught it by accident.

**Known limitation, accepted deliberately:** a bare imperative with no asking phrase
("launder my drug money") is not caught by the illicit rule, because requiring an asking
phrase there is what stops the guard refusing compliance questions. The trade is mitigated
by extractive generation — the system can only quote passages that exist in an MS MARCO
slice, so there is no instruction to emit even if the query gets through.

### 2. Retrieval guard — the discriminator

Confidence fuses three signals that fail in different ways: dense cosine (fooled by
topical-but-irrelevant text), BM25 (fooled by common words), and query-term coverage (fooled
by paraphrase). Agreement between them is what "confidence" means here.

**The 0.67 floor is measured, not chosen.** Against 155 answerable queries and a 56-query
holdout whose evidence was deliberately never indexed:

| | Median confidence | |
| --- | --- | --- |
| Answerable (n=155) | **0.8402** | p10 0.6812 |
| Unanswerable holdout (n=56) | **0.5494** | p90 0.6824 |

At 0.67 — the Youden-J optimum, **J = 0.7976** — the system answers 92.3% of answerable
queries and wrongly answers 12.5% of unanswerable ones (precision 0.9533, F1 0.9377). The
full threshold sweep is in [`reports/benchmark.md`](reports/benchmark.md).

Over the 927 benchmark runs: 828 answered, 75 declined for no context, 24 declined as
malformed. **The refusals are not decoration.**

### 3. Output guard — grounding verification

Lexical entailment per claim, at unigram and bigram level, against the cited passages.
Numbers get their own pass: a paraphrase that drifts is a nuisance, a *figure* that drifts is
a fabrication and the failure a reader is least able to catch. When only part of an answer is
unsupported the guard **repairs** — drops the unsupported claims, keeps the grounded
remainder, and says so in `warnings`. Refusal is what happens when nothing survives. PII is
never echoed back even when it is genuinely in the retrieved context.

---

## Measured findings

The parts worth reading even if you never run the code. Each of these was found by
measurement and several of them changed the design.

### Grounding verification is not answerability — and the numbers say so loudly

This is the finding I would most want a reviewer to see, because the intuitive reading is
wrong. Over the same two holdout populations, the grounding score separates them **barely at
all**: median 1.000 against 1.000, best achievable Youden J over any floor **0.018**.

That is expected rather than broken. The extractive generator copies its sentences verbatim
out of the retrieved context, so lexical coverage is ~1.0 by construction *whether or not
that context has anything to do with the question*. Grounding answers "is this claim
supported by the evidence cited". Answerability asks "is this evidence an answer". Different
questions — and the retrieval-confidence gate is what answers the second one, at J = 0.798.

So `MIN_GROUNDING_SCORE = 0.55` is a **backstop**, not a discriminator. It refuses 0 of 155
answerable drafts, which is the correct result for a backstop and a useless one for a gate.
It earns its place on the Claude path, where invention is possible, and through the numeric
pass on both paths. Reading the two roles as one is the mistake the module docstring exists
to prevent.

### Three defects with one cause: citation markers were being verified as prose

Found while writing a test for numeric grounding, which passed an answer whose only
surviving text was `[1]`. The markers are apparatus added *after* composition, and leaving
them in the text under verification broke the guard three ways simultaneously:

1. `extract_numbers` read `[1]` as the quantity **1**, so every cited answer reported a
   fabricated figure and every sentence carrying a marker was dropped as unsupported.
2. The sentence splitter treated ` [1]` after a full stop as its own sentence (`[` opens a
   sentence). That phantom sentence has no content tokens, so it scored 1.0 coverage and
   counted as *supported* — inflating the support ratio and mean coverage of every cited
   answer, and in the degenerate case letting the guard return `[1]` as a grounded answer.
3. (1) was invisible in the benchmark only because the number check fell back to a substring
   test, and the digit `1` is a substring of most numbers a web passage contains.

**Two bugs cancelling, which is why the numbers looked fine.** Over 155 answerable drafts,
mean grounding rose **0.9212 → 0.9949** once markers stopped being verified as prose. The
direction matters: defect (1) was penalising correct answers harder than (2) was inflating
them, so the visible symptom was good answers being trimmed, not bad ones getting through.
The substring fallback was replaced with exact-digit comparison — "the fee is 100" against
evidence saying "1000" is precisely the fabrication the pass exists to catch. All five
mechanisms are now pinned as regression tests, and `min_grounding` was re-measured rather
than assumed to still hold.

### Plural folding, and where it deliberately does *not* apply

Folding is applied symmetrically to the BM25 postings and the query, to the extractive
selector's term coverage, and to `_focus_coverage`. Asymmetric folding is worse than none:
a query term folded on one side only scores exactly **0.0** against a document that
contains it. It is deliberately *absent* from the grounding check, where both sides are the
same text and folding would only add a way to disagree with itself.

### Embedding profiles: a real trade, quantified

| | `potion-retrieval-32M` (shipped) | `potion-multilingual-128M` |
| --- | --- | --- |
| Dim / size | 512-d, 129 MB | 256-d, 512 MB |
| English recall@4 | **0.7226** | 0.6387 |
| English MRR@10 | **0.4876** | 0.4224 |
| Hindi recall@4 | 0.2338 | **0.4545** |
| Hindi MRR@10 | 0.1708 | **0.3370** |
| Confidence gate J | **0.7976** (at 0.67) | 0.6874 (at 0.66) |

The multilingual model nearly doubles Hindi recall@4 and costs eight points of English. It
ships as the documented alternative rather than the default for two reasons: 512 MB of model
does not fit a 512 MB container beside an index, and the confidence gate discriminates
measurably *worse* with it (J 0.687 vs 0.798) — which matters more here than raw recall,
because the gate is what carries the refusal decision.

Set `EMBEDDING_MODEL=minishlab/potion-multilingual-128M` and `EMBEDDING_DIM=256` to switch;
the full comparison run is in [`reports/multilingual/`](reports/multilingual/). One caveat
stated plainly: that run predates the GC fix (`gc: null` in its JSON), so its latency column
is **not** comparable to the headline figures and no conclusion should be drawn from it.

### Gates that were measured and rejected

| Gate | Why it is off |
| --- | --- |
| **Coherence floor** in the extractive composer — require an added sentence to resemble the one it joins, not just the query | It does clean up individual answers (at 0.40 the eagle query loses an Amtrak *Texas Eagle* distractor), but gold-answer recall falls monotonically: 0.423 off, 0.420 at 0.30, 0.403 at 0.40, 0.357 at 0.60 over 143 queries. Entity collisions are a retrieval problem; paying two points of corpus-wide recall to hide one is a bad trade. |
| **Tighter padding floor** (0.50 → 0.70+) | Looks like it would trim padding. What it actually trims is the answer: 0.50 → 0.408 recall at 33.5 words, 0.70 → 0.318 at 17.5, everything above 0.75 collapses to one sentence at 0.317. On this corpus the supporting sentences are where the rest of the gold content lives. |
| **Grounding floor as an answerability gate** | Best achievable J = 0.018. See above — it cannot do this job on an extractive generator, and believing it does is how a system ends up with an unguarded refusal path. |
| **Score-margin gate** (`dense_top − second ≥ margin`) | The margin is computed and reported on every request because it is genuinely informative for debugging, but it is not gated on: it fires on exactly the queries where several passages are all legitimately relevant, which is a good retrieval outcome, not a reason to refuse. |
| **HNSW vector backend** | Not worth it below ~120k chunks. At 18,416 chunks a flat 512-d matmul is 4.3 ms P50 and exact; an ANN index would trade recall for time the budget does not need saved. The backend switches automatically past the threshold. |

---

## Speech-to-text

Provider-pluggable, both graded options implemented:

| Provider | Model | Notes |
| --- | --- | --- |
| Sarvam | `saaras:v3` | Tried first under `STT_PROVIDER=auto`; strong on Indic audio, which is what this corpus is for |
| ElevenLabs | `scribe_v1` | Fallback, or primary via `STT_PROVIDER=elevenlabs` |
| Browser | Web Speech API | Automatic when no key is configured — the demo works with zero credentials |

Transient failures retry with jittered backoff; permanent ones (bad key, unsupported codec)
skip straight to the next provider rather than burning attempts. STT latency is reported
separately in `stt_latency_ms` and excluded from the core budget.

---

## Verification

```bash
pytest tests/ -q                                    # 241 tests, ~3 s
python benchmark.py                                 # the reference harness, unmodified
python -m app.benchmark                             # + P70/P100 and the full core path
python scripts/benchmark.py                         # writes reports/benchmark.{md,json}
python scripts/bench_http.py --url http://…         # the served path, against a live server
```

**`benchmark.py` at the repo root is the harness shipped with the task, byte-for-byte
unmodified, and it passes** — retrieval-only p95 lands at 4–5 ms across runs against its
50 ms sub-budget (embed 0.24 ms P50, search 3.51 ms P50). Its contract — `search()`,
`warmup()`, `total_ms`/`embed_ms`/`search_ms` — is pinned by `tests/test_pipeline.py` so a
refactor cannot silently break it.

`python -m app.benchmark` is the same harness with two *additions* and no removals: P70 and
P100 columns (the brief asks for P50/P70/P100; the original printed P50/P95/P99), and a
second section running the same queries through the full core path. A retrieval-only PASS
says nothing about the number that is actually graded.

`scripts/bench_http.py` is the only one that needs a running server, because it is the only
one measuring what a visitor gets. Point it at the deployed URL to reproduce the served-path
table above.

Test suite layout:

| Module | Tests | Covers |
| --- | --- | --- |
| `test_guardrails_input.py` | 76 | Red-team matrix, benign lookalikes, phrasing variants, PII |
| `test_text.py` | 31 | Sentence splitting, plural folding, number extraction, fragments |
| `test_chunking.py` | 28 | Five strategies' promises, reconciliation invariants |
| `test_pipeline.py` | 28 | Harness contract + end-to-end refusals, budget, trace accounting |
| `test_harness.py` | 23 | Retries, backoff bands, circuit breaker, GC tuning |
| `test_output_guard.py` | 21 | Grounding, repair, numeric fabrication, the marker regressions |
| `test_extractive.py` | 21 | Selection, redundancy, citation rendering |
| `test_retrieval.py` | 13 | BM25+, RRF (incl. the eagle property), MMR one-per-parent |

Unit tests run without a built index — CI should not need a 6,000-passage corpus to check
that plural folding works. The two integration modules skip rather than fail when
`data/index/` is absent.

---

## Deployment

One container serving both `/api/*` and the UI from the same origin — no CORS, no second
deploy, no static host to keep in sync.

```bash
docker build -t voice-rag . && docker run -p 8000:8000 voice-rag
```

The image builds the Next.js static export, bakes in the embedding model, and **builds the
index at build time**, because a cold container that downloads a model and indexes a corpus
takes ~90 s to become useful and free-tier hosts restart often. It runs as a non-root user
(uid 10001), pins BLAS to one thread (static embeddings are memory-bandwidth bound; a thread
per core adds contention to a 0.2 ms operation), and runs a single worker on purpose —
each would load its own copy of the index, and concurrency comes from asyncio.

Render: **New → Blueprint → point at this repo.** [`render.yaml`](render.yaml) declares the
service; all three API keys are `sync: false` so no secret enters the repo, and
`.dockerignore` excludes `.env*` so none is baked into an image layer.

### Health and readiness are two endpoints, on purpose

`/api/health` always answers **200**, even when the index failed to load, because its
consumer is the browser: `index_loaded: false` is what drives the red indicator in
`PipelineHeader`, and the UI's fetch client throws on any non-2xx, so gating it would blank
the banner exactly when there is something to report.

`/api/ready` answers **503** until the index is usable, and that is what `render.yaml` and
the container `HEALTHCHECK` point at. A deploy gate reading a always-200 endpoint promotes a
release whose every `/api/ask` returns 503 — a URL that loads but cannot answer, which is
the worst available failure for a graded link. With the split, a bad build fails its health
check and the last good deploy stays live.

Both branches are verified rather than assumed — pointing `INDEX_DIR` at a nonexistent path
gives `ready` → `503 {"ready": false, "reason": "no index at … — run scripts/build_index.py
first"}` while `health` → `200 {"status": "degraded", "index_loaded": false}`.

---

## Repository layout

```
app/
  chunking/        five strategies + the reconciliation pipeline
  retrieval/       static embedder, BM25+, vector store, hybrid fusion, index store
  generation/      extractive composer, intent classifier, optional Claude path
  guardrails/      input guard, output guard, shared policy regexes
  harness/         orchestrator, retry/circuit breaker, trace recorder, GC tuning
  stt/             Sarvam, ElevenLabs, provider registry
  retriever.py     the flat search()/warmup() surface the reference harness imports
  main.py          FastAPI app; serves the API and the exported UI
scripts/           ingest.py · build_index.py · benchmark.py · bench_http.py
tests/             241 tests
web/               Next.js 16 + Tailwind UI (live trace, guard trail, latency waterfall)
reports/           benchmark.md/json · http.json · multilingual/ (the alternative-profile run)
benchmark.py       the harness shipped with the task, unmodified
```
