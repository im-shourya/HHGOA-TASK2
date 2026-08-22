"""CLI entrypoint. Orchestrates the whole eval loop in two phases:

  Phase A (eval/pipeline.py) -- real retrieval + real generation against
  every sampled MSMARCO-XI example. Sequential-per-example dependency (a
  check can't grade an example that hasn't been retrieved/generated yet),
  so this phase runs first and is parallelized internally across examples.

  Phase B -- the five independent checks (retrieval, faithfulness,
  correctness, reliability, latency) all grade the SAME completed Phase A
  results, and don't depend on each other -- so they're dispatched as
  concurrent futures here. This is the "multiple parallel checking
  mechanism" the brief asked for: five checkers running at once, two of
  which (faithfulness, correctness) further parallelize their own judge
  calls internally across examples. Retrieval/reliability/latency are pure
  aggregation over numbers Phase A already collected, so running them
  concurrently doesn't buy real wall-clock time -- they're dispatched
  through the same pool anyway for architectural consistency, not because
  they're a bottleneck; the real parallel speedup is in the judge calls
  and in Phase A's generation calls (see those modules' docstrings for
  which parts of this suite's parallelism are "real" vs. "structural").

Usage:
    python -m eval.runner
    python -m eval.runner --num-answerable 50 --num-unanswerable 50 --top-k 5
    python -m eval.runner --rag-root D:\path\to\RAG --workers 4 --judge-workers 10

Run with the TARGET project's own virtualenv Python (see README.md) -- this
process imports and executes that project's real app/* and training/*
modules in-process.
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Windows consoles commonly default to a legacy codepage (cp1252 verified on
# the machine this suite was built on) rather than UTF-8. Judge output and
# MSMARCO-XI text routinely contain characters that codepage can't encode
# (curly quotes, Hindi Devanagari) -- without this, a plain `print()` of
# that text crashes with UnicodeEncodeError partway through the report,
# losing the whole run's output. `errors="replace"` means worst case an
# unrenderable glyph shows as a placeholder instead of crashing; it never
# affects what's written to the saved JSON report, which is opened with an
# explicit encoding="utf-8" in eval/report.py's save_report().
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval import dataset, index_build, pipeline, target
from eval import report as report_mod
from eval.checks import correctness, faithfulness, latency, reliability, retrieval

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-answerable", type=int, default=25, help="answerable MSMARCO-XI rows to sample (default: 25)")
    parser.add_argument("--num-unanswerable", type=int, default=25, help="unanswerable rows to sample (default: 25)")
    parser.add_argument("--top-k", type=int, default=5, help="results retrieved per query (default: 5)")
    parser.add_argument("--workers", type=int, default=6, help="parallel workers for retrieval+generation (default: 6; auto-clamped to 1 if the target's GENERATION_BACKEND is \"local\")")
    parser.add_argument("--judge-workers", type=int, default=8, help="parallel workers for judge calls, per check (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed (default: 42)")
    parser.add_argument("--language", default="hin", help="MSMARCO-XI language code (default: hin)")
    parser.add_argument("--split", default="validation", help="MSMARCO-XI split (default: validation)")
    parser.add_argument("--rag-root", default=None, help="path to the target RAG project (default: RAG_PROJECT_ROOT env var, then ../RAG)")
    args = parser.parse_args()

    # verify_target(), not load_target() -- actually imports the target's
    # embedder/generator modules and checks the required functions exist
    # on them, so an incompatible target fails right here with one clear
    # message instead of partway through every example in Phase A, or
    # (worse) after minutes spent downloading MSMARCO-XI first.
    root = target.verify_target(args.rag_root)
    print(f"Target project: {root}")
    print(f"Embedder: {target.EMBEDDER_MODULE}  |  Generator: {target.GENERATOR_MODULE}")
    # All OPTIONAL per eval/target.py's interface contract -- purely cosmetic
    # labels for the report meta, so a missing one just prints "unknown"
    # rather than crashing a target that doesn't declare these config names.
    generation_backend = target.optional_config("GENERATION_BACKEND", default="unknown")
    local_model = target.optional_config("LOCAL_GENERATION_MODEL", default="unknown")
    hosted_model = target.optional_config("GENERATION_MODEL", default="unknown")
    model_hint = local_model if generation_backend == "local" else hosted_model

    print(f"Loading {args.num_answerable} answerable + {args.num_unanswerable} unanswerable examples from MSMARCO-XI...")
    examples = dataset.load_examples(
        num_answerable=args.num_answerable,
        num_unanswerable=args.num_unanswerable,
        seed=args.seed,
        language=args.language,
        split=args.split,
    )

    print("Building isolated mixed-language FAISS index from sampled candidate passages...")
    index, records = index_build.build_index(examples)
    print(f"Indexed {index.ntotal} chunks.")

    print(f"Running retrieval + generation ({args.workers} workers requested)...")
    results = pipeline.run(examples, index, records, top_k=args.top_k, workers=args.workers)

    errored = [r for r in results if r.error]
    if errored:
        print(f"\n[warning] {len(errored)}/{len(results)} examples errored during retrieval/generation:")
        for r in errored[:5]:
            print(f"  - q{r.example.query_id}: {r.error}")

    print("\nRunning checks in parallel: retrieval, faithfulness, correctness, reliability, latency...")
    with ThreadPoolExecutor(max_workers=5) as pool:
        f_retrieval = pool.submit(retrieval.run, results, args.top_k)
        f_faithfulness = pool.submit(faithfulness.run, results, args.judge_workers)
        f_correctness = pool.submit(correctness.run, results, args.judge_workers)
        f_reliability = pool.submit(reliability.run, results)
        f_latency = pool.submit(latency.run, results)

        retrieval_report = f_retrieval.result()
        faithfulness_report = f_faithfulness.result()
        correctness_report = f_correctness.result()
        reliability_report = f_reliability.result()
        latency_report = f_latency.result()

    full_report = {
        "meta": {
            "target_root": str(root),
            "generation_backend": generation_backend,
            "generation_model_hint": model_hint,
            "language": args.language,
            "split": args.split,
            "num_answerable": args.num_answerable,
            "num_unanswerable": args.num_unanswerable,
            "num_examples": len(examples),
            "num_chunks": index.ntotal,
            "top_k": args.top_k,
            "seed": args.seed,
            "num_errored": len(errored),
        },
        "retrieval": retrieval_report,
        "faithfulness": faithfulness_report,
        "correctness": correctness_report,
        "reliability": reliability_report,
        "latency": latency_report,
    }

    report_mod.print_report(full_report)
    out_path = report_mod.save_report(full_report, RESULTS_DIR)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    try:
        main()
    except target.TargetNotFound as e:
        # A wrong/incompatible --rag-root or RAG_PROJECT_ROOT is a normal,
        # expected user error, not a bug in this suite -- print just the
        # message eval/target.py already built (which names the exact
        # missing module/attribute and how to fix it) instead of a raw
        # traceback that buries it.
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)
