"""Terminal report: for every rate-shaped metric, prints the measured
value next to its ideal (1.0 for "should be true every time" metrics like
faithful/correct/recall, 0.0 for "should never happen" metrics like
hallucination/false-refusal/false-confidence rate) and the gap between
them -- this is deliberately a gap-to-perfect report, not a pass/fail
against an arbitrary bar picked by this suite, per the brief: show where
it's lacking and where it's already perfect. The one place an actual
threshold exists is latency, where the target project already declares a
real budget (LATENCY_BUDGET_MS) for retrieval, and this suite declares an
explicit, labeled target of its own for generation (see
eval/checks/latency.py) -- those two are shown as PASS/FAIL since they
already have a stated bar, unlike the rate metrics.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

_BAR_WIDTH = 24


def _gap_row(label: str, value: float, ideal: float) -> str:
    gap = value - ideal
    if abs(gap) < 0.0005:
        tag = "PERFECT"
    elif ideal >= value:
        tag = f"-{abs(gap) * 100:.1f}pp short"
    else:
        tag = f"+{abs(gap) * 100:.1f}pp over"
    filled = round(_BAR_WIDTH * min(max(value, 0.0), 1.0))
    bar = "#" * filled + "." * (_BAR_WIDTH - filled)
    return f"  {label:<28}{value:>7.3f}  [{bar}]  ideal {ideal:.3f}  {tag}"


def _section(title: str):
    print(f"\n{title}")
    print("-" * len(title))


def print_report(report: dict):
    print("=" * 70)
    print("RAG Local Eval Loop -- results")
    print("=" * 70)
    meta = report["meta"]
    print(f"Target project:     {meta['target_root']}")
    print(f"Generation backend: {meta['generation_backend']} ({meta['generation_model_hint']})")
    print(f"Dataset:            ai4bharat/MSMARCO-XI ({meta['language']}, {meta['split']})")
    print(
        f"Sample:             {meta['num_answerable']} answerable + {meta['num_unanswerable']} unanswerable "
        f"(seed={meta['seed']})"
    )
    print(f"Index:              {meta['num_chunks']} chunks (EN+HI) from {meta['num_examples']} examples' candidates")
    print(f"top_k:              {meta['top_k']}")

    r = report["retrieval"]
    _section("RETRIEVAL  (reference-based -- vs. MSMARCO-XI is_selected labels)")
    print(f"  {r['num_evaluated']} answerable queries evaluated")
    for variant_key, variant_label in (("cross_lingual", "cross-lingual (either language is a hit)"), ("same_language", "same-language only")):
        v = r[variant_key]
        print(f"\n  {variant_label}:")
        for k, val in v["recall_at_k"].items():
            print(_gap_row(f"Recall@{k}", val, 1.0))
        print(_gap_row("MRR", v["mrr"], 1.0))

    f = report["faithfulness"]
    _section("FAITHFULNESS / HALLUCINATION  (reference-free -- LLM-as-judge, no ground truth shown to judge)")
    if f.get("error"):
        print(f"  SKIPPED: {f['error']}")
    else:
        print(f"  {f['num_evaluated']} answers evaluated")
        print(_gap_row("Faithful rate", f["faithful_rate"], 1.0))
        print(_gap_row("Hallucination rate", f["hallucination_rate"], 0.0))
        print(_gap_row("Self-report precision", f["self_report_precision"], 1.0))
        print(
            f"  (self-report precision: of the {f['num_confident_answers']} answers the system itself marked "
            f"grounded=True, how many the judge also confirmed faithful -- refusals excluded, see "
            f"eval/checks/faithfulness.py)"
        )
        if f["hallucinated_examples"]:
            print("\n  Sample hallucinations flagged by the judge:")
            for ex in f["hallucinated_examples"][:3]:
                print(f"    - Q: {ex['query']}")
                print(f"      A: {ex['answer']}")
                print(f"      judge: {ex['judge_reason']}")

    c = report["correctness"]
    _section("CORRECTNESS  (reference-based -- LLM-as-judge vs. MSMARCO-XI Eng_Answer)")
    if c.get("error"):
        print(f"  SKIPPED: {c['error']}")
    else:
        print(f"  {c['num_evaluated']} answerable-query answers evaluated")
        print(_gap_row("Correct rate", c["correct_rate"], 1.0))
        if c["incorrect_examples"]:
            print("\n  Sample incorrect answers:")
            for ex in c["incorrect_examples"][:3]:
                print(f"    - Q: {ex['query']}")
                print(f"      expected: {ex['reference_answer']}")
                print(f"      got:      {ex['answer']}")

    rel = report["reliability"]
    _section('RELIABILITY / "LYING FACTOR"  (should-answer vs. did-answer)')
    print(_gap_row("False refusal rate", rel["false_refusal_rate"], 0.0))
    print("    (answerable per the dataset, but the system declined -- lost, not wrong)")
    print(_gap_row("False confidence rate", rel["false_confidence_rate"], 0.0))
    print("    (unanswerable per the dataset -- no candidate passage is relevant -- but the system answered anyway: fabrication)")
    if rel["false_confidence_examples"]:
        print("\n  Sample fabrications (system answered a genuinely unanswerable query):")
        for ex in rel["false_confidence_examples"][:3]:
            print(f"    - Q: {ex['query']}")
            print(f"      fabricated: {ex['fabricated_answer']}")

    lat = report["latency"]
    _section("LATENCY")
    print(f"  {'stage':<16}{'avg':>9}{'p50':>9}{'p95':>9}{'p99':>9}   (ms)")
    for stage in ("embed", "search", "retrieval_total", "generation"):
        b = lat[stage]
        print(f"  {stage:<16}{b['avg_ms']:>9.2f}{b['p50_ms']:>9.2f}{b['p95_ms']:>9.2f}{b['p99_ms']:>9.2f}")
    r_status = "PASS" if lat["retrieval_within_budget"] else "OVER BUDGET"
    g_status = "PASS" if lat["generation_within_target"] else "OVER TARGET"
    print(f"\n  Retrieval  p95 {lat['retrieval_total']['p95_ms']:.2f}ms vs. {lat['retrieval_latency_budget_ms']}ms budget  -> {r_status}")
    print(f"  Generation p95 {lat['generation']['p95_ms']:.2f}ms vs. {lat['generation_latency_target_ms']}ms target  -> {g_status}  (suite-chosen target, see eval/checks/latency.py)")

    print("\n" + "=" * 70)


def save_report(report: dict, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
