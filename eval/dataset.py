"""Builds the eval query set straight from ai4bharat/MSMARCO-XI -- no
hand-written queries anywhere in this suite. Loads it via eval/msmarco.py,
this suite's own parquet loader -- the dataset is the fixed constant
across everyone who runs this suite; it does NOT come from the target
project (see eval/msmarco.py's docstring for why that's a deliberate
design change, not an implementation detail).

Two buckets, both real and both useful, discovered by direct inspection of
the dataset (not assumed):

  "answerable"   -- passages.is_selected contains a 1, and Eng_Answer /
                    Answer hold real text. Ground truth for retrieval
                    (which passage should come back) and for generation
                    correctness (what the answer should say).
  "unanswerable" -- passages.is_selected is all zeros, Eng_Answer is the
                    literal string "No Answer Present.". None of the 10
                    candidate passages actually answer the query.

Verified directly against validation/hinval.parquet (97,941 rows): a
random 500-row sample split almost exactly 50/50 between these two
buckets (252/500 unanswerable) -- consistent with the original MS MARCO
dataset MSMARCO-XI was translated from, which is well known to include a
large share of deliberately unanswerable queries. This is NOT a data
quality problem to filter away; it's used deliberately here as a built-in
negative control for the reliability/"lying factor" check (see
eval/checks/reliability.py) -- a well-behaved system should decline to
answer these, not fabricate a confident answer from irrelevant candidates.
"""
import random
from dataclasses import dataclass, field

from eval.msmarco import download_split, iter_rows


@dataclass
class EvalExample:
    query_id: int
    query_en: str
    query_hi: str
    is_answerable: bool
    gt_answer_en: str | None          # None for unanswerable rows
    gt_passage_index: int | None      # index into candidates_en/candidates_hi of the is_selected one; None if unanswerable
    candidates_en: list[str] = field(default_factory=list)
    candidates_hi: list[str] = field(default_factory=list)


_NO_ANSWER_MARKERS = {"no answer present.", ""}


def _row_to_example(row: dict) -> EvalExample | None:
    passages = row.get("passages") or {}
    selected = passages.get("is_selected") or []
    cands_en = passages.get("English_passages") or []
    cands_hi = passages.get("Translated_passages") or []
    query_en = (row.get("Eng_Query") or "").strip()
    query_hi = (row.get("query") or "").strip()
    if not query_en or not query_hi or not cands_en or not cands_hi:
        return None
    # Rows are 10 candidates in both languages, positionally aligned to the
    # same is_selected list -- verified directly against the schema.
    if len(cands_en) != len(selected) or len(cands_hi) != len(selected):
        return None

    pos_idx = next((i for i, s in enumerate(selected) if s == 1), None)
    answer_en = (row.get("Eng_Answer") or "").strip()

    if pos_idx is not None and answer_en.lower() not in _NO_ANSWER_MARKERS:
        if not cands_en[pos_idx] or not cands_hi[pos_idx]:
            return None
        return EvalExample(
            query_id=row["query_id"],
            query_en=query_en,
            query_hi=query_hi,
            is_answerable=True,
            gt_answer_en=answer_en,
            gt_passage_index=pos_idx,
            candidates_en=cands_en,
            candidates_hi=cands_hi,
        )

    if pos_idx is None and answer_en.lower() in _NO_ANSWER_MARKERS:
        return EvalExample(
            query_id=row["query_id"],
            query_en=query_en,
            query_hi=query_hi,
            is_answerable=False,
            gt_answer_en=None,
            gt_passage_index=None,
            candidates_en=cands_en,
            candidates_hi=cands_hi,
        )

    # Inconsistent row (e.g. a selected passage but "No Answer Present." text,
    # seen rarely in practice) -- skip rather than guess which signal to trust.
    return None


def load_examples(
    num_answerable: int,
    num_unanswerable: int,
    seed: int = 42,
    language: str = "hin",
    split: str = "validation",
    scan_limit: int = 20_000,
) -> list[EvalExample]:
    """Downloads (or reuses the HF cache for) one MSMARCO-XI language split,
    scans up to `scan_limit` rows (scanning the whole 97,941-row validation
    file is unnecessary for a sample of a few dozen-hundred), buckets them,
    and returns a fixed-seed random sample of each bucket, concatenated.
    Doesn't touch the target project at all -- see this module's docstring."""
    path = download_split(language, split)
    answerable, unanswerable = [], []
    for row in iter_rows(path, limit=scan_limit):
        ex = _row_to_example(row)
        if ex is None:
            continue
        (answerable if ex.is_answerable else unanswerable).append(ex)

    rng = random.Random(seed)
    if len(answerable) < num_answerable:
        raise ValueError(f"Only found {len(answerable)} answerable rows in the first {scan_limit} -- raise scan_limit.")
    if len(unanswerable) < num_unanswerable:
        raise ValueError(f"Only found {len(unanswerable)} unanswerable rows in the first {scan_limit} -- raise scan_limit.")

    sample = rng.sample(answerable, num_answerable) + rng.sample(unanswerable, num_unanswerable)
    rng.shuffle(sample)
    return sample
