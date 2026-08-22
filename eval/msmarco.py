"""Self-contained ai4bharat/MSMARCO-XI parquet loader -- owned by this eval
suite, not borrowed from whatever RAG project is under test.

This used to import training.data.download_split / iter_rows from the
target project instead of having its own copy. That was a real design
bug: the dataset this suite evaluates against is supposed to be the fixed
constant across everyone who runs it (see eval/dataset.py's module
docstring) -- what varies is each person's own RAG system. Requiring every
different target project to also happen to ship an identical
training/data.py module just to load the dataset made the suite far less
portable than it needed to be: someone with a totally different RAG
project (their own retriever, their own generator, their own API key or
local model) had no path to using this suite at all, even though nothing
about loading MSMARCO-XI actually depends on their project's code.

Row schema and download-path logic below are ported unchanged from that
original module -- confirmed correct via direct parquet inspection earlier
in this suite's development (not re-derived here), just relocated to where
it actually belongs.
"""
from collections.abc import Iterator
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "ai4bharat/MSMARCO-XI"
REPO_TYPE = "dataset"
BATCH_SIZE = 5000  # pyarrow read batch size, not an eval batch size


def download_split(language: str, split: str) -> Path:
    """language: e.g. 'hin'. split: 'train' or 'validation'."""
    suffix = "train" if split == "train" else "val"
    filename = f"{split}/{language}{suffix}.parquet"
    return Path(hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=filename))


def iter_rows(path: Path, limit: int | None = None) -> Iterator[dict]:
    """Yields raw parquet rows as dicts. Reads from local disk -- download
    the file first via download_split(); remote range-reads are impractical
    for this dataset (each language file is a single giant row group)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    count = 0
    for record_batch in pf.iter_batches(batch_size=BATCH_SIZE):
        for row in record_batch.to_pylist():
            yield row
            count += 1
            if limit is not None and count >= limit:
                return
