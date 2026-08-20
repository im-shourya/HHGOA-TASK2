"""Measure the *served* path: real HTTP requests against a running server.

Why this exists alongside `app/benchmark.py`. That harness measures the core path
in-process — it constructs `RagPipeline` and awaits it directly. Those are the
numbers in the README, and they are the right numbers for "how fast is the
pipeline". They are not the numbers a visitor to the deployed URL experiences,
because they exclude uvicorn's accept loop, ASGI dispatch, pydantic response
serialisation and the socket write.

This script measures what a grader clicking the live link actually gets, and it
reports the two separately rather than blending them:

    core_latency_ms   what the pipeline spent (server-measured, same field the
                      in-process harness reports) — comparable to the README table
    wall_ms           what the client waited, including HTTP framing

The gap between them is the cost of being a web service, and it should be small.
If it is not, the README's core-path figures would be technically true and
practically misleading, which is the failure mode this script is here to catch.

One measurement trap this script avoids. A naive client (`urllib.request`) opens a
fresh TCP connection per request, so its wall time includes a connect handshake
that no browser and no `fetch()` loop pays after the first call. Measured here that
artifact was ~21 ms — larger than the pipeline itself, and reporting it as "HTTP
overhead" would have been a straightforward lie about the service. The default is
therefore a single kept-alive connection; `--no-keepalive` reproduces the naive
number so the difference is visible rather than asserted.

Usage:
    python scripts/bench_http.py --url http://127.0.0.1:8000 --n 300
    python scripts/bench_http.py --no-keepalive          # per-request connect cost

Requests are issued sequentially on purpose. Concurrency measures queueing on a
single-worker asyncio server, which is a different question from per-request
latency, and mixing the two produces a P100 that describes contention.
"""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402

PERCENTILES = (50, 70, 90, 95, 99, 100)


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100)
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (k - lo) * (ordered[hi] - ordered[lo])


def load_queries(limit: int) -> tuple[list[str], str]:
    """Corpus queries when available, else a small built-in list."""
    fallback = [
        "what is a corporation",
        "how does photosynthesis work",
        "what causes inflation",
        "who invented the telephone",
        "what is the capital of australia",
    ]
    path = get_settings().corpus_dir / "eval_queries.jsonl"
    if not path.exists():
        return fallback, f"built-in list ({path} not found)"
    queries: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("eng_query", "query"):
                value = (row.get(key) or "").strip()
                if 0 < len(value) <= 400:
                    queries.append(value)
    if not queries:
        return fallback, "built-in list (corpus held no usable queries)"
    if limit and len(queries) > limit:
        queries = queries[:limit]
    return queries, f"{path.name} ({len(queries)} queries from MSMARCO-XI)"


class Client:
    """Issues /api/ask requests, optionally over one kept-alive connection.

    Keep-alive is the default because it is what a browser does. The fresh-connection
    mode is kept for comparison, not as a fallback.
    """

    def __init__(self, base: str, timeout: float, keepalive: bool = True) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.keepalive = keepalive
        parts = urllib.parse.urlsplit(self.base)
        self._host = parts.hostname or "127.0.0.1"
        self._port = parts.port or (443 if parts.scheme == "https" else 80)
        self._https = parts.scheme == "https"
        self._path = (parts.path or "") + "/api/ask"
        self._conn: http.client.HTTPConnection | None = None

    def _connection(self) -> http.client.HTTPConnection:
        if self._conn is None:
            factory = (
                http.client.HTTPSConnection if self._https else http.client.HTTPConnection
            )
            self._conn = factory(self._host, self._port, timeout=self.timeout)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def ask(self, query: str) -> tuple[float, float, str]:
        """One request. Returns (wall_ms, core_ms, verdict)."""
        body = json.dumps({"query": query, "include_trace": False}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        start = time.perf_counter()
        if self.keepalive:
            conn = self._connection()
            try:
                conn.request("POST", self._path, body=body, headers=headers)
                raw = conn.getresponse().read()
            except (http.client.HTTPException, OSError):
                # A dropped keep-alive is not a benchmark result; reconnect once.
                self.close()
                conn = self._connection()
                conn.request("POST", self._path, body=body, headers=headers)
                raw = conn.getresponse().read()
            payload = json.loads(raw)
        else:
            request = urllib.request.Request(
                f"{self.base}/api/ask", data=body, headers=headers
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        wall_ms = (time.perf_counter() - start) * 1000
        return wall_ms, payload["core_latency_ms"], payload["verdict"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=300, help="requests to issue")
    parser.add_argument("--warmup", type=int, default=10,
                        help="requests to discard before measuring")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--no-keepalive", action="store_true",
                        help="open a fresh TCP connection per request (shows connect cost)")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    queries, source = load_queries(limit=0)
    transport = "fresh connection per request" if args.no_keepalive else "one kept-alive connection"
    print(f"target : {args.url}")
    print(f"queries: {source}")
    print(f"client : {transport}")

    client = Client(args.url, args.timeout, keepalive=not args.no_keepalive)
    try:
        client.ask("warmup probe")
    except (urllib.error.URLError, OSError) as exc:
        print(f"\ncannot reach {args.url}: {exc}\n"
              f"start it first:  uvicorn app.main:app --port 8000", file=sys.stderr)
        return 2

    for i in range(args.warmup):
        client.ask(queries[i % len(queries)])

    wall: list[float] = []
    core: list[float] = []
    verdicts: dict[str, int] = {}
    failures = 0
    worst = (0.0, "")
    for i in range(args.n):
        query = queries[i % len(queries)]
        try:
            wall_ms, core_ms, verdict = client.ask(query)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
            failures += 1
            print(f"  request {i} failed: {exc}", file=sys.stderr)
            continue
        wall.append(wall_ms)
        core.append(core_ms)
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        if wall_ms > worst[0]:
            worst = (wall_ms, query)
    client.close()

    if not wall:
        print("no successful requests", file=sys.stderr)
        return 1

    budget = get_settings().core_budget_ms
    header = f"{'metric':<10}{'mean':>8}" + "".join(f"{'p' + str(p):>8}" for p in PERCENTILES)
    print(f"\n{len(wall)} requests over HTTP ({failures} failed)\n")
    print(f"{header}   (ms)")
    for name, values in (("core", core), ("wall", wall)):
        line = f"{name:<10}{statistics.mean(values):>8.2f}"
        line += "".join(f"{percentile(values, p):>8.2f}" for p in PERCENTILES)
        print(line)

    overhead = statistics.mean(w - c for w, c in zip(wall, core))
    within = 100 * sum(1 for v in core if v <= budget) / len(core)
    within_wall = 100 * sum(1 for v in wall if v <= budget) / len(wall)
    print(f"\nHTTP overhead (wall - core): mean {overhead:.2f} ms")
    print(f"Core budget {budget} ms | core within {within:.1f}% | wall within {within_wall:.1f}%")
    print(f"slowest wall: {worst[0]:.2f} ms")
    print("verdicts: " + ", ".join(f"{k}={v}" for k, v in sorted(verdicts.items())))

    # The tail is the point of running this many: report how heavy it is instead of
    # letting a single P100 stand in for it.
    tail = sorted(wall, reverse=True)[:10]
    print("10 slowest wall (ms): " + ", ".join(f"{v:.1f}" for v in tail))
    over_20 = sum(1 for v in core if v > 20)
    print(f"core > 20 ms: {over_20} of {len(core)} ({100 * over_20 / len(core):.2f}%)")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "url": args.url,
            "transport": transport,
            "requests": len(wall),
            "failures": failures,
            "budget_ms": budget,
            "core_ms": {f"p{p}": round(percentile(core, p), 3) for p in PERCENTILES},
            "wall_ms": {f"p{p}": round(percentile(wall, p), 3) for p in PERCENTILES},
            "core_mean_ms": round(statistics.mean(core), 3),
            "wall_mean_ms": round(statistics.mean(wall), 3),
            "http_overhead_mean_ms": round(overhead, 3),
            "core_within_budget_pct": round(within, 2),
            "verdicts": verdicts,
            "slowest_wall_ms": round(worst[0], 3),
            "core_over_20ms": over_20,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    return 0 if (failures == 0 and within == 100.0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
