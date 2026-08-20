"use client";

import type { Metrics } from "@/lib/types";
import { ms } from "@/lib/format";

/**
 * Live percentiles from this server process — the same instrumentation the
 * benchmark reports, so the claim in the README can be checked against the
 * running instance rather than taken on trust.
 */
export function MetricsBar({ metrics }: { metrics: Metrics | null }) {
  if (!metrics || metrics.requests === 0) return null;
  const p = metrics.core_latency_ms;
  const cells: { label: string; value: string; accent?: string }[] = [
    { label: "P50", value: ms(p.p50), accent: "text-good-400" },
    { label: "P70", value: ms(p.p70), accent: "text-good-400" },
    { label: "P95", value: ms(p.p95) },
    { label: "P100", value: ms(p.p100) },
    {
      label: "within budget",
      value: `${metrics.within_budget_pct}%`,
      accent:
        metrics.within_budget_pct >= 99 ? "text-good-400" : "text-warn-400",
    },
    { label: "requests", value: String(metrics.requests) },
  ];

  return (
    <section className="card p-4">
      <header className="flex items-baseline justify-between">
        <h2 className="text-sm font-medium">Live latency (this process)</h2>
        <span className="text-ink-500 font-mono text-[0.68rem]">
          budget {metrics.budget_ms} ms · warmup {ms(metrics.warmup_ms)}
        </span>
      </header>
      <dl className="mt-3 grid grid-cols-3 gap-3 sm:grid-cols-6">
        {cells.map((cell) => (
          <div key={cell.label}>
            <dt className="text-ink-500 font-mono text-[0.68rem] uppercase">
              {cell.label}
            </dt>
            <dd
              className={`font-mono text-sm ${cell.accent ?? "text-ink-100"}`}
            >
              {cell.value}
            </dd>
          </div>
        ))}
      </dl>
      {Object.keys(metrics.verdicts).length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {Object.entries(metrics.verdicts).map(([verdict, count]) => (
            <span key={verdict} className="tag">
              {verdict} · {count}
            </span>
          ))}
        </div>
      )}
    </section>
  );
}
