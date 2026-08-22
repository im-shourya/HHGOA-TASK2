"use client";

import type { Trace } from "@/lib/types";
import { ms, stageColor } from "@/lib/format";

/**
 * Per-stage waterfall for the request that just ran.
 *
 * Bars are scaled against the budget (200 ms) rather than against the slowest
 * stage, so the picture answers the question that matters — how much of the
 * budget did this cost — instead of flattering a fast run.
 */
export function LatencyWaterfall({ trace }: { trace: Trace }) {
  const spans = trace.spans.filter((span) => span.name !== "cache_hit");
  if (spans.length === 0) return null;

  const core = spans.filter((span) => span.name !== "transcribe");
  const coreTotal = core.reduce((sum, span) => sum + span.ms, 0);
  const scale = Math.max(trace.budget_ms, coreTotal, 1);

  return (
    <section className="card p-5">
      <header className="flex items-baseline justify-between">
        <h2 className="text-sm font-medium">Stage latency</h2>
        <span className="text-ink-700 font-mono text-xs">
          core {ms(coreTotal)} / budget {trace.budget_ms} ms
        </span>
      </header>

      <ul className="mt-4 space-y-2">
        {spans.map((span, index) => {
          const width = Math.max((span.ms / scale) * 100, 0.6);
          const outOfBudget = span.name === "transcribe";
          return (
            <li key={`${span.name}-${index}`} className="text-xs">
              <div className="flex items-center gap-2">
                <span
                  className={`w-28 shrink-0 font-mono ${
                    outOfBudget ? "text-ink-700" : "text-ink-900"
                  }`}
                >
                  {span.name}
                </span>
                <div className="bg-ink-300 relative h-3 flex-1 overflow-hidden rounded">
                  <div
                    className={`bar-grow h-full rounded ${stageColor(span.name)} ${
                      outOfBudget ? "opacity-50" : ""
                    }`}
                    style={{ width: `${Math.min(width, 100)}%` }}
                  />
                </div>
                <span className="text-ink-800 w-16 shrink-0 text-right font-mono">
                  {span.ms.toFixed(2)}
                </span>
                {span.status !== "ok" && (
                  <span
                    className={`tag ${
                      span.status === "degraded"
                        ? "text-warn-400"
                        : span.status === "failed"
                          ? "text-bad-400"
                          : ""
                    }`}
                  >
                    {span.status}
                    {span.attempts > 1 ? ` ×${span.attempts}` : ""}
                  </span>
                )}
              </div>
            </li>
          );
        })}
      </ul>

      <p className="text-ink-800 mt-4 text-xs">
        <span className="text-ink-900 font-bold">transcribe</span> is a network call to the
        speech provider and sits outside the {trace.budget_ms} ms core budget; every
        other stage is counted.
      </p>
    </section>
  );
}
