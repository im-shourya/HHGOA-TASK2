"use client";

import type { GuardFinding } from "@/lib/types";

const SEVERITY = {
  block: { icon: "✕", className: "text-bad-400 border-bad-500/40 bg-bad-500/10" },
  warn: { icon: "!", className: "text-warn-400 border-warn-400/40 bg-warn-400/10" },
  info: { icon: "✓", className: "text-good-400 border-good-600/40 bg-good-600/10" },
} as const;

const DESCRIPTIONS: Record<string, string> = {
  shape: "length and content sanity",
  gibberish: "keyboard-mash detection",
  safety: "harm categories (self-harm, weapons, malware, illicit, violence)",
  injection: "instruction-override attempts",
  pii: "PII redacted before logging",
  retrieval: "is there evidence worth answering from",
  numeric_grounding: "every figure appears in the cited text",
  citations: "citations resolve to retrieved chunks",
  repair: "unsupported sentences dropped",
  grounding: "sentence-level lexical entailment",
  output_pii: "no PII echoed back",
  pipeline: "unhandled failure",
};

/**
 * Which guards ran, in order, and what each decided. Showing the passes matters
 * as much as showing the block: it is the difference between "the system refused"
 * and "the system checked six things and one of them refused".
 */
export function GuardTrail({ guards }: { guards: GuardFinding[] }) {
  if (guards.length === 0) return null;
  return (
    <section className="card p-5">
      <h2 className="text-sm font-medium">
        Guardrails{" "}
        <span className="text-ink-400 font-mono text-xs">
          ({guards.filter((g) => g.passed).length}/{guards.length} passed)
        </span>
      </h2>
      <ul className="mt-3 space-y-1.5">
        {guards.map((guard, index) => {
          const severity = SEVERITY[guard.severity] ?? SEVERITY.info;
          return (
            <li
              key={`${guard.guard}-${index}`}
              className="flex items-start gap-2 text-xs"
            >
              <span
                className={`mt-px inline-flex h-4 w-4 shrink-0 items-center justify-center rounded border font-mono ${severity.className}`}
              >
                {severity.icon}
              </span>
              <span className="text-ink-200 w-36 shrink-0 font-mono">
                {guard.guard}
              </span>
              <span className="text-ink-400 flex-1">
                {guard.reason || DESCRIPTIONS[guard.guard] || "checked"}
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
