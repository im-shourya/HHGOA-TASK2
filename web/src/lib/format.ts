import type { Verdict } from "@/lib/types";

export const ms = (value: number): string =>
  value >= 100 ? `${Math.round(value)} ms` : `${value.toFixed(1)} ms`;

export const pct = (value: number): string => `${Math.round(value * 100)}%`;

export interface VerdictStyle {
  label: string;
  blurb: string;
  className: string;
  dot: string;
}

/**
 * A refusal is a result, not an error — so each verdict gets its own colour and
 * a one-line explanation of *why* the system stopped.
 */
export const VERDICTS: Record<Verdict, VerdictStyle> = {
  answered: {
    label: "Answered",
    blurb: "Every sentence was verified against the retrieved passages.",
    className: "border-good-600/50 bg-good-600/10 text-good-400",
    dot: "bg-good-400",
  },
  declined_no_context: {
    label: "Declined · no evidence",
    blurb: "Retrieval confidence fell below the calibrated floor.",
    className: "border-warn-400/40 bg-warn-400/10 text-warn-400",
    dot: "bg-warn-400",
  },
  declined_ungrounded: {
    label: "Declined · not grounded",
    blurb: "A draft existed but the passages did not support it.",
    className: "border-warn-400/40 bg-warn-400/10 text-warn-400",
    dot: "bg-warn-400",
  },
  declined_unsafe: {
    label: "Blocked · safety",
    blurb: "The input guardrail matched a harm category.",
    className: "border-bad-500/50 bg-bad-500/10 text-bad-400",
    dot: "bg-bad-400",
  },
  declined_injection: {
    label: "Blocked · prompt injection",
    blurb: "The input tried to override the system instructions.",
    className: "border-bad-500/50 bg-bad-500/10 text-bad-400",
    dot: "bg-bad-400",
  },
  declined_malformed: {
    label: "Declined · unreadable",
    blurb: "No usable question was found in the input.",
    className: "border-ink-600 bg-ink-800 text-ink-300",
    dot: "bg-ink-400",
  },
  error: {
    label: "Error",
    blurb: "The pipeline failed; the trace shows where.",
    className: "border-bad-500/50 bg-bad-500/10 text-bad-400",
    dot: "bg-bad-400",
  },
};

/** Colour per pipeline stage, shared by the waterfall and the stage legend. */
export const STAGE_COLORS: Record<string, string> = {
  transcribe: "bg-fuchsia-400",
  guard_input: "bg-bad-400",
  classify: "bg-amber-300",
  embed_query: "bg-brand-400",
  retrieve: "bg-cyan-300",
  retrieval_guard: "bg-teal-300",
  generate: "bg-good-400",
  verify: "bg-violet-300",
  cache_hit: "bg-ink-400",
};

export const stageColor = (name: string): string =>
  STAGE_COLORS[name] ?? "bg-ink-400";

export const STRATEGY_LABELS: Record<string, string> = {
  passage: "whole passage",
  fixed_window: "fixed 90w/24 overlap",
  sentence_window: "3-sentence window",
  recursive_char: "recursive 420ch",
  semantic: "semantic breakpoints",
};
