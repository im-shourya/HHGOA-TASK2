"use client";

import type { AskResponse } from "@/lib/types";
import { VERDICTS, ms, pct } from "@/lib/format";

/**
 * The answer, plus the two numbers that decide whether to trust it: grounding
 * score and core latency. Citations are rendered as the quotes they came from,
 * so a claim can be checked without opening the sources panel.
 */
export function AnswerCard({ response }: { response: AskResponse }) {
  const verdict = VERDICTS[response.verdict];
  const grounding = response.grounding;
  const answered = response.verdict === "answered";

  return (
    <section className="card p-5 sm:p-6">
      <header className="flex flex-wrap items-center gap-3">
        <span
          className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium ${verdict.className}`}
        >
          <span className={`h-1.5 w-1.5 rounded-full ${verdict.dot}`} />
          {verdict.label}
        </span>
        <span className="text-ink-400 text-xs">{verdict.blurb}</span>
        <span className="ml-auto flex items-center gap-2">
          <span className="tag">
            core{" "}
            <b
              className={
                response.trace.within_budget ? "text-good-400" : "text-bad-400"
              }
            >
              {ms(response.core_latency_ms)}
            </b>
          </span>
          {response.stt_latency_ms !== null && (
            <span className="tag">
              stt <b className="text-fuchsia-300">{ms(response.stt_latency_ms)}</b>
            </span>
          )}
          <span className="tag">{response.generation_mode}</span>
        </span>
      </header>

      {response.transcript && (
        <p className="text-ink-300 mt-4 text-sm">
          <span className="text-ink-500 font-mono text-xs uppercase">
            heard
          </span>{" "}
          <span className="text-ink-100">&ldquo;{response.transcript}&rdquo;</span>
          {response.detected_language && (
            <span className="tag ml-2">{response.detected_language}</span>
          )}
        </p>
      )}

      <p
        className={`mt-4 text-lg leading-relaxed ${
          answered ? "text-ink-100" : "text-ink-200"
        }`}
      >
        {response.answer}
      </p>

      {answered && (
        <div className="border-ink-700 mt-5 flex flex-wrap items-center gap-3 border-t pt-4 text-xs">
          <span className="tag">
            grounding{" "}
            <b
              className={
                grounding.score >= 0.75 ? "text-good-400" : "text-warn-400"
              }
            >
              {pct(grounding.score)}
            </b>
          </span>
          <span className="tag">
            {grounding.supported_sentences}/{grounding.total_sentences} sentences
            supported
          </span>
          <span className="tag">
            citations {grounding.citations_valid ? "valid" : "invalid"}
          </span>
          {grounding.unsupported_numbers.length > 0 && (
            <span className="tag text-bad-400">
              stray numbers: {grounding.unsupported_numbers.join(", ")}
            </span>
          )}
        </div>
      )}

      {response.citations.length > 0 && (
        <ol className="mt-4 space-y-2">
          {response.citations.map((citation) => (
            <li
              key={`${citation.marker}-${citation.chunk_id}`}
              className="border-ink-700 bg-ink-900/60 rounded-lg border p-3 text-sm"
            >
              <div className="flex items-center gap-2">
                <span className="bg-brand-600/20 text-brand-400 border-brand-600/40 rounded border px-1.5 font-mono text-xs">
                  {citation.marker}
                </span>
                {citation.strategies.map((strategy) => (
                  <span key={strategy} className="tag">
                    {strategy}
                  </span>
                ))}
                <span className="tag ml-auto">
                  score {citation.score.toFixed(3)}
                </span>
              </div>
              <p className="text-ink-300 mt-2 leading-relaxed">
                {citation.quote}
              </p>
            </li>
          ))}
        </ol>
      )}

      {response.warnings.length > 0 && (
        <ul className="text-warn-400 mt-4 space-y-1 text-xs">
          {response.warnings.map((warning) => (
            <li key={warning}>⚠ {warning}</li>
          ))}
        </ul>
      )}
    </section>
  );
}
