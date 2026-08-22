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
    <section className="card p-5 sm:p-8 relative mt-6">
      <div className="pin"></div>
      <header className="flex flex-wrap items-center gap-3">
        <span
          className={`inline-flex items-center gap-2 rounded-full border-2 px-3 py-1 text-xs font-bold font-mono uppercase shadow-[2px_2px_0px_rgba(0,0,0,0.2)] ${verdict.className}`}
        >
          <span className={`h-2 w-2 rounded-full border border-current ${verdict.dot}`} />
          {verdict.label}
        </span>
        <span className="text-goa-dark/70 font-bold font-mono text-xs">{verdict.blurb}</span>
        <span className="ml-auto flex flex-wrap items-center gap-2">
          <span className="tag">
            core{" "}
            <b
              className={
                response.trace.within_budget ? "text-goa-green" : "text-goa-pink"
              }
            >
              {ms(response.core_latency_ms)}
            </b>
          </span>
          {response.stt_latency_ms !== null && (
            <span className="tag">
              stt <b className="text-goa-pink">{ms(response.stt_latency_ms)}</b>
            </span>
          )}
          <span className="tag">{response.generation_mode}</span>
        </span>
      </header>

      {response.transcript && (
        <p className="text-goa-dark/80 mt-5 text-sm font-medium">
          <span className="text-goa-dark/50 font-mono font-bold text-xs uppercase bg-goa-yellow/20 px-1 rounded">
            heard
          </span>{" "}
          <span className="text-goa-dark italic font-mono">&ldquo;{response.transcript}&rdquo;</span>
          {response.detected_language && (
            <span className="tag ml-2">{response.detected_language}</span>
          )}
        </p>
      )}

      <p
        className={`mt-6 text-xl leading-relaxed font-sans font-bold ${
          answered ? "text-goa-dark" : "text-goa-dark/60"
        }`}
      >
        {response.answer}
      </p>

      {(() => {
        const genSpan = response.trace.spans.find((s) => s.name === "generate");
        const reasoning = genSpan?.detail?.reasoning as string | undefined;
        if (!reasoning) return null;
        return (
          <details className="mt-4 text-xs font-mono bg-goa-dark/10 p-3 rounded-lg text-goa-dark/80">
            <summary className="cursor-pointer font-bold select-none text-goa-dark">
              DeepSeek Reasoning Trace
            </summary>
            <div className="mt-2 whitespace-pre-wrap">{reasoning}</div>
          </details>
        );
      })()}

      {answered && (
        <div className="border-goa-dark/20 mt-6 flex flex-wrap items-center gap-3 border-t-2 pt-4 text-xs font-bold">
          <span className="tag">
            grounding{" "}
            <b
              className={
                grounding.score >= 0.75 ? "text-goa-green" : "text-goa-pink"
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
            <span className="tag bg-goa-pink text-white border-goa-pink shadow-none">
              stray numbers: {grounding.unsupported_numbers.join(", ")}
            </span>
          )}
        </div>
      )}

      {response.citations.length > 0 && (
        <ol className="mt-6 space-y-3">
          {response.citations.map((citation) => (
            <li
              key={`${citation.marker}-${citation.chunk_id}`}
              className="border-goa-dark/30 bg-black/5 rounded-lg border-2 p-4 text-sm shadow-inner"
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="bg-goa-yellow text-goa-dark border-goa-dark rounded border-2 px-2 py-0.5 font-mono text-xs font-bold shadow-[2px_2px_0px_#0a3d24]">
                  {citation.marker}
                </span>
                {citation.strategies.map((strategy) => (
                  <span key={strategy} className="tag !text-[0.6rem] !py-0 !shadow-none">
                    {strategy}
                  </span>
                ))}
                <span className="tag ml-auto !bg-white">
                  score {citation.score.toFixed(3)}
                </span>
              </div>
              <p className="text-goa-dark/90 mt-3 leading-relaxed font-medium">
                {citation.quote}
              </p>
            </li>
          ))}
        </ol>
      )}

      {response.warnings.length > 0 && (
        <ul className="text-goa-pink font-bold mt-5 space-y-1 text-sm font-mono">
          {response.warnings.map((warning) => (
            <li key={warning}>⚠ {warning}</li>
          ))}
        </ul>
      )}
    </section>
  );
}
