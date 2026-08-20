"use client";

import type { Health } from "@/lib/types";

const STAGES = [
  { label: "Voice", detail: "MediaRecorder / Opus" },
  { label: "STT", detail: "Sarvam · ElevenLabs" },
  { label: "Chunks", detail: "5 strategies, reconciled" },
  { label: "Retrieve", detail: "dense + BM25 → RRF → MMR" },
  { label: "Generate", detail: "grounded extraction" },
  { label: "Verify", detail: "entailment + numbers" },
];

/** Header: what this is, and what the running instance actually has loaded. */
export function PipelineHeader({ health }: { health: Health | null }) {
  return (
    <header className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            Voice RAG over{" "}
            <span className="text-brand-400">MSMARCO-XI</span>
          </h1>
          <p className="text-ink-400 mt-1 max-w-2xl text-sm">
            Speak or type a question in English or Hindi. The core path — guardrails,
            hybrid retrieval, answer composition and grounding verification — is
            budgeted at 200 ms and measured on every request.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {health ? (
            <>
              <span className="tag">
                <span
                  className={`mr-1 h-1.5 w-1.5 rounded-full ${
                    health.index_loaded ? "bg-good-400" : "bg-bad-400"
                  }`}
                />
                {health.status}
              </span>
              <span className="tag">
                {health.chunks.toLocaleString()} chunks
              </span>
              <span className="tag">
                {health.passages.toLocaleString()} passages
              </span>
              <span className="tag">{health.vector_backend}</span>
              <span className="tag">
                {health.embedding_model.split("/").pop()}
              </span>
              <span className="tag">
                stt: {health.stt_providers.join(", ") || "browser only"}
              </span>
            </>
          ) : (
            <span className="tag">connecting…</span>
          )}
        </div>
      </div>

      <ol className="flex flex-wrap items-stretch gap-1.5">
        {STAGES.map((stage, index) => (
          <li
            key={stage.label}
            className="border-ink-700 bg-ink-850/70 flex-1 rounded-lg border px-3 py-2"
          >
            <div className="text-ink-500 font-mono text-[0.6rem]">
              0{index + 1}
            </div>
            <div className="text-ink-100 text-xs font-medium">{stage.label}</div>
            <div className="text-ink-500 text-[0.65rem] leading-tight">
              {stage.detail}
            </div>
          </li>
        ))}
      </ol>
    </header>
  );
}
