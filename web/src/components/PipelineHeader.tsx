"use client";

import type { Health } from "@/lib/types";
import Image from "next/image";

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
    <header className="space-y-6">
      <div className="flex flex-col md:flex-row items-center md:items-end justify-between gap-6 pb-4 border-b-2 border-goa-dark/20">
        <div className="flex flex-col sm:flex-row items-center gap-6">
          <Image
            src="/logo.png"
            alt="3PIXLE Team Logo"
            width={150}
            height={100}
            className="h-20 sm:h-24 w-auto rounded-xl border-2 border-goa-dark shadow-[4px_4px_0px_#ffe500] object-contain bg-white p-1"
            priority
          />
          <div className="text-center sm:text-left flex flex-col items-center sm:items-start">
            <h1 className="relative inline-flex items-center justify-center select-none mb-1">
              <span className="sr-only">Hacker House Goa</span>
              <Image
                src="/Hacker_house.png"
                alt="Hacker House"
                width={1148}
                height={237}
                className="h-10 sm:h-14 md:h-16 w-auto object-contain"
                priority
              />
              <Image
                src="/goa_hindi.svg"
                alt="Goa"
                width={181}
                height={180}
                className="absolute h-12 sm:h-[72px] md:h-20 w-auto object-contain left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
                priority
              />
            </h1>
            <p className="text-goa-cream mt-2 max-w-xl text-lg font-mono">
              Voice RAG over MSMARCO-XI.
            </p>
          </div>
        </div>
        <div className="flex flex-wrap justify-center md:justify-end items-center gap-2">
          {health ? (
            <>
              <span className="tag">
                <span
                  className={`mr-1.5 h-2 w-2 rounded-full border border-goa-dark ${
                    health.index_loaded ? "bg-goa-green" : "bg-goa-pink"
                  }`}
                />
                {health.status}
              </span>
              <span className="tag">{health.chunks.toLocaleString()} chunks</span>
              <span className="tag">{health.vector_backend}</span>
              <span className="tag">
                stt: {health.stt_providers.join(", ") || "browser only"}
              </span>
            </>
          ) : (
            <span className="tag">connecting…</span>
          )}
        </div>
      </div>

      <ol className="flex flex-wrap justify-center sm:justify-start items-stretch gap-2">
        {STAGES.map((stage, index) => (
          <li
            key={stage.label}
            className="card flex-1 min-w-[120px] p-3 text-center sm:text-left relative"
          >
            <div className="pin hidden sm:block"></div>
            <div className="text-goa-pink font-mono text-[0.75rem] font-bold">
              0{index + 1}
            </div>
            <div className="text-goa-dark text-sm font-bold uppercase font-display tracking-wide">{stage.label}</div>
            <div className="text-goa-dark/70 text-[0.65rem] font-mono leading-tight mt-1">
              {stage.detail}
            </div>
          </li>
        ))}
      </ol>
    </header>
  );
}
