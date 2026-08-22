"use client";

import Image from "next/image";
import { useCallback, useEffect, useRef, useState } from "react";
import { AnswerCard } from "@/components/AnswerCard";
import { Composer } from "@/components/Composer";
import { GuardTrail } from "@/components/GuardTrail";
import { LatencyWaterfall } from "@/components/LatencyWaterfall";
import { MetricsBar } from "@/components/MetricsBar";
import { PipelineHeader } from "@/components/PipelineHeader";
import { SampleQueries } from "@/components/SampleQueries";
import { SourceList } from "@/components/SourceList";
import { ask, askVoice, getConfig, getHealth, getMetrics } from "@/lib/api";
import type { AskResponse, ClientConfig, Health, Metrics } from "@/lib/types";

export default function Page() {
  const [config, setConfig] = useState<ClientConfig | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [history, setHistory] = useState<AskResponse[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [language, setLanguage] = useState("eng_Latn");
  const [mode, setMode] = useState<"extractive" | "llm">("extractive");
  const inFlight = useRef<AbortController | null>(null);

  useEffect(() => {
    void getConfig().then(setConfig).catch(() => undefined);
    void getHealth().then(setHealth).catch(() => undefined);
  }, []);

  const refreshMetrics = useCallback(() => {
    void getMetrics().then(setMetrics).catch(() => undefined);
  }, []);

  const settle = useCallback(
    (result: AskResponse) => {
      setResponse(result);
      setHistory((previous) => [result, ...previous].slice(0, 8));
      refreshMetrics();
    },
    [refreshMetrics],
  );

  const run = useCallback(
    async (work: (signal: AbortSignal) => Promise<AskResponse>) => {
      inFlight.current?.abort();
      const controller = new AbortController();
      inFlight.current = controller;
      setBusy(true);
      setError(null);
      try {
        settle(await work(controller.signal));
      } catch (cause) {
        if ((cause as Error).name !== "AbortError") {
          setError(cause instanceof Error ? cause.message : String(cause));
        }
      } finally {
        if (inFlight.current === controller) inFlight.current = null;
        setBusy(false);
      }
    },
    [settle],
  );

  const submitText = useCallback(
    (query: string) =>
      void run((signal) => ask(query, { mode, language, signal })),
    [language, mode, run],
  );

  const submitAudio = useCallback(
    (blob: Blob, mimeType: string) =>
      void run((signal) =>
        askVoice(blob, {
          mode,
          language,
          signal,
          filename: mimeType.includes("mp4") ? "question.mp4" : "question.webm",
        }),
      ),
    [language, mode, run],
  );

  return (
    <main className="bg-grid min-h-screen relative overflow-x-hidden">
      {/* Decorative side palm trees */}
      <div className="fixed left-0 bottom-0 top-0 w-36 sm:w-48 md:w-60 lg:w-72 xl:w-80 pointer-events-none z-0 select-none hidden sm:block opacity-90">
        <Image
          src="/tree_left.png"
          alt=""
          aria-hidden="true"
          fill
          className="object-contain object-left-bottom"
          priority
        />
      </div>
      <div className="fixed right-0 bottom-0 top-0 w-36 sm:w-48 md:w-60 lg:w-72 xl:w-80 pointer-events-none z-0 select-none hidden sm:block opacity-90">
        <Image
          src="/tree_right.png"
          alt=""
          aria-hidden="true"
          fill
          className="object-contain object-right-bottom"
          priority
        />
      </div>

      <div className="mx-auto max-w-5xl space-y-6 px-4 py-8 sm:px-6 sm:py-12 relative z-10">
        <PipelineHeader health={health} />

        <Composer
          config={config}
          busy={busy}
          language={language}
          onLanguageChange={setLanguage}
          mode={mode}
          onModeChange={setMode}
          onSubmitText={submitText}
          onSubmitAudio={submitAudio}
        />

        {error && (
          <p className="border-bad-500/50 bg-bad-500/10 text-bad-400 rounded-lg border px-4 py-3 text-sm">
            {error}
          </p>
        )}

        {!response && <SampleQueries onPick={submitText} disabled={busy} />}

        {response && (
          <div className="space-y-4">
            <AnswerCard response={response} />
            <div className="grid gap-4 lg:grid-cols-2">
              <LatencyWaterfall trace={response.trace} />
              <GuardTrail guards={response.guards} />
            </div>
            <SourceList sources={response.sources} />
            <SampleQueries onPick={submitText} disabled={busy} />
          </div>
        )}

        <MetricsBar metrics={metrics} />

        {history.length > 1 && (
          <section className="card p-4">
            <h2 className="text-sm font-medium">Recent requests</h2>
            <ul className="mt-2 divide-y divide-ink-300 text-xs">
              {history.map((item) => (
                <li
                  key={item.request_id}
                  className="flex items-center gap-3 py-1.5"
                >
                  <button
                    type="button"
                    onClick={() => setResponse(item)}
                    className="text-ink-900 hover:text-ink-700 min-w-0 flex-1 truncate text-left"
                  >
                    {item.transcript ?? item.query}
                  </button>
                  <span className="tag">{item.verdict}</span>
                  <span className="text-ink-800 w-16 text-right font-mono">
                    {item.core_latency_ms.toFixed(1)} ms
                  </span>
                </li>
              ))}
            </ul>
          </section>
        )}

        <footer className="card p-4 sm:p-5 relative mt-8 text-xs">
          <div className="pin"></div>
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
            <p className="text-goa-dark/85 font-medium leading-relaxed max-w-2xl">
              <span className="font-bold text-goa-dark font-display uppercase tracking-wide mr-1">
                Corpus:
              </span>
              ai4bharat/MSMARCO-XI (English passages + Hindi translations).
              Answers are extracted verbatim from retrieved passages and verified
              before display; when the evidence is thin the system declines instead of
              guessing.
            </p>
            <div className="flex flex-wrap items-center gap-2 shrink-0">
              <span className="tag !bg-goa-pink !text-white !border-goa-dark shadow-[2px_2px_0px_#0a3d24]">
                #RAGInGoa
              </span>
              <span className="tag !bg-goa-yellow !text-goa-dark !border-goa-dark shadow-[2px_2px_0px_#0a3d24]">
                PII redacted
              </span>
            </div>
          </div>
        </footer>
      </div>
    </main>
  );
}
