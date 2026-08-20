"use client";

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
    <main className="bg-grid min-h-screen">
      <div className="mx-auto max-w-5xl space-y-6 px-4 py-8 sm:px-6 sm:py-12">
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
            <ul className="mt-2 divide-y divide-ink-800 text-xs">
              {history.map((item) => (
                <li
                  key={item.request_id}
                  className="flex items-center gap-3 py-1.5"
                >
                  <button
                    type="button"
                    onClick={() => setResponse(item)}
                    className="text-ink-300 hover:text-ink-100 min-w-0 flex-1 truncate text-left"
                  >
                    {item.transcript ?? item.query}
                  </button>
                  <span className="tag">{item.verdict}</span>
                  <span className="text-ink-400 w-16 text-right font-mono">
                    {item.core_latency_ms.toFixed(1)} ms
                  </span>
                </li>
              ))}
            </ul>
          </section>
        )}

        <footer className="text-ink-500 border-ink-800 border-t pt-5 text-xs">
          <p>
            Corpus: ai4bharat/MSMARCO-XI (English passages + Hindi translations).
            Answers are extracted verbatim from retrieved passages and verified
            before display; when the evidence is thin the system declines instead of
            guessing.
          </p>
          <p className="mt-1 font-mono">
            #RAGInGoa · request ids are logged with PII redacted
          </p>
        </footer>
      </div>
    </main>
  );
}
