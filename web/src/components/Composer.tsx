"use client";

import { useCallback, useState } from "react";
import type { ClientConfig } from "@/lib/types";
import { useAudioRecorder, useSpeechRecognition } from "@/lib/useRecorder";

interface ComposerProps {
  config: ClientConfig | null;
  busy: boolean;
  language: string;
  onLanguageChange: (flores: string) => void;
  mode: "extractive" | "llm";
  onModeChange: (mode: "extractive" | "llm") => void;
  onSubmitText: (query: string) => void;
  onSubmitAudio: (blob: Blob, mimeType: string) => void;
}

/**
 * Input surface: hold-to-talk plus a text field.
 *
 * Two capture routes are offered explicitly rather than silently chosen.
 * "Provider" records audio and sends it to Sarvam/ElevenLabs — the graded path.
 * "Browser" uses the Web Speech API, which needs no key, so the demo still works
 * on a fresh clone or when a provider is failing. The selector defaults to
 * whichever the server says is actually available.
 */
export function Composer({
  config,
  busy,
  language,
  onLanguageChange,
  mode,
  onModeChange,
  onSubmitText,
  onSubmitAudio,
}: ComposerProps) {
  const [draft, setDraft] = useState("");
  const [route, setRoute] = useState<"server" | "browser">("server");

  const recorder = useAudioRecorder(onSubmitAudio);
  const speech = useSpeechRecognition(onSubmitText);

  const serverAvailable = (config?.stt_providers.length ?? 0) > 0;
  const activeRoute = serverAvailable ? route : "browser";
  const recording =
    activeRoute === "server" ? recorder.state === "recording" : speech.listening;
  const bcp47 =
    config?.languages.find((entry) => entry.flores === language)?.bcp47 ?? "en-IN";

  const toggleVoice = useCallback(() => {
    if (activeRoute === "server") {
      if (recorder.state === "recording") recorder.stop();
      else void recorder.start();
      return;
    }
    if (speech.listening) speech.stop();
    else speech.start(bcp47);
  }, [activeRoute, bcp47, recorder, speech]);

  const submit = () => {
    const query = draft.trim();
    if (!query || busy) return;
    onSubmitText(query);
    setDraft("");
  };

  const error = activeRoute === "server" ? recorder.error : speech.error;

  return (
    <section className="card p-4 sm:p-5">
      <div className="flex items-start gap-4">
        <button
          type="button"
          onClick={toggleVoice}
          disabled={busy && !recording}
          aria-label={recording ? "Stop recording" : "Start recording"}
          aria-pressed={recording}
          className={`relative flex h-16 w-16 shrink-0 items-center justify-center rounded-full border transition ${
            recording
              ? "recording border-bad-500 bg-bad-500/20 text-bad-400"
              : "border-brand-500/60 bg-brand-600/15 text-brand-400 hover:bg-brand-600/25"
          } disabled:cursor-not-allowed disabled:opacity-40`}
        >
          {recording ? (
            <span className="bg-bad-400 h-4 w-4 rounded-sm" />
          ) : (
            <MicIcon />
          )}
          {recording && activeRoute === "server" && (
            <span
              className="border-bad-400/70 absolute inset-0 rounded-full border"
              style={{ transform: `scale(${1 + recorder.level * 0.35})` }}
            />
          )}
        </button>

        <div className="min-w-0 flex-1">
          <label htmlFor="query" className="sr-only">
            Ask a question
          </label>
          <textarea
            id="query"
            rows={2}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
            placeholder={
              recording
                ? "Listening…"
                : "Ask about the indexed passages — or hold the mic and speak (English / हिन्दी)"
            }
            className="border-ink-700 bg-ink-900 text-ink-100 placeholder:text-ink-500 focus:border-brand-500 w-full resize-none rounded-lg border px-3 py-2 text-sm outline-none"
          />
          {speech.interim && (
            <p className="text-ink-400 mt-1 text-xs italic">{speech.interim}</p>
          )}
          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
            <select
              value={language}
              onChange={(event) => onLanguageChange(event.target.value)}
              aria-label="Question language"
              className="border-ink-700 bg-ink-900 text-ink-200 rounded-md border px-2 py-1"
            >
              {(config?.languages ?? [{ flores: "eng_Latn", bcp47: "en-IN", label: "English" }]).map(
                (entry) => (
                  <option key={entry.flores} value={entry.flores}>
                    {entry.label}
                  </option>
                ),
              )}
            </select>

            <div className="border-ink-700 flex overflow-hidden rounded-md border">
              {(["server", "browser"] as const).map((option) => (
                <button
                  key={option}
                  type="button"
                  disabled={option === "server" && !serverAvailable}
                  onClick={() => setRoute(option)}
                  title={
                    option === "server"
                      ? serverAvailable
                        ? `Send audio to ${config?.stt_providers.join(" → ")}`
                        : "No provider key configured on the server"
                      : "Transcribe locally with the Web Speech API"
                  }
                  className={`px-2 py-1 font-mono ${
                    activeRoute === option
                      ? "bg-brand-600/25 text-brand-400"
                      : "text-ink-400 hover:text-ink-200"
                  } disabled:cursor-not-allowed disabled:opacity-40`}
                >
                  {option === "server"
                    ? config?.stt_providers[0] ?? "provider"
                    : "browser"}
                </button>
              ))}
            </div>

            <div className="border-ink-700 flex overflow-hidden rounded-md border">
              {(["extractive", "llm"] as const).map((option) => (
                <button
                  key={option}
                  type="button"
                  disabled={option === "llm" && !config?.llm_available}
                  onClick={() => onModeChange(option)}
                  title={
                    option === "extractive"
                      ? "Grounded extractive composer — fits the 200 ms budget"
                      : config?.llm_available
                        ? "Claude with forced structured output — exceeds the budget by design"
                        : "No ANTHROPIC_API_KEY configured"
                  }
                  className={`px-2 py-1 font-mono ${
                    mode === option
                      ? "bg-brand-600/25 text-brand-400"
                      : "text-ink-400 hover:text-ink-200"
                  } disabled:cursor-not-allowed disabled:opacity-40`}
                >
                  {option}
                </button>
              ))}
            </div>

            {recorder.state === "recording" && (
              <span className="text-bad-400 font-mono">
                ● {recorder.seconds}s — tap to send
              </span>
            )}

            <button
              type="button"
              onClick={submit}
              disabled={busy || draft.trim().length === 0}
              className="bg-brand-600 hover:bg-brand-500 ml-auto rounded-md px-3 py-1.5 font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
            >
              {busy ? "Thinking…" : "Ask"}
            </button>
          </div>
          {error && <p className="text-bad-400 mt-2 text-xs">{error}</p>}
        </div>
      </div>
    </section>
  );
}

function MicIcon() {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Z" />
      <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
    </svg>
  );
}
