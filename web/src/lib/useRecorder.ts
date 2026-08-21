"use client";

/**
 * Microphone capture, two ways.
 *
 * `useAudioRecorder` records Opus/WebM (or MP4 on Safari) and hands the blob to
 * `/api/voice`, where Sarvam or ElevenLabs transcribes it — the path the task
 * requires. It also drives a live level meter, because a mic button with no
 * feedback leaves the user guessing whether anything was heard.
 *
 * `useSpeechRecognition` uses the browser's own Web Speech API and posts text to
 * `/api/ask`. That path needs no API key at all, so the demo stays usable before
 * keys are configured and when a provider is down.
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";

export type RecorderState = "idle" | "requesting" | "recording" | "stopping";

/* Capability probes never change after load, so there is nothing to subscribe to. */
const NEVER_CHANGES = () => () => {};

/**
 * Read a browser capability safely.
 *
 * The page is a static export, so the first render happens in Node at build time
 * where `MediaRecorder` and `SpeechRecognition` do not exist. Probing during render
 * would make the prerendered HTML disagree with the hydrated tree; probing in an
 * effect and calling `setState` cascades an extra render (and is what
 * `react-hooks/set-state-in-effect` objects to). `useSyncExternalStore` is the
 * primitive built for exactly this: `false` on the server, the real answer on the
 * client, and React reconciles the switch itself.
 */
function useCapability(probe: () => boolean): boolean {
  return useSyncExternalStore(NEVER_CHANGES, probe, () => false);
}

const hasMediaRecorder = () => typeof MediaRecorder !== "undefined";
const hasSpeechRecognition = () => speechRecognitionCtor() !== null;

/* ---- minimal Web Speech typings (not in lib.dom for every TS version) ---- */
interface SpeechRecognitionAlternativeLike {
  transcript: string;
  confidence: number;
}
interface SpeechRecognitionResultLike {
  isFinal: boolean;
  0: SpeechRecognitionAlternativeLike;
  length: number;
}
interface SpeechRecognitionEventLike extends Event {
  resultIndex: number;
  results: {
    length: number;
    [index: number]: SpeechRecognitionResultLike;
  };
}
interface SpeechRecognitionLike extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: Event & { error?: string }) => void) | null;
  onend: (() => void) | null;
}
type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

function speechRecognitionCtor(): SpeechRecognitionCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

function pickMimeType(): string {
  if (typeof MediaRecorder === "undefined") return "";
  for (const candidate of [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
  ]) {
    if (MediaRecorder.isTypeSupported(candidate)) return candidate;
  }
  return "";
}

export interface AudioRecorder {
  state: RecorderState;
  level: number;
  seconds: number;
  error: string | null;
  supported: boolean;
  start: () => Promise<void>;
  stop: () => void;
  cancel: () => void;
}

export function useAudioRecorder(
  onComplete: (blob: Blob, mimeType: string) => void,
): AudioRecorder {
  const [state, setState] = useState<RecorderState>("idle");
  const [level, setLevel] = useState(0);
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const supported = useCapability(hasMediaRecorder);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const frameRef = useRef<number | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const discardRef = useRef(false);

  const teardown = useCallback(() => {
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    if (tickRef.current !== null) clearInterval(tickRef.current);
    frameRef.current = null;
    tickRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    void audioContextRef.current?.close().catch(() => undefined);
    audioContextRef.current = null;
    recorderRef.current = null;
    setLevel(0);
  }, []);

  useEffect(() => teardown, [teardown]);

  const start = useCallback(async () => {
    setError(null);
    if (typeof navigator === "undefined" || !navigator.mediaDevices) {
      setError("This browser has no microphone API.");
      return;
    }
    setState("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      const mimeType = pickMimeType();
      const recorder = new MediaRecorder(
        stream,
        mimeType ? { mimeType, audioBitsPerSecond: 32_000 } : undefined,
      );
      chunksRef.current = [];
      discardRef.current = false;
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        const type = recorder.mimeType || mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, { type });
        teardown();
        setState("idle");
        setSeconds(0);
        if (!discardRef.current && blob.size > 0) onComplete(blob, type);
      };
      recorder.start(250);
      recorderRef.current = recorder;
      setState("recording");

      // Level meter: RMS of the time-domain buffer, smoothed for readability.
      const context = new AudioContext();
      audioContextRef.current = context;
      const analyser = context.createAnalyser();
      analyser.fftSize = 512;
      context.createMediaStreamSource(stream).connect(analyser);
      const buffer = new Float32Array(analyser.fftSize);
      const sample = () => {
        analyser.getFloatTimeDomainData(buffer);
        let sum = 0;
        for (const value of buffer) sum += value * value;
        const rms = Math.sqrt(sum / buffer.length);
        setLevel((previous) => previous * 0.6 + Math.min(rms * 4, 1) * 0.4);
        frameRef.current = requestAnimationFrame(sample);
      };
      frameRef.current = requestAnimationFrame(sample);

      setSeconds(0);
      tickRef.current = setInterval(() => setSeconds((s) => s + 1), 1000);
    } catch (cause) {
      teardown();
      setState("idle");
      const message =
        cause instanceof DOMException && cause.name === "NotAllowedError"
          ? "Microphone permission denied — allow it in the browser, or type instead."
          : `Could not start recording: ${String(cause)}`;
      setError(message);
    }
  }, [onComplete, teardown]);

  const stop = useCallback(() => {
    if (recorderRef.current?.state === "recording") {
      setState("stopping");
      recorderRef.current.stop();
    }
  }, []);

  const cancel = useCallback(() => {
    discardRef.current = true;
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
    else {
      teardown();
      setState("idle");
    }
  }, [teardown]);

  return {
    state,
    level,
    seconds,
    error,
    supported,
    start,
    stop,
    cancel,
  };
}

export interface SpeechCapture {
  listening: boolean;
  interim: string;
  error: string | null;
  supported: boolean;
  start: (language: string) => void;
  stop: () => void;
}

export function useSpeechRecognition(
  onFinal: (transcript: string) => void,
): SpeechCapture {
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const [error, setError] = useState<string | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const supported = useCapability(hasSpeechRecognition);

  // Abort any in-flight recognition when the component goes away.
  useEffect(() => () => recognitionRef.current?.abort(), []);
  const start = useCallback(
    (language: string) => {
      const Ctor = speechRecognitionCtor();
      if (!Ctor) {
        setError("This browser has no Web Speech API. Use text, or add an STT key.");
        return;
      }
      setError(null);
      setInterim("");
      const recognition = new Ctor();
      recognition.lang = language;
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;
      recognition.onresult = (event) => {
        let draft = "";
        for (let i = event.resultIndex; i < event.results.length; i += 1) {
          const result = event.results[i];
          if (result.isFinal) {
            const text = result[0].transcript.trim();
            setInterim("");
            setListening(false);
            recognition.stop();
            if (text) onFinal(text);
            return;
          }
          draft += result[0].transcript;
        }
        setInterim(draft);
      };
      recognition.onerror = (event) => {
        const code = event.error ?? "unknown";
        setError(
          code === "not-allowed"
            ? "Microphone permission denied."
            : `Speech recognition failed (${code}).`,
        );
        setListening(false);
      };
      recognition.onend = () => setListening(false);
      recognitionRef.current = recognition;
      recognition.start();
      setListening(true);
    },
    [onFinal],
  );

  const stop = useCallback(() => {
    recognitionRef.current?.stop();
    setListening(false);
  }, []);

  return { listening, interim, error, supported, start, stop };
}
