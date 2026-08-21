/**
 * Typed API client.
 *
 * `API_BASE` is empty in production because FastAPI serves this bundle from the
 * same origin; in development it points at the Python process on :8000.
 */

import type {
  AskResponse,
  ClientConfig,
  Health,
  Metrics,
} from "@/lib/types";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      /* non-JSON error body: keep the status line */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export interface AskOptions {
  mode?: "extractive" | "llm" | "auto";
  topK?: number;
  language?: string;
  signal?: AbortSignal;
}

export async function ask(
  query: string,
  options: AskOptions = {},
): Promise<AskResponse> {
  const response = await fetch(`${API_BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      mode: options.mode ?? null,
      top_k: options.topK ?? null,
      language: options.language ?? null,
      include_trace: true,
    }),
    signal: options.signal,
  });
  return parse<AskResponse>(response);
}

export async function askVoice(
  audio: Blob,
  options: AskOptions & { provider?: string; filename?: string } = {},
): Promise<AskResponse> {
  const form = new FormData();
  form.append("audio", audio, options.filename ?? "question.webm");
  if (options.language) form.append("language", options.language);
  if (options.mode) form.append("mode", options.mode);
  if (options.provider) form.append("provider", options.provider);
  form.append("include_trace", "true");
  const response = await fetch(`${API_BASE}/api/voice`, {
    method: "POST",
    body: form,
    signal: options.signal,
  });
  return parse<AskResponse>(response);
}

export const getConfig = () =>
  fetch(`${API_BASE}/api/config`).then(parse<ClientConfig>);

export const getMetrics = () =>
  fetch(`${API_BASE}/api/metrics`).then(parse<Metrics>);

export const getHealth = () =>
  fetch(`${API_BASE}/api/health`).then(parse<Health>);
