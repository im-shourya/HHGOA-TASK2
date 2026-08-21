/**
 * Mirrors `app/schemas.py`. Kept hand-written rather than generated: the surface
 * is small, and an explicit copy makes a backend change fail at `tsc` instead of
 * silently rendering `undefined`.
 */

export type Verdict =
  | "answered"
  | "declined_unsafe"
  | "declined_injection"
  | "declined_malformed"
  | "declined_no_context"
  | "declined_ungrounded"
  | "error";

export type StageStatus = "ok" | "retried" | "degraded" | "skipped" | "failed";

export interface Span {
  name: string;
  ms: number;
  status: StageStatus;
  attempts: number;
  detail: Record<string, unknown>;
}

export interface Trace {
  spans: Span[];
  total_ms: number;
  budget_ms: number;
  within_budget: boolean;
}

export interface RetrievedChunk {
  chunk_id: string;
  parent_id: string;
  text: string;
  context_text: string;
  score: number;
  dense_score: number | null;
  sparse_score: number | null;
  dense_rank: number | null;
  sparse_rank: number | null;
  strategies: string[];
  lang: string;
  metadata: Record<string, unknown>;
}

export interface Citation {
  marker: number;
  chunk_id: string;
  parent_id: string;
  quote: string;
  strategies: string[];
  score: number;
}

export interface GuardFinding {
  guard: string;
  passed: boolean;
  severity: "info" | "warn" | "block";
  reason: string;
  detail: Record<string, unknown>;
}

export interface GroundingReport {
  score: number;
  supported_sentences: number;
  total_sentences: number;
  unsupported: string[];
  unsupported_numbers: string[];
  citations_valid: boolean;
}

export interface AskResponse {
  verdict: Verdict;
  answer: string;
  query: string;
  transcript: string | null;
  generation_mode: string;
  citations: Citation[];
  sources: RetrievedChunk[];
  grounding: GroundingReport;
  guards: GuardFinding[];
  trace: Trace;
  latency_ms: number;
  core_latency_ms: number;
  stt_latency_ms: number | null;
  detected_language: string | null;
  request_id: string;
  warnings: string[];
}

export interface ClientConfig {
  stt_providers: string[];
  voice_mode: "server" | "browser";
  llm_available: boolean;
  default_mode: string;
  budget_ms: number;
  languages: { flores: string; bcp47: string; label: string }[];
}

export interface Metrics {
  requests: number;
  core_latency_ms: {
    p50: number;
    p70: number;
    p90: number;
    p95: number;
    p100: number;
  };
  budget_ms: number;
  within_budget_pct: number;
  verdicts: Record<string, number>;
  uptime_s: number;
  warmup_ms: number;
  index: {
    chunks?: number;
    passages?: number;
    embedding_model?: string;
    vector_backend?: string;
    bm25_vocab?: number;
  };
}

export interface Health {
  status: "ok" | "degraded";
  index_loaded: boolean;
  chunks: number;
  passages: number;
  embedding_model: string;
  vector_backend: string;
  stt_providers: string[];
  llm_available: boolean;
  version: string;
}
