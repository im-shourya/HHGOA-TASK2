"use client";

import { useState } from "react";
import type { RetrievedChunk } from "@/lib/types";
import { STRATEGY_LABELS } from "@/lib/format";

/**
 * The retrieved evidence, with the provenance that explains *why* each chunk is
 * here: which chunking strategy produced it, and where dense and sparse retrieval
 * each ranked it. A chunk found by both is a different kind of hit from one found
 * only by BM25, and the badges make that visible.
 */
export function SourceList({ sources }: { sources: RetrievedChunk[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  if (sources.length === 0) return null;

  return (
    <section className="card p-5">
      <h2 className="text-sm font-medium">
        Retrieved context{" "}
        <span className="text-ink-400 font-mono text-xs">
          ({sources.length} chunks, one per passage)
        </span>
      </h2>
      <ul className="mt-3 space-y-2">
        {sources.map((source, index) => {
          const open = expanded === source.chunk_id;
          const body = open ? source.context_text || source.text : source.text;
          return (
            <li
              key={source.chunk_id}
              className="border-ink-700 bg-ink-900/50 rounded-lg border p-3"
            >
              <div className="flex flex-wrap items-center gap-1.5 text-xs">
                <span className="bg-ink-800 text-ink-300 rounded px-1.5 font-mono">
                  {index + 1}
                </span>
                {source.strategies.map((strategy) => (
                  <span key={strategy} className="tag" title={STRATEGY_LABELS[strategy]}>
                    {strategy}
                  </span>
                ))}
                <span className="tag">{source.lang}</span>
                <span className="ml-auto flex gap-1.5">
                  <span className="tag" title="reciprocal-rank-fusion score">
                    rrf {source.score.toFixed(3)}
                  </span>
                  {source.dense_rank !== null && (
                    <span className="tag text-cyan-300" title="dense (vector) rank">
                      dense #{source.dense_rank}
                    </span>
                  )}
                  {source.sparse_rank !== null && (
                    <span className="tag text-amber-300" title="BM25 rank">
                      bm25 #{source.sparse_rank}
                    </span>
                  )}
                </span>
              </div>
              <p className="text-ink-300 mt-2 text-sm leading-relaxed">{body}</p>
              {source.context_text && source.context_text !== source.text && (
                <button
                  type="button"
                  onClick={() => setExpanded(open ? null : source.chunk_id)}
                  className="text-brand-400 hover:text-brand-500 mt-2 font-mono text-xs"
                >
                  {open ? "− collapse" : "+ parent passage"}
                </button>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
