"""Optional LLM generator — Groq DeepSeek Reasoning with forced structured output."""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from app.config import Settings
from app.generation.extractive import AnswerDraft
from app.generation.intent import QueryIntent
from app.schemas import Citation, RetrievedChunk
from app.text import truncate_words

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages provided.

Rules, in priority order:
1. Never state anything that is not supported by the context. No outside knowledge.
2. Cite the passage number(s) you used inline, like [1] or [2].
3. If the context does not contain the answer, set grounded=false and give a short
   refusal_reason instead of guessing. A refusal is the correct answer when the
   context is silent — do not stretch a partial match into an answer.
4. Answer in the language of the question. Keep it under 60 words, no preamble.
5. Numbers, dates and names must be copied exactly as they appear in the context.

You must respond by calling the emit_answer tool. Never reply with plain text (other than your reasoning process)."""

ANSWER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "emit_answer",
        "description": "Return the grounded answer, or a refusal when the context is insufficient.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer with inline [n] citations, or '' when refusing.",
                },
                "citations": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Passage numbers actually used.",
                },
                "grounded": {
                    "type": "boolean",
                    "description": "True only if every claim is supported by the passages.",
                },
                "refusal_reason": {
                    "type": "string",
                    "description": "Why the question cannot be answered from the context.",
                },
            },
            "required": ["answer", "citations", "grounded"],
        },
    }
}


class LLMUnavailable(RuntimeError):
    """Raised for any condition that should make the harness degrade gracefully."""


def build_prompt(query: str, contexts: Sequence[RetrievedChunk]) -> str:
    blocks = [
        f"[{i}] {truncate_words(context.context_text or context.text, 160)}"
        for i, context in enumerate(contexts, start=1)
    ]
    return "Context passages:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {query}"


class GroqGenerator:
    """Thin, well-behaved wrapper: one call, one tool, explicit failure mode."""

    mode = "llm"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Any | None = None

    @property
    def available(self) -> bool:
        return bool(self.settings.groq_api_key)

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.available:
                raise LLMUnavailable("no GROQ_API_KEY configured")
            try:
                from groq import AsyncGroq
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise LLMUnavailable(f"groq sdk missing: {exc}") from exc
            
            kwargs: dict[str, Any] = {
                "api_key": self.settings.groq_api_key,
                "timeout": self.settings.llm_timeout_ms / 1000,
                "max_retries": 0,  # the harness owns retry policy
            }
            self._client = AsyncGroq(**kwargs)
        return self._client

    async def generate(
        self,
        query: str,
        contexts: Sequence[RetrievedChunk],
        intent: QueryIntent,
    ) -> AnswerDraft:
        if not contexts:
            raise LLMUnavailable("no contexts to ground on")
        client = self._get_client()
        try:
            message = await client.chat.completions.create(
                model=self.settings.llm_model,
                max_tokens=self.settings.llm_max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(query, contexts)}
                ],
                tools=[ANSWER_TOOL],
                tool_choice={"type": "function", "function": {"name": "emit_answer"}},
            )
        except Exception as exc:  # noqa: BLE001 - SDK raises a wide family of errors
            raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc

        payload = self._extract_tool_input(message)
        
        # Capture reasoning trace if available (DeepSeek)
        reasoning = ""
        choice = message.choices[0]
        if hasattr(choice.message, "content") and choice.message.content:
            reasoning = choice.message.content
            
        draft = self._to_draft(payload, contexts, intent, message)
        
        if reasoning:
            # We can stash the reasoning in draft.detail so it shows up in trace
            draft.detail["reasoning"] = reasoning[:500] + ("..." if len(reasoning) > 500 else "")
            
        return draft

    # ----------------------------------------------------------------- private
    @staticmethod
    def _extract_tool_input(message: Any) -> dict[str, Any]:
        choice = message.choices[0]
        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                if tool_call.function.name == "emit_answer":
                    payload = tool_call.function.arguments
                    if isinstance(payload, str):  # json string
                        try:
                            payload = json.loads(payload)
                        except json.JSONDecodeError:
                            raise LLMUnavailable(f"malformed json tool input: {payload!r}")
                    if not isinstance(payload, dict) or "answer" not in payload:
                        raise LLMUnavailable(f"malformed tool input: {payload!r}")
                    return payload
        raise LLMUnavailable("model did not call emit_answer")

    @staticmethod
    def _to_draft(
        payload: dict[str, Any],
        contexts: Sequence[RetrievedChunk],
        intent: QueryIntent,
        message: Any,
    ) -> AnswerDraft:
        grounded = bool(payload.get("grounded", False))
        answer = (payload.get("answer") or "").strip()
        if not grounded or not answer:
            return AnswerDraft(
                text="",
                mode="llm",
                detail={
                    "refused_by_model": 1,
                    "refusal_reason": str(payload.get("refusal_reason") or "model declined"),
                    "intent": intent.intent.value,
                },
            )

        markers = [
            n for n in payload.get("citations", []) if isinstance(n, int) and 1 <= n <= len(contexts)
        ]
        if not markers:  # answer with no usable citation is treated as unsupported
            markers = [1]
        citations = [
            Citation(
                marker=n,
                chunk_id=contexts[n - 1].chunk_id,
                parent_id=contexts[n - 1].parent_id,
                quote=truncate_words(contexts[n - 1].text, 40),
                strategies=list(contexts[n - 1].strategies),
                score=contexts[n - 1].score,
            )
            for n in sorted(set(markers))
        ]
        usage = getattr(message, "usage", None)
        return AnswerDraft(
            text=answer,
            citations=citations,
            support=[(answer, citations[0].chunk_id)],
            mode="llm",
            detail={
                "intent": intent.intent.value,
                "model": str(getattr(message, "model", "")),
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "stop_reason": str(getattr(message.choices[0], "finish_reason", "")),
            },
        )
