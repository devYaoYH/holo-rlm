"""Small OpenAI-compatible server around the native, instrumented checkpoint."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from .model import InstrumentedHolo, UnsupportedImageSource, parse_assistant_output
from .settings import Settings
from .traces import TraceOptions, append_trace_response

logger = logging.getLogger(__name__)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)


def create_app(settings: Settings | None = None, engine: InstrumentedHolo | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()
    engine = engine or InstrumentedHolo(settings)
    app = FastAPI(title="Instrumented Holo local server", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "loaded": engine.model is not None, "model_path": str(settings.model_path)}

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "Hcompany/Holo-3.1-4B",
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                    "revision": engine.checkpoint_revision(),
                    "capabilities": {
                        "vision": True,
                        "function_calling": True,
                        "structured_output": "schema-validated-tool-calls",
                        "activation_tracing": True,
                        "generated_token_logprobs": True,
                    },
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, request: Request) -> dict[str, Any]:
        raw_request = await request.json()
        trace_options = TraceOptions.from_request(raw_request)
        max_tokens = payload.max_completion_tokens or payload.max_tokens or settings.default_max_tokens
        try:
            text, prompt_tokens, completion_tokens, trace_id, thinking_enabled = engine.complete(
                messages=payload.messages,
                tools=payload.tools,
                max_tokens=max_tokens,
                temperature=payload.temperature,
                top_p=payload.top_p,
                trace_options=trace_options,
                raw_request=raw_request,
            )
        except (FileNotFoundError, UnsupportedImageSource, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Local inference failed")
            raise HTTPException(status_code=500, detail=f"Local inference failed: {exc!r}") from exc

        visible_text, tool_calls = parse_assistant_output(text, thinking_enabled=thinking_enabled)
        message: dict[str, Any] = {"role": "assistant", "content": visible_text or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        response: dict[str, Any] = {
            "id": f"chatcmpl-{time.time_ns()}",
            "object": "chat.completion",
            "created": int(datetime.now(UTC).timestamp()),
            "model": payload.model or "Hcompany/Holo-3.1-4B",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": (
                        "tool_calls" if tool_calls else "length" if completion_tokens >= max_tokens else "stop"
                    ),
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        if trace_id:
            response["instrumented_trace_id"] = trace_id
            append_trace_response(settings.trace_dir, trace_id, response)
        return response

    return app


app = create_app()


def main() -> None:
    settings = Settings.from_environment()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
