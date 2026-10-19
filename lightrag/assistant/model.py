"""Bounded structured model calls; cancellation and provider errors remain visible."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic import ValidationError

from lightrag.utils import is_truncated_response

from .schemas import Contract

T = TypeVar("T", bound=Contract)
LLMFunction = Callable[..., Awaitable[object]]


class AssistantFailure(Exception):
    def __init__(self, code: str, stage: str, message: str):
        super().__init__(message)
        self.code, self.stage, self.message = code, stage, message


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError("Non-finite JSON value")


class ModelSession:
    """Own one request's audit trail; the service shares only concurrency capacity."""

    def __init__(
        self,
        llm: LLMFunction,
        semaphore: asyncio.Semaphore,
        count_tokens: Callable[[str], int],
        max_input_tokens: int,
        model_identity: Callable[[], dict] | None = None,
    ):
        self.llm, self.semaphore = llm, semaphore
        self.count_tokens, self.max_input_tokens = count_tokens, max_input_tokens
        self.model_identity = model_identity
        self.trace: list[dict] = []
        self.stage = "routing"

    async def call(
        self,
        schema: type[T],
        instruction: str,
        payload: dict,
        *,
        stage: str,
        timeout: float = 60,
        validate: Callable[[T], None] | None = None,
    ) -> T:
        self.stage = stage
        deadline = time.monotonic() + timeout
        system = (
            instruction
            + "\nReturn ONLY a complete JSON object matching this schema:\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        )
        repair = False
        repair_feedback = ""
        for attempt in range(2):
            prompt = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            if repair:
                prompt += (
                    "\nThe previous response violated the output contract. "
                    "Recheck the schema and semantic constraints. Return complete JSON only."
                    + repair_feedback
                )
            if self.count_tokens(system + "\n" + prompt) > self.max_input_tokens:
                raise AssistantFailure(
                    "input_budget_exceeded",
                    stage,
                    "Model input exceeds the configured token budget",
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssistantFailure(
                    "model_timeout", stage, "Model stage deadline exceeded"
                )
            entry = {"stage": stage, "attempt": attempt + 1}
            self.trace.append(entry)
            started = time.monotonic()
            validation_phase = "response_contract"
            try:

                async def invoke(prompt=prompt, entry=entry):
                    async with self.semaphore:
                        if self.model_identity:
                            entry["model_identity"] = self.model_identity()
                        entry["invoked"] = True
                        return await self.llm(
                            prompt, system_prompt=system, stream=False
                        )

                response = await asyncio.wait_for(invoke(), remaining)
                if not isinstance(response, str):
                    raise ValueError("A nonstreaming text response is required")  # noqa: TRY004
                entry["response_sha256"] = hashlib.sha256(response.encode()).hexdigest()
                if len(response) > 65536:
                    raise ValueError("Structured response exceeds 65536 characters")
                entry["raw_response"] = str(response)
                if is_truncated_response(response):
                    raise ValueError("Truncated structured response")
                validation_phase = "json"
                parsed = json.loads(
                    response,
                    object_pairs_hook=_object,
                    parse_constant=_invalid_constant,
                )
                validation_phase = "schema"
                value = schema.model_validate(parsed)
                if validate:
                    validation_phase = "semantic"
                    validate(value)
                entry["status"] = "success"
                return value
            except (ValueError, ValidationError) as exc:
                entry.update(
                    status="invalid_response",
                    error_type=type(exc).__name__,
                    validation_phase=validation_phase,
                )
                if attempt:
                    raise AssistantFailure(
                        "invalid_model_output",
                        stage,
                        "Model output failed structural or semantic validation",
                    ) from exc
                repair = True
                if type(exc) is ValueError and validate is not None:
                    repair_feedback = "\nValidation issue: " + str(exc)[:500]
            except Exception as exc:
                entry.update(status="provider_error", error_type=type(exc).__name__)
                status = getattr(exc, "status_code", None)
                transient = (
                    isinstance(exc, (TimeoutError, asyncio.TimeoutError, OSError))
                    or status in {408, 429, 500, 502, 503, 504}
                    or type(exc).__name__ in {"APITimeoutError", "APIConnectionError"}
                )
                if not transient or attempt:
                    code = (
                        "model_timeout"
                        if isinstance(exc, (TimeoutError, asyncio.TimeoutError))
                        else "model_request_failed"
                    )
                    raise AssistantFailure(
                        code,
                        stage,
                        "Model request failed; inspect provider configuration or availability",
                    ) from exc
            finally:
                entry["elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)
                entry.setdefault("status", "cancelled")
        raise AssertionError("Unreachable structured call state")
