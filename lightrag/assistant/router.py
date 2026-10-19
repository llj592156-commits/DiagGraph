"""Replaceable five-strategy router with source-checked context and bounded calls."""

from __future__ import annotations

import re
from typing import Protocol

from .conversation import conversation_context, is_regeneration_request
from .model import ModelSession
from .router_prompt import ROUTING
from .schemas import AssistantRequest, RouteDecision


def _normalize(text: str) -> str:
    return "".join(text.casefold().split())


def danger_flags(text: str) -> list[str]:
    patterns = {
        "smoke_or_sparks": r"冒烟|焦味|火花|起火|\b(?:smoking|sparks|burning smell)\b",
        "electrical_repair": r"(?:带电|通电).{0,8}(?:拆|检修)|(?:拆开|拆修).{0,8}电源|短接.{0,8}(?:保护|开关)|repair.{0,12}live wiring",
    }
    return [
        name
        for name, pattern in patterns.items()
        if re.search(pattern, text, re.IGNORECASE)
    ]


def unsafe_instruction(text: str) -> bool:
    for clause in re.split(r"[。；;\n]", text):
        if re.search(r"不要|切勿|不能|禁止|不应|do not|never", clause, re.IGNORECASE):
            continue
        if "electrical_repair" in danger_flags(clause):
            return True
    return False


def risk_decision(flags: list[str]) -> RouteDecision:
    return RouteDecision(
        schema_version="diaggraph-route-v1",
        intent="troubleshooting",
        strategy="escalate",
        extracted_context=[],
        missing_fields=[],
        clarification_question=None,
        reason="明确的设备风险需要优先停止排查并寻求专业售后。",
        risk_flags=flags,
    )


def validate_route_context(decision: RouteDecision, request: AssistantRequest) -> None:
    conversation = conversation_context(request)
    if conversation["action"] == "regenerate":
        if conversation["target_user_message_index"] is None:
            if decision.strategy not in {"clarify", "escalate"}:
                raise ValueError(
                    "No prior answer is available; clarify what to regenerate"
                )
        elif conversation["requires_evidence_check"] and decision.strategy == "direct":
            raise ValueError(
                "Regenerating a previously cited answer requires fresh evidence checking. "
                "Resolve the original business question and use retrieve_once or diagnose; "
                "this is not a direct transformation of user-supplied text."
            )
        if decision.resolved_query and is_regeneration_request(decision.resolved_query):
            raise ValueError(
                "resolved_query must describe the original task, not the retry command"
            )
    for fact in decision.extracted_context:
        if fact.message_index == -1:
            text = request.query
        elif (
            fact.message_index >= len(request.history)
            or request.history[fact.message_index].role != "user"
        ):
            raise ValueError("Context must cite a real user message")
        else:
            text = request.history[fact.message_index].content
        if fact.quote not in text or _normalize(fact.value) not in _normalize(
            fact.quote
        ):
            raise ValueError(
                "Context value and quote must be present in the cited user message"
            )
    device_messages = [
        (i, m.content) for i, m in enumerate(request.history) if m.role == "user"
    ] + [(-1, request.query)]
    latest = set()
    for _, text in device_messages:
        found = set(
            re.findall(r"(?<![A-Z0-9])[A-Z]{1,6}\d{3,6}(?![A-Z0-9])", text.upper())
        )
        if found:
            latest = found
    models = [
        f.value.upper() for f in decision.extracted_context if f.field == "device_model"
    ]
    # Device-dependent routes need inherited identity; a new greeting or pure
    # text transformation may omit unrelated device facts from earlier turns.
    needs_device_context = decision.strategy in {"retrieve_once", "diagnose"}
    if latest and (needs_device_context or models) and set(models) != latest:
        raise ValueError("Device context must preserve the latest explicit model set")
    if needs_device_context and decision.intent in {"general_chat", "text_transform"}:
        raise ValueError(
            "General chat and pure text transformation do not require retrieval; "
            "route the current request directly instead of repeating a history question"
        )
    if needs_device_context and request.history and not decision.resolved_query:
        raise ValueError(
            "Provide resolved_query as a standalone question for this conversational "
            "follow-up. Resolve references using relevant history without inventing "
            "user observations or completed actions."
        )
    if (
        needs_device_context
        and decision.resolved_query
        and any(
            _normalize(model) not in _normalize(decision.resolved_query)
            for model in models
        )
    ):
        raise ValueError("resolved_query must retain the confirmed device models")
    if decision.strategy == "direct" and decision.intent not in {
        "general_chat",
        "text_transform",
    }:
        raise ValueError("Device facts cannot bypass retrieval")
    if (
        decision.strategy == "direct"
        and decision.intent == "general_chat"
        and re.search(
            r"墨水|耗材|保修|故障码|清洗|怎么修|ink|warranty|error code",
            request.query,
            re.IGNORECASE,
        )
    ):
        raise ValueError("Device facts disguised as general chat require evidence")
    if decision.intent == "order_status" and decision.strategy not in {
        "clarify",
        "escalate",
    }:
        raise ValueError("No live order system is connected")


class RouterBackend(Protocol):
    async def decide(self, request: AssistantRequest) -> RouteDecision: ...


class LLMRouter:
    def __init__(self, session: ModelSession, timeout: float = 60):
        self.session, self.timeout = session, timeout
        self.original_strategy: str | None = None
        self.overrides: list[str] = []

    async def decide(self, request: AssistantRequest) -> RouteDecision:
        flags = danger_flags(request.query)
        if flags:
            self.overrides.append("explicit_risk_preflight")
            return risk_decision(flags)

        decision = await self.session.call(
            RouteDecision,
            ROUTING,
            {
                **request.model_dump(),
                "conversation_context": conversation_context(request),
            },
            stage="routing",
            timeout=self.timeout,
            validate=lambda decision: validate_route_context(decision, request),
        )
        self.original_strategy = decision.strategy
        if decision.risk_flags and decision.strategy != "escalate":
            self.overrides.append("model_reported_risk")
            decision = decision.model_copy(
                update={
                    "strategy": "escalate",
                    "missing_fields": [],
                    "clarification_question": None,
                }
            )
        return decision
