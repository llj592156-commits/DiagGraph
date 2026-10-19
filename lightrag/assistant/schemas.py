"""Validated contracts shared by replaceable routing and execution backends."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Strategy = Literal["direct", "retrieve_once", "diagnose", "clarify", "escalate"]
Intent = Literal[
    "general_chat",
    "text_transform",
    "device_fact",
    "fault_code_lookup",
    "troubleshooting",
    "after_sales_policy",
    "order_status",
    "unknown",
]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Message(Contract):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4096)


class AssistantRequest(Contract):
    query: str = Field(min_length=1, max_length=8192)
    history: list[Message] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def validate_input(self) -> AssistantRequest:
        self.query = self.query.strip()
        if not self.query:
            raise ValueError("query must contain non-whitespace text")
        if len(json.dumps(self.model_dump(), ensure_ascii=False)) > 24000:
            raise ValueError("Combined conversation input exceeds 24000 characters")
        return self


class ContextFact(Contract):
    field: Literal[
        "device_model", "fault_code", "code_location", "symptom", "attempted_action"
    ]
    value: str = Field(min_length=1, max_length=256)
    message_index: int = Field(ge=-1, le=31)
    quote: str = Field(min_length=1, max_length=512)


class RouteDecision(Contract):
    schema_version: Literal["diaggraph-route-v1"]
    intent: Intent
    strategy: Strategy
    resolved_query: str | None = Field(default=None, min_length=1, max_length=2000)
    extracted_context: list[ContextFact] = Field(max_length=24)
    missing_fields: list[str] = Field(max_length=8)
    clarification_question: str | None = Field(max_length=1000)
    reason: str = Field(min_length=1, max_length=1000)
    risk_flags: list[str] = Field(max_length=8)

    @model_validator(mode="after")
    def validate_strategy(self) -> RouteDecision:
        if self.resolved_query is not None and not self.resolved_query.strip():
            raise ValueError("resolved_query must contain non-whitespace text")
        if any(not field.strip() or len(field) > 100 for field in self.missing_fields):
            raise ValueError("Missing fields must be nonempty bounded names")
        if any(not flag.strip() or len(flag) > 100 for flag in self.risk_flags):
            raise ValueError("Risk flags must be nonempty bounded names")
        if self.strategy == "clarify":
            if (
                not self.missing_fields
                or not (self.clarification_question or "").strip()
            ):
                raise ValueError("Clarification requires missing fields and a question")
        elif self.missing_fields and self.strategy != "escalate":
            raise ValueError("Necessary missing information requires clarification")
        if self.intent == "unknown" and self.strategy not in {"clarify", "escalate"}:
            raise ValueError("Unknown intent requires clarification or escalation")
        return self

    @property
    def complexity(self) -> str | None:
        return {"direct": "A", "retrieve_once": "B", "diagnose": "C"}.get(self.strategy)


class EvidenceSource(Contract):
    chunk_id: str
    doc_id: str
    file_path: str
    parent_child: dict[str, Any] | None = None
    business_context: dict[str, Any] | None = None


class ParentSource(Contract):
    parent_id: str
    doc_id: str
    metadata: dict[str, Any]


class EvidenceItem(Contract):
    evidence_id: str
    content: str
    doc_id: str
    file_path: str
    source_chunk_ids: list[str]
    sources: list[EvidenceSource]
    parents: list[ParentSource] = Field(default_factory=list)
    primary_parent_id: str | None = None
    evidence_level: Literal["chunk", "full_parent", "child_with_prerequisites"] = (
        "chunk"
    )
    original_reference_id: str = ""


class RetrievalBatch(Contract):
    evidence: list[EvidenceItem]
    raw_data: dict[str, Any]


class EvidenceReview(Contract):
    accepted_evidence_ids: list[str] = Field(max_length=20)
    sufficient: bool
    missing_information: list[str] = Field(max_length=8)
    followup_queries: list[str] = Field(max_length=2)
    clarification_question: str | None = Field(max_length=1000)
    reason: str = Field(min_length=1, max_length=1500)


class SupportedClaim(Contract):
    text: str = Field(min_length=1, max_length=1500)
    evidence_ids: list[str] = Field(max_length=10)


class AnswerDraft(Contract):
    claims: list[SupportedClaim] = Field(min_length=1, max_length=12)


class AnswerReview(Contract):
    verdict: Literal["pass", "revise", "clarify", "escalate"]
    reason: str = Field(min_length=1, max_length=1500)
    clarification_question: str | None = Field(max_length=1000)

    @model_validator(mode="after")
    def validate_clarification(self) -> AnswerReview:
        if (
            self.verdict == "clarify"
            and not (self.clarification_question or "").strip()
        ):
            raise ValueError("Clarification verdict requires a targeted question")
        return self


class AssistantResponse(Contract):
    status: Literal[
        "answered",
        "needs_clarification",
        "escalated",
        "insufficient_evidence",
        "failed",
    ]
    answer: str
    route: RouteDecision | None = None
    complexity: str | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, str] | None = None
