"""Bounded assistant execution using the exact evidence checked before generation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from lightrag.base import QueryParam
from lightrag.utils import logger

from .conversation import conversation_context
from .evidence import model_evidence
from .model import AssistantFailure, LLMFunction, ModelSession
from .retrieval import LightRAGAdapter
from .router import (
    LLMRouter,
    RouterBackend,
    danger_flags,
    risk_decision,
    unsafe_instruction,
    validate_route_context,
)
from .router_prompt import ANSWER_REVIEW, EVIDENCE_REVIEW, GENERATION, PROMPT_VERSION
from .schemas import (
    AnswerDraft,
    AnswerReview,
    AssistantRequest,
    AssistantResponse,
    EvidenceItem,
    EvidenceReview,
    RouteDecision,
)
from .validation import asks_user_observation, validate_clarification


@dataclass(frozen=True)
class AssistantConfig:
    total_timeout: float = 300
    routing_timeout: float = 60
    model_timeout: float = 60
    max_input_tokens: int = 16000
    evidence_tokens: int = 8000
    max_retrievals: int = 3
    max_parallel_llm: int = 2

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("Assistant budgets must be positive")
        if self.max_retrievals > 3:
            raise ValueError("The first version permits at most three retrievals")


ESCALATION = "请停止相关操作；如设备冒烟、有火花或明显异常，请远离设备并联系专业售后，不要尝试拆修。当前仅提供升级建议，尚未实际转接人工或创建工单。"
INSUFFICIENT = "现有资料不足以可靠回答这个问题，暂不提供未经证据支持的判断或操作建议。"


class AssistantService:
    """Use initialized storages; inject an alternative model or router without changing retrieval."""

    def __init__(
        self,
        rag,
        *,
        llm: LLMFunction | None = None,
        config: AssistantConfig | None = None,
        query_param: QueryParam | None = None,
        router_factory: Callable[[ModelSession], RouterBackend] | None = None,
        audit_sink=None,
        model_identity: str | None = None,
    ):
        self.rag, self.config = rag, config or AssistantConfig()
        self.adapter = LightRAGAdapter(rag, query_param)
        self.semaphore = asyncio.Semaphore(self.config.max_parallel_llm)
        self.router_factory = router_factory or (
            lambda session: LLMRouter(session, self.config.routing_timeout)
        )

        async def role_llm(*args, **kwargs):
            return await rag.role_llm_funcs["query"](*args, **kwargs)

        self.llm = llm or role_llm
        self.model_role = "injected" if llm is not None else "query"
        self.audit_sink = audit_sink
        self.injected_model_identity = model_identity

    def _model_identity(self) -> dict:
        if self.model_role == "injected":
            return {"role": "injected", "model": self.injected_model_identity}
        getter = getattr(self.rag, "get_llm_role_config", None)
        config = getter("query") if callable(getter) else {}
        return {
            "role": "query",
            "binding": config.get("binding"),
            "model": config.get("model"),
        }

    async def run(self, request: AssistantRequest) -> AssistantResponse:
        started = time.monotonic()
        tokenizer = getattr(self.rag, "tokenizer", None)
        if tokenizer is None:
            return AssistantResponse(
                status="failed",
                answer="",
                error={
                    "code": "missing_tokenizer",
                    "stage": "input",
                    "message": "An initialized LightRAG tokenizer is required",
                },
            )
        session = ModelSession(
            self.llm,
            self.semaphore,
            lambda text: len(tokenizer.encode(text)),
            self.config.max_input_tokens,
            self._model_identity,
        )
        audit = {
            "retrievals": [],
            "evidence_reviews": [],
            "answer_reviews": [],
            "omitted_evidence_ids": [],
            "evidence_selection": [],
            "conversation_context": conversation_context(request),
            "termination_reason": None,
        }
        route = None
        router = self.router_factory(session)

        async def execute():
            nonlocal route
            flags = danger_flags(request.query)
            if flags:
                route = risk_decision(flags)
                if hasattr(router, "overrides"):
                    router.overrides.append("explicit_risk_preflight")
            else:
                route = await router.decide(request)
            # Custom backends must also return validated contracts.
            if not isinstance(route, RouteDecision):
                raise AssistantFailure(
                    "invalid_router_backend",
                    "routing",
                    "Router backend returned an invalid contract",
                )
            if not flags:
                try:
                    route = RouteDecision.model_validate(route.model_dump())
                    validate_route_context(route, request)
                except ValueError as exc:
                    raise AssistantFailure(
                        "invalid_router_backend",
                        "routing",
                        "Router decision violates the execution contract",
                    ) from exc
                if route.risk_flags:
                    route = route.model_copy(
                        update={
                            "strategy": "escalate",
                            "missing_fields": [],
                            "clarification_question": None,
                        }
                    )
            if route.strategy == "escalate":
                answer = (
                    ESCALATION
                    if route.risk_flags
                    else "该问题需要专业售后或具有相应权限的工作人员处理。当前尚未接通订单查询、实际转接人工或创建工单。"
                )
                return AssistantResponse(status="escalated", answer=answer)
            if route.strategy == "clarify":
                return AssistantResponse(
                    status="needs_clarification",
                    answer=route.clarification_question or "请补充必要信息。",
                )
            approved = []
            if route.strategy != "direct":
                approved, review = await self._retrieve(request, route, session, audit)
                if not approved:
                    question = review.clarification_question
                    audit["termination_reason"] = (
                        "user_clarification_required"
                        if question
                        else "evidence_insufficient"
                    )
                    return AssistantResponse(
                        status="needs_clarification"
                        if question
                        else "insufficient_evidence",
                        answer=question or INSUFFICIENT,
                    )
            return await self._generate(request, route, approved, session, audit)

        try:
            response = await asyncio.wait_for(execute(), self.config.total_timeout)
        except AssistantFailure as exc:
            response = AssistantResponse(
                status="failed",
                answer="",
                error={"code": exc.code, "stage": exc.stage, "message": exc.message},
            )
        except (TimeoutError, asyncio.TimeoutError):
            response = AssistantResponse(
                status="failed",
                answer="",
                error={
                    "code": "request_timeout",
                    "stage": session.stage,
                    "message": "Assistant request deadline exceeded",
                },
            )
        except Exception as exc:  # noqa: BLE001 - explicit orchestration failure boundary
            # No provider exception text (possibly containing credentials) is sent to clients.
            logger.error(
                "Assistant request failed during %s (%s)",
                session.stage,
                type(exc).__name__,
            )
            response = AssistantResponse(
                status="failed",
                answer="",
                error={
                    "code": "assistant_execution_failed",
                    "stage": session.stage,
                    "message": "Assistant execution failed; check server diagnostics",
                },
            )
        response.route = route
        response.complexity = route.complexity if route else None
        response.metadata = {
            "prompt_version": PROMPT_VERSION,
            "backend": "llm"
            if isinstance(router, LLMRouter)
            else type(router).__name__,
            "mode": "mix"
            if route and route.strategy in {"retrieve_once", "diagnose"}
            else None,
            "original_strategy": getattr(router, "original_strategy", None),
            "route_overrides": getattr(router, "overrides", []),
            "input_sha256": hashlib.sha256(
                request.model_dump_json().encode()
            ).hexdigest(),
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            "model_calls": sum(bool(entry.get("invoked")) for entry in session.trace),
            "model_attempts": len(session.trace),
            "provider_http_requests": None,
            "token_usage": None,
            "model_role": self.model_role,
            "model_trace": [
                {k: v for k, v in entry.items() if k != "raw_response"}
                for entry in session.trace
            ],
            **audit,
        }
        if self.audit_sink is not None:
            try:
                remaining = max(
                    0.001, self.config.total_timeout - (time.monotonic() - started)
                )
                await asyncio.wait_for(
                    self.audit_sink(
                        {
                            "response": response.model_dump(),
                            "model_trace": session.trace,
                        }
                    ),
                    remaining,
                )
            except Exception as exc:  # noqa: BLE001 - configured audit failures must be reported
                response.status, response.answer = "failed", ""
                response.evidence, response.citations = [], []
                response.error = {
                    "code": "audit_write_failed",
                    "stage": "audit",
                    "message": "Configured audit sink failed; results were not reported as complete",
                }
                response.metadata["audit_error_type"] = type(exc).__name__
        logger.info(
            "Assistant request completed: status=%s strategy=%s model_calls=%s "
            "retrievals=%s termination=%s input_sha256=%s",
            response.status,
            route.strategy if route else None,
            response.metadata["model_calls"],
            len(audit["retrievals"]),
            audit["termination_reason"],
            response.metadata["input_sha256"],
        )
        if response.status == "failed":
            # Log only controlled diagnostics, never provider text, prompts or replies.
            model_errors = [
                {
                    key: entry[key]
                    for key in (
                        "stage",
                        "attempt",
                        "status",
                        "error_type",
                        "validation_phase",
                    )
                    if key in entry
                }
                for entry in session.trace
                if entry.get("status") in {"invalid_response", "provider_error"}
            ]
            logger.warning(
                "Assistant request failed: code=%s stage=%s strategy=%s "
                "input_sha256=%s model_errors=%s",
                (response.error or {}).get("code"),
                (response.error or {}).get("stage"),
                route.strategy if route else None,
                response.metadata["input_sha256"],
                model_errors,
            )
        return response

    def _payload(self, request: AssistantRequest, route: RouteDecision) -> dict:
        conversation = conversation_context(request)
        question = request.query
        if route.strategy in {"retrieve_once", "diagnose"}:
            question = route.resolved_query or question
        elif (
            conversation["action"] == "regenerate"
            and conversation["target_user_message_index"] is not None
        ):
            question = request.history[
                conversation["target_user_message_index"]
            ].content
        return {
            "request": request.model_dump(),
            "route": route.model_dump(),
            "current_question": question,
            "conversation_context": conversation,
        }

    @staticmethod
    def _search_query(query: str, route: RouteDecision) -> str:
        facts = [f"{f.field}: {f.value}" for f in route.extracted_context]
        return query + ("\n用户已确认信息：\n" + "\n".join(facts) if facts else "")

    async def _retrieve(self, request, route, session, audit):
        collected: dict[str, EvidenceItem] = {}
        seen_queries = set()
        queries = [route.resolved_query or request.query]
        accepted_ids = []
        limit = self.config.max_retrievals if route.strategy == "diagnose" else 1
        for round_index in range(2):
            previous_ids = set(collected)
            round_ids = []
            for query in queries:
                search = self._search_query(query, route)
                canonical = "".join(search.casefold().split())
                if canonical in seen_queries or len(audit["retrievals"]) >= limit:
                    continue
                seen_queries.add(canonical)
                session.stage = "retrieval"
                attempt = {
                    "query": search,
                    "round": round_index + 1,
                    "status": "pending",
                }
                audit["retrievals"].append(attempt)
                try:
                    batch = await self.adapter.retrieve(search)
                    attempt.update(
                        status="success", metadata=batch.raw_data.get("metadata", {})
                    )
                except Exception as exc:
                    attempt.update(status="failed", error_type=type(exc).__name__)
                    raise
                finally:
                    if attempt["status"] == "pending":
                        attempt["status"] = "cancelled"
                for item in batch.evidence:
                    collected[item.evidence_id] = item
                    round_ids.append(item.evidence_id)
            fitted = []
            # Retain previously accepted support and admit gap results before old
            # unaccepted groups. Insertion order must not starve the second round.
            new_ids = [key for key in round_ids if key not in previous_ids]
            priority = dict.fromkeys([*accepted_ids, *new_ids, *round_ids, *collected])
            omitted = []
            for key in priority:
                item = collected[key]
                candidate = [*fitted, item]
                serialized = json.dumps(
                    [model_evidence(e) for e in candidate], ensure_ascii=False
                )
                if session.count_tokens(serialized) <= self.config.evidence_tokens:
                    fitted.append(item)
                else:
                    omitted.append(item.evidence_id)
                    if item.evidence_id not in audit["omitted_evidence_ids"]:
                        audit["omitted_evidence_ids"].append(item.evidence_id)
            audit["evidence_selection"].append(
                {
                    "round": round_index + 1,
                    "considered_count": len(collected),
                    "retained_evidence_ids": [e.evidence_id for e in fitted],
                    "omitted_evidence_ids": omitted,
                    "model_evidence_tokens": session.count_tokens(
                        json.dumps(
                            [model_evidence(e) for e in fitted], ensure_ascii=False
                        )
                    ),
                }
            )
            ids = {e.evidence_id for e in fitted}

            def validate(review: EvidenceReview, ids=ids):
                accepted = review.accepted_evidence_ids
                if len(accepted) != len(set(accepted)) or not set(accepted) <= ids:
                    raise ValueError("Evidence review cites unknown or duplicate IDs")
                if review.sufficient and (not accepted or review.missing_information):
                    raise ValueError(
                        "Sufficient evidence must support the whole question"
                    )
                validate_clarification(review.clarification_question, route)
                if review.sufficient and review.clarification_question:
                    raise ValueError(
                        "Sufficient evidence cannot require user clarification"
                    )
                if any(not q.strip() or len(q) > 2000 for q in review.followup_queries):
                    raise ValueError("Follow-up queries must be bounded and nonempty")
                if not review.clarification_question and any(
                    asks_user_observation(q) for q in review.followup_queries
                ):
                    raise ValueError(
                        "User observations belong in clarification_question. "
                        "followup_queries must search for knowledge in manuals."
                    )

            review = await session.call(
                EvidenceReview,
                EVIDENCE_REVIEW,
                {
                    **self._payload(request, route),
                    "evidence": [model_evidence(e) for e in fitted],
                },
                stage="evidence_review",
                timeout=self.config.model_timeout,
                validate=validate,
            )
            audit["evidence_reviews"].append(review.model_dump())
            accepted_ids = review.accepted_evidence_ids
            if review.sufficient:
                return [
                    e for e in fitted if e.evidence_id in review.accepted_evidence_ids
                ], review
            if review.clarification_question:
                break
            if round_index or len(audit["retrievals"]) >= limit:
                break
            queries = list(dict.fromkeys(q.strip() for q in review.followup_queries))[
                :2
            ]
            if not queries:
                break
        return [], review

    async def _generate(self, request, route, evidence, session, audit):
        payload = {
            **self._payload(request, route),
            "evidence": [model_evidence(e) for e in evidence],
        }
        ids = {e.evidence_id for e in evidence}
        feedback = None
        for revision in range(2):

            def validate(draft: AnswerDraft):
                for claim in draft.claims:
                    if not claim.text.strip() or re.search(r"\[\d+\]", claim.text):
                        raise ValueError(
                            "Claims require text without invented reference numbers"
                        )
                    if (
                        len(claim.evidence_ids) != len(set(claim.evidence_ids))
                        or not set(claim.evidence_ids) <= ids
                    ):
                        raise ValueError("Claim cites unknown or duplicate evidence")
                    if route.strategy != "direct" and not claim.evidence_ids:
                        raise ValueError("Every retrieved claim requires evidence")
                    if route.strategy == "direct" and claim.evidence_ids:
                        raise ValueError("Direct mode cannot cite nonexistent evidence")

            draft = await session.call(
                AnswerDraft,
                GENERATION,
                {**payload, "revision_feedback": feedback},
                stage="generation",
                timeout=self.config.model_timeout,
                validate=validate,
            )
            if any(unsafe_instruction(c.text) for c in draft.claims):
                audit["answer_reviews"].append(
                    {
                        "verdict": "escalate",
                        "reason": "independent_unsafe_instruction_rule",
                    }
                )
                return AssistantResponse(status="escalated", answer=ESCALATION)
            review = await session.call(
                AnswerReview,
                ANSWER_REVIEW,
                {**payload, "draft": draft.model_dump()},
                stage="answer_review",
                timeout=self.config.model_timeout,
                validate=lambda review: validate_clarification(
                    review.clarification_question, route
                ),
            )
            audit["answer_reviews"].append(review.model_dump())
            if review.verdict == "pass":
                used = {key for claim in draft.claims for key in claim.evidence_ids}
                final = [e for e in evidence if e.evidence_id in used]
                numbered = {e.evidence_id: str(i + 1) for i, e in enumerate(final)}
                answer = "\n\n".join(
                    c.text + "".join(f" [{numbered[key]}]" for key in c.evidence_ids)
                    for c in draft.claims
                )
                citations = [
                    {
                        "reference_id": numbered[e.evidence_id],
                        "evidence_id": e.evidence_id,
                        "doc_id": e.doc_id,
                        "source_chunk_ids": e.source_chunk_ids,
                        "file_path": e.file_path,
                    }
                    for e in final
                ]
                return AssistantResponse(
                    status="answered",
                    answer=answer,
                    evidence=final,
                    citations=citations,
                )
            if review.verdict == "escalate":
                return AssistantResponse(status="escalated", answer=ESCALATION)
            if review.verdict == "clarify":
                return AssistantResponse(
                    status="needs_clarification",
                    answer=review.clarification_question
                    or "请补充与当前问题有关的设备信息或操作情况。",
                )
            feedback = review.reason
        audit["termination_reason"] = "answer_revision_exhausted"
        return AssistantResponse(status="insufficient_evidence", answer=INSUFFICIENT)
