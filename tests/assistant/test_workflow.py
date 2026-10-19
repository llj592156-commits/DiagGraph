"""Exercise bounded routing/retrieval/generation without live model services."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from lightrag.assistant import AssistantConfig, AssistantRequest, AssistantService
from lightrag.assistant import model as model_module
from lightrag.assistant import workflow as workflow_module
from lightrag.assistant.schemas import RouteDecision
from tests.assistant.conftest import FlowLLM, route_response

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_deadlines_must_be_finite(value):
    with pytest.raises(ValueError):
        AssistantConfig(total_timeout=value)


@pytest.mark.parametrize(
    "instruction,elapsed",
    [("You route", 20), ("Evaluate the supplied", 40)],
)
async def test_slow_model_retry_uses_remaining_stage_budget(
    rag_data, monkeypatch, instruction, elapsed
):
    """Simulate slow transient calls without waiting or calling a provider."""
    rag, _, _ = rag_data
    clock = [0.0]
    monkeypatch.setattr(
        model_module, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    normal = FlowLLM()
    delayed = False

    async def slow_transient(prompt, *, system_prompt, stream):
        nonlocal delayed
        if system_prompt.startswith(instruction) and not delayed:
            delayed = True
            clock[0] += elapsed
            raise OSError("simulated transient provider failure")
        return await normal(prompt, system_prompt=system_prompt, stream=stream)

    result = await AssistantService(rag, llm=slow_transient).run(
        AssistantRequest(query="L3251 怎么操作？")
    )
    assert delayed
    assert result.status == "answered"
    assert result.metadata["model_calls"] == 5
    assert (
        sum(e["status"] == "provider_error" for e in result.metadata["model_trace"])
        == 1
    )


async def test_two_requests_do_not_share_model_trace_or_evidence(rag_data):
    rag, _, _ = rag_data
    service = AssistantService(rag, llm=FlowLLM())
    first, second = await asyncio.gather(
        service.run(AssistantRequest(query="L3251 怎么操作？")),
        service.run(AssistantRequest(query="L3251 如何维护？")),
    )
    assert first.status == second.status == "answered"
    assert first.metadata["model_calls"] == second.metadata["model_calls"] == 4
    assert first.metadata["input_sha256"] != second.metadata["input_sha256"]
    assert first.metadata["retrievals"] is not second.metadata["retrievals"]


async def test_replacement_router_cannot_bypass_safety_gate(rag_data):
    rag, _, _ = rag_data
    backend, llm = AsyncMock(), AsyncMock()
    result = await AssistantService(
        rag, llm=llm, router_factory=lambda session: backend
    ).run(AssistantRequest(query="打印机冒烟"))
    assert result.status == "escalated"
    backend.decide.assert_not_awaited()
    llm.assert_not_awaited()


async def test_replacement_router_cannot_forward_unvalidated_device_facts(rag_data):
    rag, _, _ = rag_data
    backend = AsyncMock()
    backend.decide.return_value = RouteDecision.model_validate(
        route_response("direct", query="", intent="device_fact")
    )
    result = await AssistantService(rag, router_factory=lambda session: backend).run(
        AssistantRequest(query="墨水编号是什么？")
    )
    assert result.status == "failed"
    assert result.error["code"] == "invalid_router_backend"
    rag.aquery_data.assert_not_awaited()


async def test_missing_source_fails_without_being_reported_as_insufficient(rag_data):
    rag, rows, _ = rag_data
    rows.clear()
    result = await AssistantService(rag, llm=FlowLLM()).run(
        AssistantRequest(query="L3251 怎么操作？")
    )
    assert result.status == "failed"
    assert result.error["code"] == "source_attribution_error"
    assert result.error["stage"] == "retrieval"
    assert len(result.metadata["retrievals"]) == 1
    assert result.metadata["retrievals"][0]["status"] == "failed"


async def test_model_identity_is_auditable_without_credentials(rag_data):
    rag, _, _ = rag_data
    rag.role_llm_funcs = {"query": FlowLLM()}
    rag.get_llm_role_config = lambda role: {
        "model": "test-model-v1",
        "binding": "test-provider",
        "api_key": "must-not-appear",
    }
    result = await AssistantService(rag).run(AssistantRequest(query="L3251 怎么操作？"))
    assert result.status == "answered"
    assert all(
        item["model_identity"]["model"] == "test-model-v1"
        for item in result.metadata["model_trace"]
    )
    assert "must-not-appear" not in result.model_dump_json()


async def test_queue_timeouts_do_not_claim_provider_calls_were_made(rag_data):
    rag, _, _ = rag_data
    service = AssistantService(
        rag, llm=AsyncMock(), config=AssistantConfig(routing_timeout=0.01)
    )
    service.semaphore = asyncio.Semaphore(0)
    result = await service.run(AssistantRequest(query="你好"))
    assert result.status == "failed"
    assert result.metadata["model_attempts"] == 1
    assert result.metadata["model_calls"] == 0


async def test_shared_model_capacity_is_enforced(rag_data):
    rag, _, _ = rag_data
    normal = FlowLLM(strategy="direct")
    active, maximum = 0, 0

    async def limited(*args, **kwargs):
        nonlocal active, maximum
        active += 1
        maximum = max(active, maximum)
        try:
            await asyncio.sleep(0)
            return await normal(*args, **kwargs)
        finally:
            active -= 1

    service = AssistantService(
        rag, llm=limited, config=AssistantConfig(max_parallel_llm=1)
    )
    answers = await asyncio.gather(
        *[service.run(AssistantRequest(query="你好")) for _ in range(3)]
    )
    assert all(r.status == "answered" for r in answers)
    assert maximum == 1


async def test_single_retrieval_uses_same_approved_evidence_and_programmatic_citations(
    rag_data,
):
    rag, _, _ = rag_data
    llm = FlowLLM()
    result = await AssistantService(rag, llm=llm).run(
        AssistantRequest(query="L3251 怎么操作？")
    )
    assert result.status == "answered"
    assert result.complexity == "B"
    assert rag.aquery_data.await_count == 1
    assert result.answer.endswith("[1]")
    assert result.citations[0]["source_chunk_ids"] == ["chunk-1"]
    generated = next(
        payload for system, payload in llm.calls if system.startswith("Write a useful")
    )
    reviewed = next(
        payload
        for system, payload in llm.calls
        if system.startswith("Check the proposed")
    )
    assert generated["evidence"] == reviewed["evidence"]
    assert generated["evidence"][0]["content"] == result.evidence[0].content
    assert all("raw_response" not in entry for entry in result.metadata["model_trace"])
    assert result.metadata["provider_http_requests"] is None


@pytest.mark.parametrize(
    "strategy,query,status",
    [
        ("direct", "你好", "answered"),
        ("clarify", "应该买什么墨水？", "needs_clarification"),
    ],
)
async def test_non_retrieval_strategies_do_not_touch_stores(
    rag_data, strategy, query, status
):
    rag, _, _ = rag_data
    result = await AssistantService(rag, llm=FlowLLM(strategy=strategy)).run(
        AssistantRequest(query=query)
    )
    assert result.status == status
    assert result.evidence == []
    rag.aquery_data.assert_not_awaited()


async def test_greeting_after_ink_answer_never_retrieves(rag_data):
    rag, _, _ = rag_data
    normal = FlowLLM(strategy="direct")
    result = await AssistantService(rag, llm=normal).run(
        AssistantRequest(
            query="你好",
            history=[
                {"role": "user", "content": "我的打印机有问题"},
                {"role": "assistant", "content": "请问打印机型号和具体问题是什么？"},
                {"role": "user", "content": "型号是L4263，应该使用什么墨水编号"},
                {"role": "assistant", "content": "L4263使用002墨水。[1]"},
            ],
        )
    )
    assert result.status == "answered"
    assert "你好" in result.answer
    assert result.route.strategy == "direct"
    assert result.metadata["model_calls"] == 3
    assert result.metadata["retrievals"] == []
    rag.aquery_data.assert_not_awaited()


async def test_chat_retrieval_misroute_is_repaired_before_tools(rag_data):
    rag, _, _ = rag_data
    normal = FlowLLM(strategy="direct")
    routed = 0

    async def misroute_once(prompt, **kwargs):
        nonlocal routed
        if kwargs["system_prompt"].startswith("You route"):
            routed += 1
            if routed == 1:
                return json.dumps(
                    route_response(
                        "retrieve_once",
                        query="",
                        intent="general_chat",
                        extracted_context=[
                            {
                                "field": "device_model",
                                "value": "L4263",
                                "quote": "L4263",
                                "message_index": 0,
                            }
                        ],
                    )
                )
            return json.dumps(route_response("direct", query="", intent="general_chat"))
        return await normal(prompt, **kwargs)

    result = await AssistantService(rag, llm=misroute_once).run(
        AssistantRequest(
            query="你好",
            history=[{"role": "user", "content": "L4263应该使用什么墨水编号？"}],
        )
    )
    assert result.status == "answered"
    assert result.route.strategy == "direct"
    assert routed == 2
    assert result.metadata["model_trace"][0]["validation_phase"] == "semantic"
    rag.aquery_data.assert_not_awaited()


@pytest.mark.parametrize(
    "raw,phase",
    [
        ("secret-provider-output", "json"),
        ('{"secret-provider-field": "secret-provider-output"}', "schema"),
        (
            json.dumps(
                route_response("retrieve_once", query="", intent="general_chat")
            ),
            "semantic",
        ),
    ],
)
async def test_failed_request_logs_validation_phase_without_raw_text(
    rag_data, monkeypatch, raw, phase
):
    rag, _, _ = rag_data
    warning = Mock()
    monkeypatch.setattr(workflow_module.logger, "warning", warning)
    result = await AssistantService(rag, llm=AsyncMock(return_value=raw)).run(
        AssistantRequest(query="private-user-query")
    )
    assert result.status == "failed"
    assert result.error["code"] == "invalid_model_output"
    assert result.metadata["model_trace"][0]["validation_phase"] == phase
    warning.assert_called_once()
    template, *args = warning.call_args.args
    log = template % tuple(args)
    assert "invalid_model_output" in log
    assert "routing" in log
    assert phase in log
    assert "secret-provider" not in log
    assert "private-user-query" not in log


async def test_risk_escalation_avoids_models_and_retrieval(rag_data):
    rag, _, _ = rag_data
    llm = AsyncMock()
    result = await AssistantService(rag, llm=llm).run(
        AssistantRequest(query="电源线有火花，应该如何拆开维修？")
    )
    assert result.status == "escalated"
    assert result.metadata["route_overrides"] == ["explicit_risk_preflight"]
    llm.assert_not_awaited()
    rag.aquery_data.assert_not_awaited()


async def test_multi_step_stops_at_three_queries_and_two_rounds(rag_data):
    rag, _, _ = rag_data
    llm = FlowLLM(strategy="diagnose", first_insufficient=True)
    result = await AssistantService(rag, llm=llm).run(
        AssistantRequest(query="L3251 清洗后仍然不行")
    )
    assert result.status == "answered"
    assert rag.aquery_data.await_count == 3
    assert [r["round"] for r in result.metadata["retrievals"]] == [1, 2, 2]
    assert llm.evidence_reviews == 2


async def test_multi_step_stops_early_when_first_evidence_is_sufficient(rag_data):
    rag, _, _ = rag_data
    result = await AssistantService(rag, llm=FlowLLM(strategy="diagnose")).run(
        AssistantRequest(query="L3251 清洗后仍然不行")
    )
    assert result.status == "answered"
    assert rag.aquery_data.await_count == 1


@pytest.mark.parametrize("strategy,count", [("retrieve_once", 1), ("diagnose", 3)])
async def test_insufficient_evidence_never_generates_plausible_device_answer(
    rag_data, strategy, count
):
    rag, _, _ = rag_data
    llm = FlowLLM(strategy=strategy, always_insufficient=True)
    result = await AssistantService(rag, llm=llm).run(
        AssistantRequest(query="L3251 怎么操作？")
    )
    assert result.status == "insufficient_evidence"
    assert rag.aquery_data.await_count == count
    assert not any(system.startswith("Write a useful") for system, _ in llm.calls)
    assert result.evidence == []


async def test_total_deadline_cancels_model_work(rag_data):
    rag, _, _ = rag_data
    cancelled = asyncio.Event()

    async def slow(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    result = await AssistantService(
        rag, llm=slow, config=AssistantConfig(total_timeout=0.02)
    ).run(AssistantRequest(query="你好"))
    assert result.status == "failed"
    assert result.error["code"] == "request_timeout"
    assert cancelled.is_set()
    rag.aquery_data.assert_not_awaited()


async def test_caller_cancellation_is_not_an_http_success(rag_data):
    rag, _, _ = rag_data
    task = asyncio.create_task(
        AssistantService(rag, llm=AsyncMock(side_effect=asyncio.CancelledError())).run(
            AssistantRequest(query="你好")
        )
    )
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_invalid_route_never_calls_tools(rag_data):
    rag, _, _ = rag_data
    result = await AssistantService(rag, llm=AsyncMock(return_value="broken")).run(
        AssistantRequest(query="你好")
    )
    assert result.status == "failed"
    assert result.error["code"] == "invalid_model_output"
    rag.aquery_data.assert_not_awaited()


async def test_unknown_evidence_ids_are_rejected_before_generation(rag_data):
    rag, _, _ = rag_data
    normal = FlowLLM()

    async def bad_review(prompt, **kwargs):
        if kwargs["system_prompt"].startswith("Evaluate the supplied"):
            return json.dumps(
                {
                    "accepted_evidence_ids": ["invented"],
                    "sufficient": True,
                    "missing_information": [],
                    "followup_queries": [],
                    "clarification_question": None,
                    "reason": "bad ID",
                }
            )
        return await normal(prompt, **kwargs)

    result = await AssistantService(rag, llm=bad_review).run(
        AssistantRequest(query="L3251 怎么操作？")
    )
    assert result.status == "failed"
    assert result.error["stage"] == "evidence_review"
    assert not any(system.startswith("Write a useful") for system, _ in normal.calls)


@pytest.mark.parametrize(
    "invalid", ["missing_citation", "invented_citation", "invented_number"]
)
async def test_ungrounded_claims_cannot_reach_answer(rag_data, invalid):
    rag, _, _ = rag_data
    normal = FlowLLM()

    async def bad_draft(prompt, **kwargs):
        if kwargs["system_prompt"].startswith("Write a useful"):
            payload = json.loads(prompt)
            key = payload["evidence"][0]["evidence_id"]
            return json.dumps(
                {
                    "claims": [
                        {
                            "text": "Some answer [99]"
                            if invalid == "invented_number"
                            else "Some answer",
                            "evidence_ids": []
                            if invalid == "missing_citation"
                            else [
                                "invented" if invalid == "invented_citation" else key
                            ],
                        }
                    ]
                }
            )
        return await normal(prompt, **kwargs)

    result = await AssistantService(rag, llm=bad_draft).run(
        AssistantRequest(query="L3251 怎么操作？")
    )
    assert result.status == "failed"
    assert result.answer == ""
    assert result.error["stage"] == "generation"


async def test_one_answer_revision_then_stop(rag_data):
    rag, _, _ = rag_data
    llm = FlowLLM(review_verdict="revise")
    result = await AssistantService(rag, llm=llm).run(
        AssistantRequest(query="L3251 怎么操作？")
    )
    assert result.status == "insufficient_evidence"
    assert llm.answer_reviews == 2
    assert rag.aquery_data.await_count == 1
    drafts = [
        payload for system, payload in llm.calls if system.startswith("Write a useful")
    ]
    assert drafts[0]["evidence"] == drafts[1]["evidence"]
    assert drafts[1]["revision_feedback"]


async def test_dangerous_draft_is_blocked_even_if_grader_would_pass(rag_data):
    rag, _, _ = rag_data
    normal = FlowLLM()

    async def unsafe(prompt, **kwargs):
        if kwargs["system_prompt"].startswith("Write a useful"):
            key = json.loads(prompt)["evidence"][0]["evidence_id"]
            return json.dumps(
                {
                    "claims": [
                        {"text": "请带电拆开电源进行检修。", "evidence_ids": [key]}
                    ]
                }
            )
        return await normal(prompt, **kwargs)

    result = await AssistantService(rag, llm=unsafe).run(
        AssistantRequest(query="L3251 怎么操作？")
    )
    assert result.status == "escalated"
    assert "请带电" not in result.model_dump_json()
    assert normal.answer_reviews == 0


async def test_audit_sink_gets_raw_outputs_but_write_failure_is_visible(rag_data):
    rag, _, _ = rag_data
    sink = AsyncMock()
    service = AssistantService(rag, llm=FlowLLM(), audit_sink=sink)
    result = await service.run(AssistantRequest(query="L3251 怎么操作？"))
    assert result.status == "answered"
    assert "raw_response" in sink.await_args.args[0]["model_trace"][0]
    sink.side_effect = OSError("sensitive path")
    result = await service.run(AssistantRequest(query="L3251 怎么操作？"))
    assert result.status == "failed"
    assert result.error["code"] == "audit_write_failed"
    assert "sensitive path" not in result.model_dump_json()


async def test_combined_evidence_budget_keeps_complete_groups(rag_data):
    rag, _, _ = rag_data
    llm = FlowLLM(always_insufficient=True)
    result = await AssistantService(
        rag, llm=llm, config=AssistantConfig(evidence_tokens=1)
    ).run(AssistantRequest(query="L3251 怎么操作？"))
    assert result.status == "insufficient_evidence"
    assert result.metadata["omitted_evidence_ids"]


async def test_query_role_is_resolved_at_call_time(rag_data):
    rag, _, _ = rag_data
    first, second = FlowLLM(), FlowLLM()
    rag.role_llm_funcs = {"query": first}
    service = AssistantService(rag)
    assert (
        await service.run(AssistantRequest(query="L3251 怎么操作？"))
    ).status == "answered"
    rag.role_llm_funcs = {"query": second}
    assert (
        await service.run(AssistantRequest(query="L3251 怎么操作？"))
    ).status == "answered"
    assert len(first.calls) == len(second.calls) == 4


@pytest.mark.parametrize(
    "kwargs", [{"max_retrievals": 4}, {"max_parallel_llm": 0}, {"total_timeout": -1}]
)
def test_execution_budgets_are_not_unlimited(kwargs):
    with pytest.raises(ValueError):
        AssistantConfig(**kwargs)
