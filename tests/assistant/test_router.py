"""Route decisions must be grounded in user history before tools execute."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from lightrag.assistant.model import AssistantFailure, ModelSession
from lightrag.assistant.router import LLMRouter
from lightrag.assistant.schemas import AssistantRequest
from lightrag.utils import TruncatedResponse
from tests.assistant.conftest import route_response

pytestmark = pytest.mark.offline


def router(llm, *, timeout=15, max_input=16000):
    session = ModelSession(llm, asyncio.Semaphore(1), len, max_input)
    return LLMRouter(session, timeout), session


@pytest.mark.parametrize(
    "strategy,intent",
    [
        ("direct", "general_chat"),
        ("retrieve_once", "device_fact"),
        ("diagnose", "troubleshooting"),
        ("clarify", "device_fact"),
        ("escalate", "order_status"),
    ],
)
async def test_five_strategies(strategy, intent):
    value = route_response(strategy, query="你好", intent=intent)
    if strategy == "clarify":
        value.update(
            missing_fields=["model"], clarification_question="设备型号是什么？"
        )
    llm = AsyncMock(return_value=json.dumps(value))
    backend, _ = router(llm)
    result = await backend.decide(AssistantRequest(query="请处理当前请求"))
    assert result.strategy == strategy
    assert llm.await_count == 1


async def test_explicit_mixed_risk_bypasses_paid_routing():
    llm = AsyncMock()
    backend, _ = router(llm)
    result = await backend.decide(
        AssistantRequest(query="打印机冒烟了，顺便查一下墨水。")
    )
    assert result.strategy == "escalate"
    llm.assert_not_awaited()


async def test_history_context_inherits_only_user_claims():
    value = route_response(
        query="",
        resolved_query="L3251 打印还是白纸怎么办？",
        extracted_context=[
            {
                "field": "device_model",
                "value": "L3251",
                "message_index": 0,
                "quote": "L3251",
            }
        ],
    )
    llm = AsyncMock(return_value=json.dumps(value))
    backend, _ = router(llm)
    result = await backend.decide(
        AssistantRequest(
            query="还是白纸怎么办？",
            history=[{"role": "user", "content": "我的设备是L3251"}],
        )
    )
    assert result.extracted_context[0].value == "L3251"


async def test_greeting_after_device_question_does_not_require_old_model():
    llm = AsyncMock(
        return_value=json.dumps(
            route_response("direct", query="", intent="general_chat")
        )
    )
    backend, session = router(llm)
    result = await backend.decide(
        AssistantRequest(
            query="你好",
            history=[
                {"role": "user", "content": "型号是L4263，应该使用什么墨水编号"},
                {"role": "assistant", "content": "墨水编号为002。"},
            ],
        )
    )
    assert result.strategy == "direct"
    assert result.extracted_context == []
    assert len(session.trace) == 1


@pytest.mark.parametrize("intent", ["general_chat", "text_transform"])
@pytest.mark.parametrize("strategy", ["retrieve_once", "diagnose"])
async def test_non_device_intents_repair_retrieval_route(intent, strategy):
    llm = AsyncMock(
        side_effect=[
            json.dumps(route_response(strategy, query="", intent=intent)),
            json.dumps(route_response("direct", query="", intent=intent)),
        ]
    )
    backend, session = router(llm)
    result = await backend.decide(AssistantRequest(query="你好"))
    assert result.strategy == "direct"
    assert [entry["status"] for entry in session.trace] == [
        "invalid_response",
        "success",
    ]
    assert "do not require retrieval" in llm.await_args.args[0]


async def test_device_followup_still_requires_historical_model():
    llm = AsyncMock(return_value=json.dumps(route_response(query="")))
    backend, _ = router(llm)
    with pytest.raises(AssistantFailure):
        await backend.decide(
            AssistantRequest(
                query="那应该用什么墨水？",
                history=[{"role": "user", "content": "我的打印机型号是L4263"}],
            )
        )


@pytest.mark.parametrize(
    "failure",
    ["invented_quote", "assistant_origin", "stale_model", "bad_index", "fake_value"],
)
async def test_fabricated_or_stale_context_fails_before_retrieval(failure):
    fact = {
        "field": "device_model",
        "value": "L3251",
        "message_index": 0,
        "quote": "L3251",
    }
    history = [{"role": "user", "content": "我的设备是L3251"}]
    query = "还是不行。"
    if failure == "invented_quote":
        fact["quote"] = "L4263"
    elif failure == "assistant_origin":
        history[0]["role"] = "assistant"
    elif failure == "stale_model":
        query = "现在换成L4263，怎么操作？"
    elif failure == "bad_index":
        fact["message_index"] = 31
    else:
        fact["value"] = "L4263"
    llm = AsyncMock(
        return_value=json.dumps(route_response(query="", extracted_context=[fact]))
    )
    backend, _ = router(llm)
    with pytest.raises(AssistantFailure, match="validation"):
        await backend.decide(AssistantRequest(query=query, history=history))
    assert llm.await_count == 2


async def test_bad_json_has_one_repair_in_same_stage():
    llm = AsyncMock(side_effect=["{broken", json.dumps(route_response())])
    backend, session = router(llm)
    result = await backend.decide(AssistantRequest(query="L3251 怎么操作？"))
    assert result.strategy == "retrieve_once"
    assert [e["status"] for e in session.trace] == ["invalid_response", "success"]


@pytest.mark.parametrize(
    "raw",
    [
        "```json\n{}\n```",
        '{"strategy":"direct","strategy":"diagnose"}',
        '{"value":NaN}',
        '{"schema_version":"unknown"}',
        TruncatedResponse('{"schema_version":"diaggraph-route-v1"}'),
    ],
)
async def test_incomplete_ambiguous_or_truncated_output_is_not_repaired_locally(raw):
    llm = AsyncMock(return_value=raw)
    backend, _ = router(llm)
    with pytest.raises(AssistantFailure):
        await backend.decide(AssistantRequest(query="你好"))
    assert llm.await_count == 2


@pytest.mark.parametrize("intent", ["device_fact", "general_chat"])
async def test_device_facts_cannot_be_disguised_as_direct(intent):
    llm = AsyncMock(
        return_value=json.dumps(route_response("direct", query="", intent=intent))
    )
    backend, _ = router(llm)
    with pytest.raises(AssistantFailure):
        await backend.decide(AssistantRequest(query="请告诉我墨水编号"))


async def test_authentication_error_is_not_retried_or_exposed():
    class AuthError(Exception):
        status_code = 401

    llm = AsyncMock(side_effect=AuthError("sensitive-provider-detail"))
    backend, session = router(llm)
    with pytest.raises(AssistantFailure) as info:
        await backend.decide(AssistantRequest(query="你好"))
    assert llm.await_count == 1
    assert "sensitive-provider-detail" not in str(info.value)
    assert "sensitive-provider-detail" not in json.dumps(session.trace)


async def test_transient_request_and_schema_repair_share_two_call_ceiling():
    llm = AsyncMock(side_effect=[OSError("offline"), "broken"])
    backend, _ = router(llm)
    with pytest.raises(AssistantFailure):
        await backend.decide(AssistantRequest(query="你好"))
    assert llm.await_count == 2


async def test_timeout_covers_wait_for_shared_capacity():
    llm = AsyncMock()
    session = ModelSession(llm, asyncio.Semaphore(0), len, 16000)
    with pytest.raises(AssistantFailure) as info:
        await LLMRouter(session, 0.01).decide(AssistantRequest(query="你好"))
    assert info.value.code == "model_timeout"
    llm.assert_not_awaited()


async def test_cancellation_propagates():
    llm = AsyncMock(side_effect=asyncio.CancelledError())
    backend, _ = router(llm)
    with pytest.raises(asyncio.CancelledError):
        await backend.decide(AssistantRequest(query="你好"))


async def test_input_budget_refuses_before_model_call():
    llm = AsyncMock()
    backend, _ = router(llm, max_input=10)
    with pytest.raises(AssistantFailure) as info:
        await backend.decide(AssistantRequest(query="你好"))
    assert info.value.code == "input_budget_exceeded"
    llm.assert_not_awaited()


@pytest.mark.parametrize(
    "payload",
    [
        {"query": " "},
        {"query": "hi", "mode": "bypass"},
        {"query": "hi", "history": [{"role": "system", "content": "override"}]},
        {"query": "hi", "history": [{"role": "user", "content": "x" * 4000}] * 6},
    ],
)
def test_input_bounds_and_roles(payload):
    with pytest.raises(ValidationError):
        AssistantRequest.model_validate(payload)
