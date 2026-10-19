"""Resolve conversational follow-ups without mistaking questions for user facts."""

import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from lightrag.assistant import AssistantRequest, AssistantService
from lightrag.assistant.model import ModelSession
from lightrag.assistant.router import LLMRouter
from lightrag.assistant.schemas import RouteDecision
from tests.assistant.conftest import FlowLLM, route_response

pytestmark = pytest.mark.offline

HISTORY = [
    {"role": "user", "content": "打印有条纹，清洗一次还是不行"},
    {"role": "assistant", "content": "请问您使用的是哪款打印机型号？"},
    {"role": "user", "content": "L4263"},
    {
        "role": "assistant",
        "content": "请问您执行过喷嘴检查吗？图案中是否有断线或缺失？墨仓中的墨水液位是否充足？",
    },
]
RESOLVED = "L4263 如何执行喷嘴检查并查看墨仓中的墨水液位？"
PROCEDURE = "在操作面板选择维护，再选择喷嘴检查，按照屏幕提示装纸并打印检查图案。"


def followup_route(**updates):
    return route_response(
        query="",
        resolved_query=RESOLVED,
        extracted_context=[
            {
                "field": "device_model",
                "value": "L4263",
                "quote": "L4263",
                "message_index": 2,
            },
            {
                "field": "symptom",
                "value": "打印有条纹",
                "quote": "打印有条纹",
                "message_index": 0,
            },
            {
                "field": "attempted_action",
                "value": "清洗一次",
                "quote": "清洗一次",
                "message_index": 0,
            },
        ],
        **updates,
    )


async def test_resolved_followup_reaches_retrieval_and_all_answer_stages(rag_data):
    rag, rows, raw = rag_data
    procedure = PROCEDURE + "目视检查墨仓中的墨水液位。"
    rows["chunk-1"]["content"] = procedure
    raw["data"]["chunks"][0]["content"] = procedure

    async def search(query, **kwargs):
        result = deepcopy(raw)
        if "喷嘴检查" not in query or "墨水液位" not in query:
            result["data"]["chunks"] = []
        return result

    rag.aquery_data.side_effect = search
    normal = FlowLLM()
    stages = []

    async def model(prompt, **kwargs):
        payload = json.loads(prompt)
        if kwargs["system_prompt"].startswith("You route"):
            return json.dumps(followup_route(), ensure_ascii=False)
        stages.append(payload)
        if kwargs["system_prompt"].startswith("Write a useful"):
            return json.dumps(
                {
                    "claims": [
                        {
                            "text": payload["evidence"][0]["content"],
                            "evidence_ids": [payload["evidence"][0]["evidence_id"]],
                        }
                    ]
                },
                ensure_ascii=False,
            )
        return await normal(prompt, **kwargs)

    result = await AssistantService(rag, llm=model).run(
        AssistantRequest(query="怎么进行检查", history=deepcopy(HISTORY))
    )
    assert result.status == "answered"
    assert PROCEDURE in result.answer
    assert "目视检查墨仓" in result.answer
    assert len(result.citations) == 1
    assert rag.aquery_data.await_count == 1
    actual_query = rag.aquery_data.await_args.args[0]
    assert actual_query.startswith(RESOLVED)
    assert "attempted_action: 清洗一次" in actual_query
    assert all(payload["current_question"] == RESOLVED for payload in stages)
    assert all(payload["request"]["query"] == "怎么进行检查" for payload in stages)
    assert result.route.resolved_query == RESOLVED
    assert result.metadata["model_calls"] == 4


async def test_missing_followup_resolution_is_repaired_in_existing_routing_call():
    invalid = followup_route()
    invalid.pop("resolved_query")
    llm = AsyncMock(side_effect=[json.dumps(invalid), json.dumps(followup_route())])
    session = ModelSession(llm, asyncio.Semaphore(1), len, 16000)
    result = await LLMRouter(session).decide(
        AssistantRequest(query="怎么进行检查", history=deepcopy(HISTORY))
    )
    assert result.resolved_query == RESOLVED
    assert [entry["status"] for entry in session.trace] == [
        "invalid_response",
        "success",
    ]
    assert "resolved_query" in llm.await_args.args[0]


async def test_assistant_proposed_check_is_a_topic_not_a_completed_user_action():
    llm = AsyncMock(return_value=json.dumps(followup_route()))
    session = ModelSession(llm, asyncio.Semaphore(1), len, 16000)
    result = await LLMRouter(session).decide(
        AssistantRequest(query="怎么进行检查", history=deepcopy(HISTORY))
    )
    assert "喷嘴检查" in result.resolved_query
    assert [
        fact.value
        for fact in result.extracted_context
        if fact.field == "attempted_action"
    ] == ["清洗一次"]


@pytest.mark.parametrize("resolved", ["", "   ", "x" * 2001, 123])
def test_invalid_resolved_question_is_rejected(resolved):
    value = followup_route()
    value["resolved_query"] = resolved
    with pytest.raises(ValidationError):
        RouteDecision.model_validate(value)


async def test_unresolved_followup_never_retrieves_raw_short_question(rag_data):
    rag, _, _ = rag_data
    value = followup_route()
    value.pop("resolved_query")
    result = await AssistantService(
        rag, llm=AsyncMock(return_value=json.dumps(value))
    ).run(AssistantRequest(query="怎么进行检查", history=deepcopy(HISTORY)))
    assert result.status == "failed"
    assert result.error["stage"] == "routing"
    rag.aquery_data.assert_not_awaited()


@pytest.mark.parametrize("resolved", ["如何执行喷嘴检查？", "L3251 如何执行喷嘴检查？"])
async def test_rewrite_cannot_drop_or_replace_confirmed_model(rag_data, resolved):
    rag, _, _ = rag_data
    value = followup_route()
    value["resolved_query"] = resolved
    result = await AssistantService(
        rag, llm=AsyncMock(return_value=json.dumps(value))
    ).run(AssistantRequest(query="怎么进行检查", history=deepcopy(HISTORY)))
    assert result.status == "failed"
    assert result.error["stage"] == "routing"
    rag.aquery_data.assert_not_awaited()


async def test_rewrite_does_not_authorize_facts_sourced_from_assistant(rag_data):
    rag, _, _ = rag_data
    value = followup_route()
    value["extracted_context"].append(
        {
            "field": "attempted_action",
            "value": "喷嘴检查",
            "quote": "喷嘴检查",
            "message_index": 3,
        }
    )
    result = await AssistantService(
        rag, llm=AsyncMock(return_value=json.dumps(value))
    ).run(AssistantRequest(query="怎么进行检查", history=deepcopy(HISTORY)))
    assert result.status == "failed"
    assert result.error["stage"] == "routing"
    rag.aquery_data.assert_not_awaited()
