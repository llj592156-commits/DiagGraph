"""Regeneration targets the prior task and never promotes old replies to evidence."""

import json
from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest

from lightrag.assistant import AssistantRequest, AssistantService
from lightrag.assistant import workflow as workflow_module
from tests.assistant.conftest import FlowLLM, route_response

pytestmark = pytest.mark.offline

HISTORY = [
    {"role": "user", "content": "打印有条纹，清洗一次还是不行"},
    {"role": "assistant", "content": "请问打印机是什么型号？"},
    {"role": "user", "content": "L4263"},
    {"role": "assistant", "content": "在维护菜单执行喷嘴检查，再根据检查图案处理。[1]"},
]
QUESTION = "L4263 打印有条纹，清洗一次后仍然不行，下一步如何检查和处理？"


def regeneration_route():
    return route_response(
        "diagnose",
        query="",
        intent="troubleshooting",
        resolved_query=QUESTION,
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
    )


@pytest.mark.parametrize("repeat_after_insufficient", [False, True])
async def test_cited_answer_regeneration_repairs_direct_route_and_rechecks_sources(
    rag_data, repeat_after_insufficient
):
    rag, _, _ = rag_data
    history = deepcopy(HISTORY)
    # Previous assistant output is untrusted even if it contains citation markers.
    history[3]["content"] += "未经资料支持的旧建议。"
    if repeat_after_insufficient:
        history.extend(
            [
                {"role": "user", "content": "请你再生成一遍"},
                {"role": "assistant", "content": "现有资料不足以可靠回答。"},
            ]
        )
    normal = FlowLLM()
    routing = 0
    downstream = []

    async def llm(prompt, **kwargs):
        nonlocal routing
        payload, _ = json.JSONDecoder().raw_decode(prompt)
        if kwargs["system_prompt"].startswith("You route"):
            routing += 1
            if routing == 1:
                return json.dumps(
                    route_response("direct", query="", intent="text_transform")
                )
            return json.dumps(regeneration_route())
        downstream.append(payload)
        return await normal(prompt, **kwargs)

    result = await AssistantService(rag, llm=llm).run(
        AssistantRequest(query="请你再生成一遍", history=history)
    )
    assert result.status == "answered"
    assert routing == 2
    assert rag.aquery_data.await_count == 1
    assert rag.aquery_data.await_args.args[0].startswith(QUESTION)
    assert result.citations
    assert "未经资料支持的旧建议" not in result.answer
    assert all(p["current_question"] == QUESTION for p in downstream)
    assert all(p["conversation_context"]["action"] == "regenerate" for p in downstream)
    assert all(
        p["conversation_context"]["target_user_message_index"] == 2 for p in downstream
    )
    assert all(
        p["conversation_context"]["target_assistant_message_index"] == 3
        for p in downstream
    )
    assert result.metadata["conversation_context"]["action"] == "regenerate"


@pytest.mark.parametrize(
    "text",
    ["请你再生成一遍", "重新回答一次", "再来一遍", "换个说法", "Please try again."],
)
def test_standalone_regeneration_commands_resolve_the_latest_complete_turn(text):
    from lightrag.assistant.conversation import conversation_context

    context = conversation_context(
        AssistantRequest(query=text, history=deepcopy(HISTORY))
    )
    assert context["action"] == "regenerate"
    assert context["target_user_message_index"] == 2
    assert context["requires_evidence_check"] is True


@pytest.mark.parametrize(
    "text", ["L4263 怎样重新生成喷嘴检查图案？", "重新回答，型号换成L3251", "你好"]
)
def test_new_business_details_are_not_swallowed_by_regeneration_control(text):
    from lightrag.assistant.conversation import conversation_context

    assert (
        conversation_context(AssistantRequest(query=text, history=deepcopy(HISTORY)))[
            "action"
        ]
        == "answer"
    )


async def test_regeneration_without_history_asks_for_a_target(rag_data):
    rag, _, _ = rag_data
    clarify = route_response(
        "clarify",
        query="",
        intent="unknown",
        missing_fields=["previous_question"],
        clarification_question="您希望我重新回答哪个问题？",
    )
    llm = AsyncMock(
        side_effect=[
            json.dumps(route_response("direct", query="", intent="general_chat")),
            json.dumps(clarify),
        ]
    )
    result = await AssistantService(rag, llm=llm).run(
        AssistantRequest(query="请你再生成一遍")
    )
    assert result.status == "needs_clarification"
    assert result.answer == "您希望我重新回答哪个问题？"
    rag.aquery_data.assert_not_awaited()


async def test_regenerated_greeting_stays_direct(rag_data):
    rag, _, _ = rag_data
    normal = FlowLLM(strategy="direct")
    result = await AssistantService(rag, llm=normal).run(
        AssistantRequest(
            query="再来一遍",
            history=[
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好，我可以帮助查找设备手册。"},
            ],
        )
    )
    assert result.status == "answered"
    generation = next(
        p for system, p in normal.calls if system.startswith("Write a useful")
    )
    assert generation["current_question"] == "你好"
    rag.aquery_data.assert_not_awaited()


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"always_insufficient": True}, "evidence_insufficient"),
        ({"review_verdict": "revise"}, "answer_revision_exhausted"),
    ],
)
async def test_insufficient_outcome_logs_the_actual_stopping_reason(
    rag_data, monkeypatch, kwargs, reason
):
    rag, _, _ = rag_data
    info = Mock()
    monkeypatch.setattr(workflow_module.logger, "info", info)
    result = await AssistantService(rag, llm=FlowLLM(**kwargs)).run(
        AssistantRequest(query="L3251 怎么操作？")
    )
    assert result.status == "insufficient_evidence"
    assert result.metadata["termination_reason"] == reason
    matching = [
        c for c in info.call_args_list if "Assistant request completed" in c.args[0]
    ]
    assert len(matching) == 1
    message = matching[0].args[0] % tuple(matching[0].args[1:])
    assert reason in message
    assert "L3251 怎么操作" not in message
