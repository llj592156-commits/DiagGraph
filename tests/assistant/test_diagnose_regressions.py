"""Pin failures observed in a user-run diagnosis; all services are simulated."""

import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from lightrag.assistant import AssistantConfig, AssistantRequest, AssistantService
from lightrag.assistant.schemas import EvidenceItem, RetrievalBatch
from tests.assistant.conftest import FlowLLM
from tests.assistant.test_retrieval import parent_fixture

pytestmark = pytest.mark.offline

QUERY = "L3251 打印有条纹，清洗一次后仍然存在"


def review_response(payload, *, question=None, queries=(), sufficient=False):
    return json.dumps(
        {
            "accepted_evidence_ids": [e["evidence_id"] for e in payload["evidence"]],
            "sufficient": sufficient,
            "missing_information": [] if sufficient else ["条纹方向"],
            "followup_queries": list(queries),
            "clarification_question": question,
            "reason": "区分用户观察与手册知识。",
        },
        ensure_ascii=False,
    )


def diagnosis_llm(review, *, action_index=-1):
    normal = FlowLLM(strategy="diagnose")

    async def llm(prompt, **kwargs):
        payload, _ = json.JSONDecoder().raw_decode(prompt)
        if kwargs["system_prompt"].startswith("You route"):
            result = json.loads(await normal(prompt, **kwargs))
            result["extracted_context"].append(
                {
                    "field": "attempted_action",
                    "value": "清洗一次",
                    "quote": "清洗一次",
                    "message_index": action_index,
                }
            )
            return json.dumps(result, ensure_ascii=False)
        if kwargs["system_prompt"].startswith("Evaluate the supplied"):
            return review(payload)
        return await normal(prompt, **kwargs)

    return llm


async def test_user_observation_stops_before_followup_retrieval(rag_data):
    rag, _, _ = rag_data
    question = "条纹是横向还是纵向？"
    llm = diagnosis_llm(
        lambda payload: review_response(
            payload,
            question=question,
            queries=["打印条纹方向的判断方法", "您是否打印过喷嘴检查图案？"],
        )
    )
    result = await AssistantService(rag, llm=llm).run(AssistantRequest(query=QUERY))
    assert result.status == "needs_clarification"
    assert result.answer == question
    assert rag.aquery_data.await_count == 1
    assert len(result.metadata["evidence_reviews"]) == 1
    assert result.metadata["model_calls"] == 2


@pytest.mark.parametrize(
    "bad_kind",
    ["repeat_count", "user_question_as_search", "stripe_observation_as_search"],
)
async def test_invalid_gap_decision_is_repaired_before_execution(rag_data, bad_kind):
    rag, _, _ = rag_data
    calls = 0

    def review(payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            if bad_kind == "repeat_count":
                return review_response(payload, question="您总共清洗了几次？")
            if bad_kind == "stripe_observation_as_search":
                return review_response(
                    payload, queries=["打印出的条纹是水平方向还是垂直方向？"]
                )
            return review_response(payload, queries=["您是否打印过喷嘴检查图案？"])
        return review_response(payload, question="条纹是横向还是纵向？")

    result = await AssistantService(rag, llm=diagnosis_llm(review)).run(
        AssistantRequest(query=QUERY)
    )
    assert result.status == "needs_clarification"
    assert result.answer == "条纹是横向还是纵向？"
    assert rag.aquery_data.await_count == 1
    assert calls == 2
    assert result.metadata["model_trace"][1]["status"] == "invalid_response"


async def test_unrepaired_known_count_question_is_not_published(rag_data):
    rag, _, _ = rag_data
    llm = diagnosis_llm(
        lambda payload: review_response(payload, question="清洗了多少次？")
    )
    result = await AssistantService(rag, llm=llm).run(AssistantRequest(query=QUERY))
    assert result.status == "failed"
    assert result.answer == ""
    assert result.error["code"] == "invalid_model_output"
    assert result.error["stage"] == "evidence_review"
    assert rag.aquery_data.await_count == 1


async def test_known_count_in_user_history_is_not_asked_again(rag_data):
    rag, _, _ = rag_data
    llm = diagnosis_llm(
        lambda payload: review_response(payload, question="您清洗了几次？"),
        action_index=0,
    )
    result = await AssistantService(rag, llm=llm).run(
        AssistantRequest(
            query="L3251 打印还是有条纹",
            history=[{"role": "user", "content": "L3251 清洗一次后仍然存在条纹"}],
        )
    )
    assert result.status == "failed"
    assert result.answer == ""
    assert result.error["code"] == "invalid_model_output"
    assert rag.aquery_data.await_count == 1


async def test_known_count_does_not_prevent_asking_unknown_cleaning_type(rag_data):
    rag, _, _ = rag_data
    question = "您执行的是普通清洗还是强力清洗？"
    result = await AssistantService(
        rag,
        llm=diagnosis_llm(lambda payload: review_response(payload, question=question)),
    ).run(AssistantRequest(query=QUERY))
    assert result.status == "needs_clarification"
    assert result.answer == question
    assert rag.aquery_data.await_count == 1


async def test_location_metadata_does_not_displace_source_text_or_scope(rag_data):
    rag, rows, _, _ = parent_fixture(rag_data)
    for row in rows.values():
        row["parent_child"].update(
            source_positions=[{"positions": [{"bbox": list(range(10000))}]}],
            source_spans=[{"block_id": "location" * 2000}],
            conditions={"platforms": ["Windows"]},
            required_notes=["Wait before the next operation."],
        )
    original = deepcopy(rows)
    llm = FlowLLM()
    result = await AssistantService(rag, llm=llm).run(
        AssistantRequest(query="L3251 怎么操作？")
    )
    assert result.status == "answered"
    assert not result.metadata["omitted_evidence_ids"]
    projected = [payload["evidence"] for _, payload in llm.calls[1:]]
    assert projected[0] == projected[1] == projected[2]
    item = projected[0][0]
    assert item["content"] == result.evidence[0].content
    assert len(item["sources"]) == 2
    source = item["sources"][0]["parent_child"]
    assert "source_positions" not in source
    assert "source_spans" not in source
    assert source["required_notes"] == ["Wait before the next operation."]
    assert source["conditions"]["platforms"] == ["Windows"]
    assert item["parents"][0]["metadata"]["scope"]["applicable_models"] == ["L3251"]
    assert item["parents"][1]["metadata"]["scope"]["applicable_models"] == [
        "L3251",
        "L4263",
    ]
    assert result.evidence[0].sources[0].parent_child["source_positions"]
    selection = result.metadata["evidence_selection"][0]
    assert selection["retained_evidence_ids"] == [result.evidence[0].evidence_id]
    assert selection["model_evidence_tokens"] < 8000
    assert rows == original


async def test_new_gap_evidence_can_replace_unaccepted_old_evidence(rag_data):
    rag, _, _ = rag_data
    old = EvidenceItem(
        evidence_id="old",
        content="旧资料" * 50,
        doc_id="d",
        file_path="manual.pdf",
        source_chunk_ids=["old"],
        sources=[],
    )
    new = old.model_copy(update={"evidence_id": "new", "content": "新资料" * 50})
    llm = FlowLLM(strategy="diagnose")
    reviews = []

    async def model(prompt, **kwargs):
        if kwargs["system_prompt"].startswith("Evaluate the supplied"):
            payload = json.loads(prompt)
            ids = [e["evidence_id"] for e in payload["evidence"]]
            reviews.append(ids)
            sufficient = "new" in ids
            result = json.loads(
                review_response(
                    payload,
                    queries=[] if sufficient else ["L3251 是否支持喷嘴检查"],
                    sufficient=sufficient,
                )
            )
            if not sufficient:
                result["accepted_evidence_ids"] = []
            return json.dumps(result)
        return await llm(prompt, **kwargs)

    # Each group fits alone, while the combined groups exceed this budget.
    budget = len(json.dumps([old.model_dump()], ensure_ascii=False)) + 20
    service = AssistantService(
        rag, llm=model, config=AssistantConfig(evidence_tokens=budget)
    )
    service.adapter.retrieve = AsyncMock(
        side_effect=[
            RetrievalBatch(evidence=[old], raw_data={}),
            RetrievalBatch(evidence=[old, new], raw_data={}),
        ]
    )
    result = await service.run(AssistantRequest(query="L3251 怎么操作？"))
    assert result.status == "answered"
    assert reviews == [["old"], ["new"]]
    assert result.evidence[0].evidence_id == "new"
