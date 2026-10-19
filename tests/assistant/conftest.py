"""Small in-memory retrieval and scripted model fixtures; no external calls."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def route_response(strategy="retrieve_once", *, query="L3251 怎么操作？", **updates):
    context = []
    if "L3251" in query:
        context = [
            {
                "field": "device_model",
                "value": "L3251",
                "message_index": -1,
                "quote": "L3251",
            }
        ]
    data = {
        "schema_version": "diaggraph-route-v1",
        "intent": "device_fact",
        "strategy": strategy,
        "resolved_query": query
        if strategy in {"retrieve_once", "diagnose"} and query
        else None,
        "extracted_context": context,
        "missing_fields": [],
        "clarification_question": None,
        "reason": "需要手册原文依据。",
        "risk_flags": [],
    }
    data.update(updates)
    return data


def make_rag(*, text="先打印喷嘴检查图案，再根据结果处理。"):
    rows = {
        "chunk-1": {"content": text, "full_doc_id": "doc-1", "file_path": "manual.pdf"}
    }

    async def get_rows(keys):
        return [deepcopy(rows.get(k)) for k in keys]

    raw = {
        "status": "success",
        "message": "ok",
        "data": {
            "entities": [],
            "relationships": [],
            "references": [],
            "chunks": [
                {
                    "chunk_id": "chunk-1",
                    "content": text,
                    "file_path": "manual.pdf",
                    "reference_id": "1",
                }
            ],
        },
        "metadata": {"retrieval": {"rerank_applied": True}},
    }
    rag = SimpleNamespace(
        workspace="test",
        tokenizer=SimpleNamespace(encode=lambda text: list(text)),
        text_chunks=SimpleNamespace(get_by_ids=AsyncMock(side_effect=get_rows)),
        full_docs=SimpleNamespace(get_by_ids=AsyncMock(return_value=[])),
        aquery_data=AsyncMock(side_effect=lambda *args, **kwargs: deepcopy(raw)),
    )
    return rag, rows, raw


class FlowLLM:
    def __init__(
        self,
        *,
        strategy="retrieve_once",
        first_insufficient=False,
        always_insufficient=False,
        review_verdict="pass",
    ):
        self.strategy, self.first_insufficient = strategy, first_insufficient
        self.always_insufficient, self.review_verdict = (
            always_insufficient,
            review_verdict,
        )
        self.calls, self.evidence_reviews, self.answer_reviews = [], 0, 0

    async def __call__(self, prompt, *, system_prompt, stream):
        payload = json.loads(prompt)
        self.calls.append((system_prompt, payload))
        if system_prompt.startswith("You route"):
            result = route_response(self.strategy, query=payload["query"])
            if self.strategy == "direct":
                result["intent"] = "general_chat"
            if self.strategy == "clarify":
                result.update(
                    missing_fields=["device_model"],
                    clarification_question="请问设备型号是什么？",
                )
            return json.dumps(result, ensure_ascii=False)
        if system_prompt.startswith("Evaluate the supplied"):
            self.evidence_reviews += 1
            sufficient = (
                bool(payload["evidence"])
                and not self.always_insufficient
                and not (self.first_insufficient and self.evidence_reviews == 1)
            )
            return json.dumps(
                {
                    "accepted_evidence_ids": [
                        e["evidence_id"] for e in payload["evidence"]
                    ],
                    "sufficient": sufficient,
                    "missing_information": [] if sufficient else ["后续步骤"],
                    "followup_queries": []
                    if sufficient
                    else ["L3251 喷嘴检查", "L3251 等待条件"],
                    "clarification_question": None,
                    "reason": "来源条件检查。",
                },
                ensure_ascii=False,
            )
        if system_prompt.startswith("Write a useful"):
            return json.dumps(
                {
                    "claims": [
                        {
                            "text": "请先打印喷嘴检查图案。"
                            if payload["evidence"]
                            else "你好，我可以协助查询设备手册。",
                            "evidence_ids": [payload["evidence"][0]["evidence_id"]]
                            if payload["evidence"]
                            else [],
                        }
                    ]
                },
                ensure_ascii=False,
            )
        self.answer_reviews += 1
        return json.dumps(
            {
                "verdict": self.review_verdict,
                "reason": "原文支持已检查。",
                "clarification_question": "请补充设备情况。"
                if self.review_verdict == "clarify"
                else None,
            },
            ensure_ascii=False,
        )


@pytest.fixture
def rag_data():
    return make_rag()
