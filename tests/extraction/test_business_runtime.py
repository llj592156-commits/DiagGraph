"""Business identity, validation and cache reconstruction use one conversion."""

import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from lightrag.extraction import printer
from lightrag.extraction.printer_examples import worked_examples
from lightrag.extraction.runtime import build_records, chunk_context, snapshot_config
from lightrag.operate import _rebuild_from_extraction_result, extract_entities
from tests.extraction.test_extract_entities import _make_global_config

pytestmark = pytest.mark.offline


def fixture_row(doc="doc-B", models="L3266_L5298", heading="网络连接报告", parent=None):
    row = {
        "full_doc_id": doc,
        "file_path": models + ".pdf",
        "content": worked_examples()[0]["source_text"],
        "llm_cache_list": [],
        "extraction_config": snapshot_config("printer_v2"),
    }
    if parent:
        row["parent_child"] = {
            "doc_id": doc,
            "parent_id": parent,
            "heading_path": [heading],
            "feature_domain": "network",
            "scope": {"status": "filename", "applicable_models": ["L3266", "L5298"]},
        }
    return row, chunk_context(row, "chunk-one", row["extraction_config"], heading)


def convert(context, output=None):
    example = worked_examples()[0]
    return build_records(
        example["source_text"],
        context,
        json.dumps(output or example["output"], ensure_ascii=False),
        1,
    )


def test_config_is_explicit_and_contract_changes_fail_closed():
    assert snapshot_config() == {"profile": "generic"}
    with pytest.raises(ValueError, match="extraction_profile"):
        snapshot_config("typo")
    _, context = fixture_row()
    context["config"]["prompt_sha256"] = "old"
    with pytest.raises(ValueError, match="contract changed"):
        convert(context)


def test_over_limit_output_quarantines_surplus_and_dependent_relations():
    _, context = fixture_row()
    text = "E-12 " + " ".join(f"Step{i}" for i in range(1, 16))
    entities = [
        {
            "name": "E-12",
            "type": "FaultCode",
            "description": "Code",
            "evidence_quotes": ["E-12"],
        }
    ]
    entities += [
        {
            "name": f"Step{i}",
            "type": "Procedure",
            "description": "Action",
            "evidence_quotes": [f"Step{i}"],
        }
        for i in range(1, 16)
    ]
    relationships = [
        {
            "source": "E-12",
            "target": name,
            "relation": "HANDLED_BY",
            "description": "Action",
            "evidence_quotes": ["E-12", name],
            "qualifiers": [],
        }
        for name in ["Step1", "Step15"]
    ]
    raw = json.dumps({"entities": entities, "relationships": relationships})
    nodes, edges, evidence = build_records(text, context, raw, 1)
    assert len(nodes) == 12 and len(edges) == 1
    assert len(evidence["review"]["rejected"]) == 5
    assert not any(k.startswith("Step15 ") for k in nodes)
    response = {"choices": [{"finish_reason": "stop", "message": {"content": raw}}]}
    assert printer.review_response(response, {"input_text": text})["sample_error"]


def test_environment_and_constructor_config_have_explicit_precedence(monkeypatch):
    from lightrag.addon_params import default_addon_params, normalize_addon_params

    monkeypatch.delenv("EXTRACTION_PROFILE", raising=False)
    assert default_addon_params()["extraction_profile"] == "generic"
    monkeypatch.setenv("EXTRACTION_PROFILE", "printer_v2")
    assert normalize_addon_params({})["extraction_profile"] == "printer_v2"
    assert (
        normalize_addon_params({"extraction_profile": "generic"})["extraction_profile"]
        == "generic"
    )


def test_explicit_model_support_is_not_spread_to_filename_siblings():
    _, context = fixture_row()
    source = "L5298 支持传真。"
    value = {
        "entities": [
            {
                "name": "L5298",
                "type": "DeviceModel",
                "description": "设备型号",
                "evidence_quotes": ["L5298"],
            },
            {
                "name": "传真",
                "type": "Feature",
                "description": "传真功能",
                "evidence_quotes": ["传真"],
            },
        ],
        "relationships": [
            {
                "source": "L5298",
                "target": "传真",
                "relation": "SUPPORTS",
                "description": source,
                "evidence_quotes": [source],
                "qualifiers": [],
            }
        ],
    }
    _, _, evidence = build_records(source, context, json.dumps(value), 1)
    assert evidence["facts"][0]["models"] == ["L5298"]
    assert context["scope"]["applicable_models"] == ["L3266", "L5298"]


def test_same_code_separates_documents_scenes_types_and_model_scopes():
    _, baseline = fixture_row()
    variants = [
        baseline,
        {**baseline, "doc_id": "doc-C"},
        {**baseline, "domain": "device_display_errors"},
        {**baseline, "scope": {"applicable_models": ["L5298"]}},
    ]
    keys = [next(k for k in convert(c)[0] if k.startswith("E-12 ")) for c in variants]
    assert len(set(keys)) == 4
    repeat = {**baseline, "chunk_id": "other-chunk"}
    assert next(k for k in convert(repeat)[0] if k.startswith("E-12 ")) == keys[0]
    unknown = {**baseline, "domain": "unknown"}
    other = {**unknown, "chunk_id": "other-chunk"}
    assert set(convert(unknown)[0]).isdisjoint(convert(other)[0])


def test_non_parent_chunks_keep_real_sources_without_fabricating_a_parent():
    row, context = fixture_row()
    assert context["parent_id"] is None
    nodes, edges, evidence = convert(context)
    assert all(pair[0] in nodes and pair[1] in nodes for pair in edges)
    assert {item[0]["source_id"] for item in nodes.values()} == {"chunk-one"}
    assert evidence["context"]["doc_id"] == row["full_doc_id"]
    assert evidence["facts"][0]["record"]["evidence_quotes"]
    assert evidence["semantic_review_required"] is True


@pytest.mark.parametrize("mode", ["only", "excluded", "conflict", "unresolved"])
def test_fact_model_limit_does_not_narrow_parent_scope(mode):
    _, context = fixture_row(parent="parent-one")
    before = deepcopy(context)
    example = deepcopy(worked_examples()[0])
    quotes = {
        "only": "仅对于 L5298",
        "excluded": "L5298 除外",
        "conflict": "仅对于 L9999",
        "unresolved": "适用型号待确认",
    }
    quote = quotes[mode]
    example["source_text"] += "\n" + quote
    example["output"]["relationships"][0]["qualifiers"] = [
        {
            "kind": "model_restriction",
            "text": quote,
            "evidence_quotes": [quote],
        }
    ]
    _, _, evidence = build_records(
        example["source_text"], context, json.dumps(example["output"]), 1
    )
    assert context == before
    if mode in {"conflict", "unresolved"}:
        assert evidence["review"]["rejected"]
        assert len(evidence["facts"]) == 2
    else:
        assert evidence["facts"][0]["models"] == (
            ["L5298"] if mode == "only" else ["L3266"]
        )


@pytest.mark.asyncio
async def test_business_extraction_preserves_new_cache_refs_and_rebuilds_same_keys(
    monkeypatch,
):
    from lightrag import operate

    row, _ = fixture_row()
    stored = deepcopy(row)
    db = AsyncMock()
    db.get_by_id.side_effect = lambda _: deepcopy(stored)

    async def upsert(data):
        stored.update(data["chunk-one"])

    db.upsert.side_effect = upsert
    raw = json.dumps(worked_examples()[0]["output"], ensure_ascii=False)
    calls = []

    async def cached(prompt, *_args, **kwargs):
        calls.append(kwargs)
        stored["llm_cache_list"] = ["new-reference"]
        return raw, 1

    monkeypatch.setattr(operate, "use_llm_func_with_cache", cached)
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "100000")
    result = await extract_entities(
        {"chunk-one": row}, _make_global_config(), text_chunks_storage=db
    )
    assert len(calls) == 1  # Business JSON has no implicit generic gleaning call.
    assert calls[0]["system_prompt"] == printer.system_prompt()
    assert stored["llm_cache_list"] == ["new-reference"]
    assert stored["business_extraction"]["context"]["parent_id"] is None
    rebuilt = await _rebuild_from_extraction_result(db, raw, "chunk-one", 1)
    assert rebuilt == result[0]


@pytest.mark.asyncio
async def test_managed_identity_rename_and_merge_refuse_before_storage_changes():
    from lightrag.utils_graph import _edit_entity_impl, _merge_entities_impl

    graph, entities, relations = AsyncMock(), AsyncMock(), AsyncMock()
    name = "E-12 ⟦printer:abc⟧"
    with pytest.raises(ValueError, match="source-managed"):
        await _edit_entity_impl(
            graph, entities, relations, name, {"entity_name": "E-12"}
        )
    with pytest.raises(ValueError, match="cannot be manually merged"):
        await _merge_entities_impl(graph, entities, relations, [name], "E-12")
    assert not graph.mock_calls and not entities.mock_calls and not relations.mock_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("truncated", [False, True])
async def test_failed_output_keeps_raw_attempt_without_native_graph_records(
    monkeypatch, truncated
):
    from lightrag import operate
    from lightrag.utils import TruncatedResponse

    row, _ = fixture_row()
    stored = deepcopy(row)
    db = AsyncMock()
    db.get_by_id.side_effect = lambda _: deepcopy(stored)

    async def upsert(data):
        stored.update(data["chunk-one"])

    db.upsert.side_effect = upsert
    raw = TruncatedResponse('{"entities":') if truncated else "invalid JSON"
    monkeypatch.setattr(
        operate, "use_llm_func_with_cache", AsyncMock(return_value=(raw, 1))
    )
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "100000")
    result = await extract_entities(
        {"chunk-one": row}, _make_global_config(), text_chunks_storage=db
    )
    assert result == [({}, {})]
    assert stored["business_extraction_attempt"]["raw_response"] == str(raw)
    assert stored["business_extraction_attempt"]["truncated"] is truncated
    assert stored["business_extraction"]["review"]["skipped"] is True
    assert stored["business_extraction"]["review"]["sample_error"]
    if not truncated:
        assert await _rebuild_from_extraction_result(db, str(raw), "chunk-one", 1) == (
            {},
            {},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["provider", "storage", "transport_storage", "auth"]
)
async def test_skip_policy_does_not_swallow_provider_or_storage_failures(
    monkeypatch, failure
):
    from lightrag import operate
    from lightrag.extraction.errors import (
        ExtractionRequestError,
        ExtractionRetriesExhausted,
    )

    row, _ = fixture_row()
    db = AsyncMock()
    db.get_by_id.return_value = deepcopy(row)
    provider = AsyncMock(return_value=("invalid JSON", 1))
    if failure == "provider":
        provider.side_effect = RuntimeError("provider unavailable")
    elif failure == "auth":
        provider.side_effect = ExtractionRequestError(
            "auth unavailable", retryable=False, status=401
        )
    elif failure == "transport_storage":
        provider.side_effect = ExtractionRetriesExhausted(
            ExtractionRequestError("temporary timeout", retryable=True), 3
        )
        db.upsert.side_effect = RuntimeError("transport_storage unavailable")
    else:
        db.upsert.side_effect = RuntimeError("storage unavailable")
    monkeypatch.setattr(operate, "use_llm_func_with_cache", provider)
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "100000")
    with pytest.raises(RuntimeError, match=f"{failure} unavailable"):
        await extract_entities(
            {"chunk-one": row}, _make_global_config(), text_chunks_storage=db
        )
