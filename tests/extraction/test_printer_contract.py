"""Business schema plumbing; these checks are not model quality evaluation."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from lightrag.extraction import printer
from lightrag.prompt import (
    PROMPTS,
    load_entity_extraction_prompt_profile,
    resolve_entity_extraction_prompt_profile,
)

pytestmark = pytest.mark.offline


def sample():
    return {
        "input_text": "E-12：请联系爱普生认证服务机构更换。它不能由用户更换。",
        "document_name": "L3266_L5298.pdf",
        "heading_path": ["显示屏上显示错误代码"],
        "domain": "device_display_errors",
    }


def extraction():
    return {
        "entities": [
            {
                "name": "E-12",
                "type": "FaultCode",
                "description": "显示屏故障码",
                "evidence_quotes": ["E-12"],
            },
            {
                "name": "联系服务机构更换",
                "type": "Procedure",
                "description": "不能由用户更换",
                "evidence_quotes": ["请联系爱普生认证服务机构更换。它不能由用户更换。"],
            },
            {
                "name": "爱普生认证服务机构",
                "type": "Organization",
                "description": "执行更换的机构",
                "evidence_quotes": ["爱普生认证服务机构"],
            },
        ],
        "relationships": [
            {
                "source": "E-12",
                "target": "联系服务机构更换",
                "relation": "HANDLED_BY",
                "description": "联系机构更换，不能由用户更换",
                "evidence_quotes": [
                    "E-12：请联系爱普生认证服务机构更换。它不能由用户更换。"
                ],
                "qualifiers": [
                    {
                        "kind": "prohibition",
                        "text": "不能由用户更换",
                        "evidence_quotes": ["它不能由用户更换。"],
                    }
                ],
            },
            {
                "source": "联系服务机构更换",
                "target": "爱普生认证服务机构",
                "relation": "PERFORMED_BY",
                "description": "由认证服务机构执行",
                "evidence_quotes": ["请联系爱普生认证服务机构更换。"],
                "qualifiers": [],
            },
        ],
    }


def response(value=None):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(extraction() if value is None else value)
                },
            }
        ]
    }


def test_valid_records_keep_organization_and_prohibition_without_touching_raw():
    raw = response()
    before = deepcopy(raw)
    result = printer.review_response(raw, sample())
    assert result["accepted"] == extraction()
    assert result["rejected"] == []
    assert result["semantic_review_required"] is True
    assert raw == before


@pytest.mark.parametrize(
    "quote",
    ["请联系……更换。", "请联系更换。", "“它不能由用户更换。”", "它能由用户更换。"],
)
def test_damaged_quotes_are_isolated_without_discarding_other_records(quote):
    value = extraction()
    value["relationships"][0]["qualifiers"][0]["evidence_quotes"] = [quote]
    result = printer.review_response(response(value), sample())
    assert len(result["accepted"]["entities"]) == 3
    assert len(result["accepted"]["relationships"]) == 1
    assert result["rejected"][0]["path"] == "relationships[0]"


def test_rejected_entity_also_quarantines_dependent_relationships():
    value = extraction()
    value["entities"][1]["type"] = "InventedType"
    result = printer.review_response(response(value), sample())
    assert len(result["accepted"]["entities"]) == 2
    assert result["accepted"]["relationships"] == []
    assert len(result["rejected"]) == 3


def test_duplicate_name_with_different_types_is_not_silently_resolved():
    value = extraction()
    value["entities"].append({**value["entities"][1], "type": "Component"})
    result = printer.review_response(response(value), sample())
    assert len(result["rejected"]) == 4
    assert result["accepted"]["relationships"] == []


def test_handler_must_be_procedure_not_organization():
    value = extraction()
    value["relationships"][0]["target"] = "爱普生认证服务机构"
    result = printer.review_response(response(value), sample())
    assert len(result["accepted"]["relationships"]) == 1
    assert "endpoint types" in result["rejected"][0]["reason"]


@pytest.mark.parametrize(
    "damage", ["unknown_relation", "missing_qualifiers", "extra_field"]
)
def test_contract_errors_are_not_mapped_to_different_facts(damage):
    value = extraction()
    if damage == "unknown_relation":
        value["relationships"][0]["relation"] = "DEPENDS_ON"
    elif damage == "missing_qualifiers":
        del value["relationships"][0]["qualifiers"]
    else:
        value["relationships"][0]["confidence"] = 1
    result = printer.review_response(response(value), sample())
    assert len(result["accepted"]["relationships"]) == 1
    assert len(result["rejected"]) == 1


def test_filename_model_is_not_accepted_as_explicit_source_entity():
    value = extraction()
    value["entities"].append(
        {
            "name": "L5298",
            "type": "DeviceModel",
            "description": "文件名型号",
            "evidence_quotes": ["E-12"],
        }
    )
    result = printer.review_response(response(value), sample())
    assert result["rejected"][0]["path"] == "entities[3]"


def test_numeric_code_is_preserved_with_leading_zeros():
    value = {
        "entities": [
            {
                "name": "002",
                "type": "FaultCode",
                "description": "代码",
                "evidence_quotes": ["002"],
            }
        ],
        "relationships": [],
    }
    source = {**sample(), "input_text": "错误代码 002"}
    assert printer.review_response(response(value), source)["accepted"] == value
    value["entities"][0]["name"] = "2"
    # Presence is not code identity validation: quote and exact name must agree.
    result = printer.review_response(response(value), source)
    assert result["rejected"]


def test_fault_code_cannot_be_a_natural_language_source_phrase():
    value = extraction()
    value["entities"][0]["name"] = "更换"
    value["entities"][0]["evidence_quotes"] = ["更换"]
    result = printer.review_response(response(value), sample())
    assert result["rejected"][0]["path"] == "entities[0]"


def test_explicit_cause_relation_does_not_treat_an_operation_as_a_cause():
    source = {**sample(), "input_text": "连接失败是由于 IP 地址冲突。检查 IP 地址。"}
    value = {
        "entities": [
            {
                "name": "连接失败",
                "type": "Symptom",
                "description": "连接失败",
                "evidence_quotes": ["连接失败"],
            },
            {
                "name": "IP 地址冲突",
                "type": "Cause",
                "description": "明确原因",
                "evidence_quotes": ["IP 地址冲突"],
            },
        ],
        "relationships": [
            {
                "source": "连接失败",
                "target": "IP 地址冲突",
                "relation": "CAUSED_BY",
                "description": "原文明确指出由地址冲突导致连接失败",
                "evidence_quotes": ["连接失败是由于 IP 地址冲突。"],
                "qualifiers": [],
            }
        ],
    }
    assert printer.review_response(response(value), source)["rejected"] == []
    value["entities"][1]["type"] = "Procedure"
    assert len(printer.review_response(response(value), source)["rejected"]) == 1


def test_limit_relation_requires_explicit_model_qualifier():
    source = {**sample(), "input_text": "以下功能仅适用于 L5298：传真。"}
    value = {
        "entities": [
            {
                "name": "传真",
                "type": "Feature",
                "description": "功能",
                "evidence_quotes": ["传真"],
            },
            {
                "name": "L5298",
                "type": "DeviceModel",
                "description": "型号",
                "evidence_quotes": ["L5298"],
            },
        ],
        "relationships": [
            {
                "source": "传真",
                "target": "L5298",
                "relation": "LIMITED_TO",
                "description": "仅适用于 L5298",
                "evidence_quotes": [source["input_text"]],
                "qualifiers": [],
            }
        ],
    }
    assert len(printer.review_response(response(value), source)["rejected"]) == 1
    value["relationships"][0]["qualifiers"] = [
        {
            "kind": "model_restriction",
            "text": "仅适用于 L5298",
            "evidence_quotes": [source["input_text"]],
        }
    ]
    assert printer.review_response(response(value), source)["rejected"] == []


@pytest.mark.parametrize("damage", ["truncated", "json", "arrays"])
def test_sample_failure_does_not_salvage_truncated_output(damage):
    raw = response()
    if damage == "truncated":
        raw["choices"][0]["finish_reason"] = "length"
    elif damage == "json":
        raw["choices"][0]["message"]["content"] = "```json\n{}\n```"
    else:
        raw = response({"entities": {}, "relationships": []})
    result = printer.review_response(raw, sample())
    assert result["sample_error"]
    assert result["accepted"]["entities"] == []


def test_only_whitespace_is_ignored_when_checking_quotes():
    assert printer.quoted_in_source("废墨收集垫", "废墨收 集垫")
    assert not printer.quoted_in_source("按 X 按钮", "按 x 按钮")
    assert not printer.quoted_in_source("按 x 按钮", "按 × 按钮")


def test_committed_native_profile_loads_and_formats_in_both_modes(
    monkeypatch, tmp_path
):
    repo = Path(__file__).resolve().parents[2]
    path = repo / "prompts/samples/printer_diagnosis.v1.yml"
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == printer.native_profile()
    target = tmp_path / "entity_type"
    target.mkdir()
    (target / path.name).write_bytes(path.read_bytes())
    monkeypatch.setenv("PROMPT_DIR", str(tmp_path))
    for use_json in (False, True):
        profile = resolve_entity_extraction_prompt_profile(
            {"entity_type_prompt_file": path.name}, use_json
        )
        assert "Organization" in profile["entity_types_guidance"]
        example_key = (
            "entity_extraction_json_examples"
            if use_json
            else "entity_extraction_examples"
        )
        key = (
            "entity_extraction_json_system_prompt"
            if use_json
            else "entity_extraction_system_prompt"
        )
        rendered = PROMPTS[key].format(
            entity_types_guidance=profile["entity_types_guidance"],
            examples="\n".join(profile[example_key]),
            language="Chinese",
            max_total_records=28,
            max_entity_records=12,
            tuple_delimiter="<|#|>",
            completion_delimiter="<|COMPLETE|>",
        )
        assert "PERFORMED_BY" in rendered
    assert load_entity_extraction_prompt_profile(path) == printer.native_profile()


def test_profile_export_refuses_to_overwrite_operator_customizations(tmp_path):
    path = tmp_path / "profile.yml"
    assert printer.main(["--export-profile", str(path)]) == 0
    with pytest.raises(FileExistsError):
        printer.main(["--export-profile", str(path)])
