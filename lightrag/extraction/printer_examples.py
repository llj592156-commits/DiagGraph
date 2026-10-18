"""Teaching examples for prompts, never a substitute for real extraction results."""

from __future__ import annotations

import json


def _entity(name: str, kind: str, description: str, *quotes: str) -> dict:
    return {
        "name": name,
        "type": kind,
        "description": description,
        "evidence_quotes": list(quotes),
    }


def _qualifier(kind: str, text: str, *quotes: str) -> dict:
    return {"kind": kind, "text": text, "evidence_quotes": list(quotes)}


def _relation(
    source: str,
    target: str,
    kind: str,
    description: str,
    quotes: list[str],
    qualifiers: list[dict] | None = None,
) -> dict:
    return {
        "source": source,
        "target": target,
        "relation": kind,
        "description": description,
        "evidence_quotes": quotes,
        "qualifiers": qualifiers or [],
    }


def worked_examples() -> list[dict]:
    check = "请检查 IP 地址是否与其他设备冲突。"
    condition = "如果确认以上事项之后仍无法连接，请尝试以下解决方法。"
    restart = "关闭无线路由器。等待大约 10 秒，然后打开无线路由器。"
    network = {
        "purpose": "Inspection is a Procedure; only a stated execution prerequisite is a Condition.",
        "source_text": f"E-12\n{check}\n{condition}\n{restart}",
        "output": {
            "entities": [
                _entity("E-12", "FaultCode", "本片段中的网络故障码", "E-12"),
                _entity("检查 IP 地址是否冲突", "Procedure", "检查网络地址冲突", check),
                _entity(
                    "重启无线路由器", "Procedure", "关闭后等待大约 10 秒再打开", restart
                ),
                _entity(
                    "确认检查事项后仍无法连接",
                    "Condition",
                    "继续尝试重启的明确前提",
                    condition,
                ),
            ],
            "relationships": [
                _relation(
                    "E-12",
                    "检查 IP 地址是否冲突",
                    "HANDLED_BY",
                    "针对 E-12，检查 IP 地址是否与其他设备冲突",
                    ["E-12", check],
                ),
                _relation(
                    "E-12",
                    "重启无线路由器",
                    "HANDLED_BY",
                    "确认检查事项后仍无法连接时，可关闭路由器，等待大约 10 秒再打开",
                    ["E-12", condition, restart],
                    [_qualifier("prerequisite", "确认检查事项后仍无法连接", condition)],
                ),
                _relation(
                    "重启无线路由器",
                    "确认检查事项后仍无法连接",
                    "REQUIRES",
                    "本片段将确认检查事项后仍无法连接作为尝试重启的前提",
                    [condition, restart],
                    [_qualifier("prerequisite", "确认检查事项后仍无法连接", condition)],
                ),
            ],
        },
    }
    state = "废墨收集垫已接近使用寿命。"
    service = "请联系爱普生认证服务机构更换废墨收集垫。"
    prohibition = "它不能由用户更换。"
    notification = "在更换废墨收集垫之前会一直显示该信息。"
    button = "按 x 按钮恢复打印。"
    footnote = "*1 需要更换废墨收集垫并不表示本产品已达到使用寿命。"
    table = json.dumps(
        [
            ["代码", "情况", "解决方法"],
            ["W-11", state, service + prohibition + notification + button + " *1"],
        ],
        ensure_ascii=False,
    )
    display = {
        "purpose": "Quote table cells separately; prohibitions and persistent messages are not prerequisites.",
        "source_text": table + "\n" + footnote,
        "output": {
            "entities": [
                _entity("W-11", "FaultCode", "本片段中的提示代码", "W-11"),
                _entity(
                    "废墨收集垫已接近使用寿命", "Symptom", "部件接近寿命的状态", state
                ),
                _entity(
                    "联系认证服务机构更换废墨收集垫",
                    "Procedure",
                    "联系认证机构更换，不能由用户更换；更换前提示持续存在，部件需更换不等于整机达到寿命",
                    service,
                    prohibition,
                    notification,
                    footnote,
                ),
                _entity(
                    "爱普生认证服务机构",
                    "Organization",
                    "执行更换的认证机构",
                    "爱普生认证服务机构",
                ),
                _entity(
                    "按 x 按钮恢复打印",
                    "Procedure",
                    "按 x 恢复打印，不表示已完成部件更换",
                    button,
                ),
            ],
            "relationships": [
                _relation(
                    "W-11",
                    "废墨收集垫已接近使用寿命",
                    "INDICATES",
                    "W-11 表示废墨收集垫已接近使用寿命",
                    ["W-11", state],
                ),
                _relation(
                    "W-11",
                    "联系认证服务机构更换废墨收集垫",
                    "HANDLED_BY",
                    "联系认证机构更换，不能由用户更换；更换前持续显示提示，部件需更换不等于整机达到寿命",
                    ["W-11", service, prohibition, notification, footnote],
                    [
                        _qualifier("prohibition", "不能由用户更换", prohibition),
                        _qualifier(
                            "footnote", "部件需更换不等于整机达到使用寿命", footnote
                        ),
                    ],
                ),
                _relation(
                    "联系认证服务机构更换废墨收集垫",
                    "爱普生认证服务机构",
                    "PERFORMED_BY",
                    "联系认证服务机构进行更换，不能由用户自行更换",
                    [service, prohibition],
                    [_qualifier("prohibition", "不能由用户更换", prohibition)],
                ),
                _relation(
                    "W-11",
                    "按 x 按钮恢复打印",
                    "HANDLED_BY",
                    "可按 x 按钮恢复打印；此动作不表示更换已经完成",
                    ["W-11", button],
                    [_qualifier("button", "按钮符号为 x", button)],
                ),
            ],
        },
    }
    restriction = "ADF 规格（仅对于 L5298）。"
    specification = "自动双面扫描：不支持。"
    warning = "即使原稿符合介质规格要求，也可能无法从 ADF 送入。"
    adf = {
        "purpose": "Keep model limits and negative capability; even-if is not an execution prerequisite.",
        "source_text": restriction + "\n" + specification + "\n" + warning,
        "output": {
            "entities": [
                _entity(
                    "ADF", "Feature", "本片段仅适用于 L5298 的 ADF 功能", restriction
                ),
                _entity("L5298", "DeviceModel", "原文明确的适用型号", "L5298"),
                _entity(
                    "自动双面扫描：不支持",
                    "Specification",
                    "本片段中 ADF 不支持自动双面扫描",
                    specification,
                ),
                _entity(
                    "原稿可能无法从 ADF 送入",
                    "Symptom",
                    "即使符合介质规格仍可能发生，并非必然发生",
                    warning,
                ),
            ],
            "relationships": [
                _relation(
                    "ADF",
                    "L5298",
                    "LIMITED_TO",
                    "该 ADF 规格段仅适用于 L5298",
                    [restriction],
                    [_qualifier("model_restriction", "仅对于 L5298", restriction)],
                ),
                _relation(
                    "ADF",
                    "自动双面扫描：不支持",
                    "HAS_SPEC",
                    "仅对 L5298，本段 ADF 的自动双面扫描规格为不支持",
                    [restriction, specification],
                    [_qualifier("model_restriction", "仅对于 L5298", restriction)],
                ),
                _relation(
                    "原稿可能无法从 ADF 送入",
                    "ADF",
                    "INVOLVES",
                    "仅对本段 L5298 的 ADF，即使原稿符合介质规格仍可能无法送入；不是必然故障",
                    [restriction, warning],
                    [_qualifier("model_restriction", "仅对于 L5298", restriction)],
                ),
            ],
        },
    }
    return [network, display, adf]


def render_examples() -> str:
    blocks = []
    for index, example in enumerate(worked_examples(), 1):
        blocks.append(
            f"Example {index}: {example['purpose']}\n"
            f"Example source_text:\n{example['source_text']}\n"
            "Example output:\n" + json.dumps(example["output"], ensure_ascii=False)
        )
    return "\n\n".join(blocks)
