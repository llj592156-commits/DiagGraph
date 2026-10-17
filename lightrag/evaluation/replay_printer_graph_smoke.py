"""Review saved real extraction responses without issuing model requests.

Raw records and the original strict outcome stay immutable. Only an extra
matching outer quote pair can be normalized. Unsupported relations, missing
endpoints and noncontiguous evidence are quarantined, never repaired by guess.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from lightrag.document_structure.parents import file_hash
from lightrag.evaluation.batch_baseline import write_json
from lightrag.evaluation.compare_parent_child_retrieval import digest
from lightrag.evaluation.printer_graph_smoke import (
    build_graph,
    graph_counts,
    normalized,
    query_graph,
    validate_extraction,
)

VERSION = "printer-graph-record-review-v1"


def normalize_quotes(item: dict, text: str, path: str) -> tuple[dict, list[dict]]:
    result, changes = deepcopy(item), []
    if not isinstance(result, dict) or not isinstance(
        result.get("evidence_quotes"), list
    ):
        return result, changes
    for index, quote in enumerate(result["evidence_quotes"]):
        if not isinstance(quote, str) or normalized(quote) in normalized(text):
            continue
        stripped = quote.strip()
        if len(stripped) < 3 or (stripped[0], stripped[-1]) not in {
            ('"', '"'),
            ("'", "'"),
            ("“", "”"),
            ("「", "」"),
        }:
            continue
        inner = stripped[1:-1]
        if inner.strip() and normalized(inner) in normalized(text):
            result["evidence_quotes"][index] = inner
            changes.append(
                {
                    "path": f"{path}.evidence_quotes[{index}]",
                    "rule": "one_matching_outer_quote_pair",
                    "original": quote,
                    "normalized": inner,
                }
            )
    return result, changes


def local_payload(entities: list[dict], relationships: list[dict]) -> dict:
    # Adapt already-produced records to the original validator's interface;
    # this object is never represented as a new provider response or sent out.
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {"entities": entities, "relationships": relationships}
                    ),
                },
            }
        ]
    }


def review_response(response: dict, sample: dict) -> dict:
    rejected, transformations = [], []
    result = {"entities": [], "relationships": []}
    strict_error = None
    try:
        validate_extraction(response, sample)
    except (ValueError, TypeError) as exc:
        strict_error = str(exc)
    try:
        choices = response.get("choices", [])
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            raise ValueError("Missing, truncated or non-text extraction completion")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str):
            raise TypeError("Extraction completion has no text")
        value = json.loads(content)
        if not isinstance(value, dict) or set(value) != {"entities", "relationships"}:
            raise ValueError("Extraction must contain entities and relationships only")
        if (
            not isinstance(value["entities"], list)
            or not isinstance(value["relationships"], list)
            or len(value["entities"]) > 12
            or len(value["relationships"]) > 16
        ):
            raise ValueError("Invalid arrays or record limit")
    except (ValueError, TypeError) as exc:
        return {
            "sample_id": sample["id"],
            "strict_passed": False,
            "strict_error": strict_error or str(exc),
            "sample_error": str(exc),
            "accepted": result,
            "rejected": [],
            "format_normalizations": [],
        }
    names = set()
    for kind in ("entities", "relationships"):
        for position, item in enumerate(value[kind]):
            path = f"{kind}[{position}]"
            candidate, changes = normalize_quotes(item, sample["input_text"], path)
            try:
                if kind == "entities":
                    validate_extraction(local_payload([candidate], []), sample)
                    key = normalized(candidate["name"])
                    if key in names:
                        raise ValueError("Duplicate entity name")
                    names.add(key)
                else:
                    validate_extraction(
                        local_payload(result["entities"], [candidate]), sample
                    )
                result[kind].append(candidate)
                transformations.extend(changes)
            except (ValueError, TypeError) as exc:
                rejected.append(
                    {"path": path, "reason": str(exc), "original_record": item}
                )
    return {
        "sample_id": sample["id"],
        "strict_passed": strict_error is None,
        "strict_error": strict_error,
        "sample_error": None,
        "accepted": result,
        "rejected": rejected,
        "format_normalizations": transformations,
    }


def load_saved(source: Path) -> tuple[dict, list[dict], dict, dict]:
    plan = json.loads((source / "experiment.json").read_text(encoding="utf-8"))
    if plan.get("version") != "printer-real-graph-smoke-v1":
        raise ValueError("Unsupported source graph experiment version")
    unhashed = {key: value for key, value in plan.items() if key != "plan_hash"}
    if digest(unhashed) != plan["plan_hash"]:
        raise ValueError("Original experiment plan hash mismatch")
    samples = json.loads((source / "samples.json").read_text(encoding="utf-8"))
    if samples != plan["samples"] or len({s["id"] for s in samples}) != len(samples):
        raise ValueError("Saved samples differ from original plan")
    records, hashes = (
        {},
        {
            "experiment.json": file_hash(source / "experiment.json"),
            "samples.json": file_hash(source / "samples.json"),
        },
    )
    for sample in samples:
        path = source / "raw" / (sample["id"] + ".json")
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record["plan_hash"] != plan["plan_hash"]
            or record["sample_id"] != sample["id"]
        ):
            raise ValueError("Raw response belongs to a different source experiment")
        records[sample["id"]] = record["response"]
        hashes[str(path.relative_to(source))] = file_hash(path)
    return plan, samples, records, hashes


def write_review_report(
    output: Path,
    reviews: list[dict],
    records: list[dict],
    summary: dict,
    queries: list[dict],
) -> None:
    lines = [
        "# 已保存真实抽取结果的逐条复核",
        "",
        "本次仅处理此前模型真实返回的内容，新增模型请求为 0。原始响应、首轮严格验证和原图谱全部保留。",
        "",
        f"原始整份样本严格通过 {summary['original_strict_passed']}/{len(reviews)}；复核保留 {summary['accepted_entities']} 个实体、{summary['accepted_relationships']} 条关系，隔离 {summary['rejected_records']} 条记录，格式处理 {summary['format_normalizations']} 处。保留部分记录不表示原始失败样本已经全部通过。",
        "",
        "只在去掉一层配对外引号后仍能在原文连续定位时处理引号。省略号、拼接引文和未知关系类型仍被拒绝；不会手工补节点或关系。",
        "",
        "| 样本 | 原始严格通过 | 保留实体 | 保留关系 | 拒绝记录 | 引号处理 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in reviews:
        lines.append(
            f"| {item['sample_id']} | {item['strict_passed']} | {len(item['accepted']['entities'])} | {len(item['accepted']['relationships'])} | {len(item['rejected'])} | {len(item['format_normalizations'])} |"
        )
        if item["sample_error"]:
            lines.append(f"样本级错误：{item['sample_error']}")
    lines.extend(
        [
            "",
            "## 图谱查询",
            "",
            "两张图都只使用上面保留的真实记录。一跳查询和条件筛选规则与原实验相同。引用能定位不等于事实语义正确；类型、方向、操作与条件仍须人工复核。",
            "",
            "| 问题 | 按名称：证据数 | 场景及型号：状态 / 证据数 |",
            "| --- | ---: | --- |",
        ]
    )
    for query in queries:
        selected = {r["policy"]: r for r in records if r["id"] == query["id"]}
        raw, scoped = selected["name_only"], selected["scoped"]
        lines.append(
            f"| {query['id']} {query['query']} | {len(raw['evidence'])} | {scoped['status']} / {len(scoped['evidence'])} |"
        )
    lines.extend(
        [
            "",
            "查看 record_review.json 中每条拒绝记录与引号处理前后文本，查看 extractions.json 中实际保留的模型记录。某条关键关系被隔离后，相关查询可能仍缺证据；不能据此宣布语义消歧或按钮识别已全部正确。",
            "",
            "此处没有自动重试模型、补抽或回答生成。若后续改变抽取提示，需要另开真实实验，不能把新模型响应混入本轮。",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Review output exists; choose a new directory")
    plan, samples, saved, hashes = load_saved(args.source_directory)
    reviews = [review_response(saved[s["id"]], s) for s in samples]
    summary = {
        "version": VERSION,
        "source_directory": str(args.source_directory.resolve()),
        "source_plan_hash": plan["plan_hash"],
        "source_artifact_hashes": hashes,
        "review_script_sha256": file_hash(Path(__file__)),
        "graph_builder_sha256": file_hash(
            Path(__file__).with_name("printer_graph_smoke.py")
        ),
        "original_strict_passed": sum(r["strict_passed"] for r in reviews),
        "sample_parse_failures": sum(r["sample_error"] is not None for r in reviews),
        "accepted_entities": sum(len(r["accepted"]["entities"]) for r in reviews),
        "accepted_relationships": sum(
            len(r["accepted"]["relationships"]) for r in reviews
        ),
        "rejected_records": sum(len(r["rejected"]) for r in reviews),
        "format_normalizations": sum(len(r["format_normalizations"]) for r in reviews),
        "model_requests": 0,
        "semantic_review_required": True,
        "status": "reviewed_with_rejections"
        if any(r["rejected"] or r["sample_error"] for r in reviews)
        else "reviewed",
    }
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in {"source_artifact_hashes", "source_directory"}
            },
            ensure_ascii=False,
        )
    )
    if args.preflight:
        return 0
    extractions = {
        r["sample_id"]: r["accepted"] for r in reviews if r["sample_error"] is None
    }
    args.output.mkdir(parents=True, exist_ok=False)
    records, graph_stats = [], {}
    for scoped in (False, True):
        graph = build_graph(samples, extractions, scoped)
        policy = "scoped" if scoped else "name_only"
        graph_stats[policy] = graph_counts(graph)
        write_json(
            args.output / f"graph_{policy}.json",
            {
                "policy": policy,
                "directed": True,
                "multigraph": True,
                "nodes": [{"id": key, **data} for key, data in graph.nodes(data=True)],
                "edges": [
                    {"source_node": left, "target_node": right, "key": key, **data}
                    for left, right, key, data in graph.edges(keys=True, data=True)
                ],
            },
        )
        records.extend(query_graph(graph, query, scoped) for query in plan["queries"])
    summary["graphs"] = graph_stats
    write_json(args.output / "record_review.json", reviews)
    write_json(args.output / "extractions.json", extractions)
    write_json(args.output / "retrieval_results.json", records)
    write_review_report(args.output, reviews, records, summary, plan["queries"])
    write_json(args.output / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
