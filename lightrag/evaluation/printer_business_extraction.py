"""Prepare or run six real requests using the versioned printer business contract.

Preflight never calls a provider. --run is an explicit paid extraction action.
No graph, embedding, keyword or answer model is invoked by this entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path

from lightrag.document_structure.children import load_local_tokenizer
from lightrag.document_structure.parents import file_hash
from lightrag.evaluation.batch_baseline import write_json
from lightrag.evaluation.compare_parent_child_retrieval import digest, experiment_lock
from lightrag.evaluation.printer_graph_samples import load_samples
from lightrag.evaluation.printer_graph_smoke import call_model, config, public_config
from lightrag.extraction import printer

VERSION = "printer-business-extraction-experiment-v1"


def source_metadata(sample: dict) -> dict:
    """Copy known provenance; local model restrictions never rewrite parent scope."""
    keys = (
        "doc_id",
        "document_name",
        "parent_id",
        "child_ids",
        "heading_path",
        "domain",
        "scope",
        "pages",
        "source_spans",
        "projection",
        "table_notes",
        "inline_notes",
    )
    return {
        **{key: deepcopy(sample[key]) for key in keys if key in sample},
        "source_text_sha256": hashlib.sha256(sample["input_text"].encode()).hexdigest(),
        "scope_policy": "Preserved parent scope; extracted local restrictions need review",
    }


def write_report(output: Path, reviews: dict, summary: dict) -> None:
    lines = [
        "# 打印机业务抽取检查",
        "",
        "本轮只检查真实模型抽取，不入库、不生成答案、不评估检索准确率。",
        "结构及引文存在不等于语义正确；限定条件是否完整、是否适用于整个事实仍需审阅。",
        "",
        f"规范版本：`{printer.VERSION}`；状态：`{summary['status']}`。",
        "",
        "| 样本 | 保留实体 | 保留关系 | 隔离记录 | 整份解析错误 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for key, review in reviews.items():
        lines.append(
            f"| {key} | {len(review['accepted']['entities'])} | "
            f"{len(review['accepted']['relationships'])} | {len(review['rejected'])} | "
            f"{'有，见 validation.json' if review['sample_error'] else '无'} |"
        )
    lines.extend(
        [
            "",
            "## 人工语义检查",
            "",
            "- E-12：网络报告和显示屏两种含义、操作应分别正确，不能凭处理方法推断原因。",
            "- W-11：检查两份手册的按钮、恢复条件、临时恢复与更换要求是否保留。",
            "- 废墨垫：区分无边距/有边距打印，部件寿命不能解释为整机寿命。",
            "- 服务机构必须是 Organization；更换动作才是 Procedure。",
            "- 功能列表：L5298 的明确限制应保留；ADF 规格段不能凭文件名生成支持关系。",
            "- 逐条核对 qualifiers：否定、前提、按钮、脚注、局部型号限制是否遗漏或过度扩展。",
            "",
            "evidence_records.json 将每条记录与程序附加的原始范围及来源关联。",
            "该范围没有因局部限制自动缩小；使用事实前仍须结合已审阅的局部限制。",
            "本轮尚未实现正式图谱身份合并、适用性过滤或父块回查接入。",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path("outputs/printer_parent_child/20261017_v3"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--tokenizer-file",
        type=Path,
        default=Path("tmp/parent_blocks_tools/cl100k_base.tiktoken"),
    )
    parser.add_argument("--max-output-tokens", type=int, default=6144)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not 256 <= args.max_output_tokens <= 16384:
        raise ValueError("max-output-tokens must be between 256 and 16384")
    settings = config(args.env_file, args.max_output_tokens)
    samples, source_hashes = load_samples(args.bundle_root)
    tokenizer = load_local_tokenizer(args.tokenizer_file)
    prompts = [{"id": s["id"], "messages": printer.messages(s)} for s in samples]
    dependencies = [
        Path(__file__),
        Path(printer.__file__),
        Path(__file__).with_name("printer_graph_smoke.py"),
        Path(__file__).with_name("printer_graph_samples.py"),
        Path(__file__).with_name("compare_parent_child_retrieval.py"),
        Path(__file__).with_name("batch_baseline.py"),
    ]
    plan = {
        "version": VERSION,
        "contract_version": printer.VERSION,
        "provider": public_config(settings),
        "samples": samples,
        "source_hashes": source_hashes,
        "prompts": prompts,
        "dependency_hashes": {path.name: file_hash(path) for path in dependencies},
        "max_requests": len(samples),
        "local_prompt_tokens": sum(
            len(tokenizer.encode(m["content"])) for p in prompts for m in p["messages"]
        ),
        "embedding_requests": 0,
        "answer_generation_requests": 0,
    }
    plan["plan_hash"] = digest(plan)
    try:
        args.output.mkdir(parents=True)
        created = True
    except FileExistsError:
        created = False
    if not created and not args.resume:
        raise ValueError("Output exists; use --resume or a new directory")
    with experiment_lock(args.output):
        plan_path = args.output / "experiment.json"
        if not created:
            if (
                not plan_path.exists()
                or json.loads(plan_path.read_text(encoding="utf-8")) != plan
            ):
                raise ValueError(
                    "Frozen plan changed or foreign output; use a new directory"
                )
        else:
            write_json(plan_path, plan)
            write_json(args.output / "samples.json", samples)
            write_json(args.output / "prompts.json", prompts)
        raw_dir = args.output / "raw"
        cached = sum((raw_dir / f"{s['id']}.json").exists() for s in samples)
        print(
            json.dumps(
                {
                    "contract_version": printer.VERSION,
                    "samples": len(samples),
                    "model": settings["model"],
                    "cached_responses": cached,
                    "remaining_requests": len(samples) - cached,
                    "local_prompt_tokens": plan["local_prompt_tokens"],
                    "max_output_tokens_per_request": args.max_output_tokens,
                    "embedding_requests": 0,
                    "answer_generation_requests": 0,
                }
            ),
            flush=True,
        )
        if not args.run:
            print("Preflight only; no model requests made.")
            return 0
        if not settings["api_key"]:
            raise ValueError("No extraction API key; no requests made")
        raw_dir.mkdir(exist_ok=True)
        reviews, usage, new_requests = {}, {}, 0
        evidence_records = []
        for index, sample in enumerate(samples):
            path = raw_dir / f"{sample['id']}.json"
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                if (
                    record.get("plan_hash") != plan["plan_hash"]
                    or record.get("sample_id") != sample["id"]
                ):
                    raise ValueError(
                        "Cached response belongs to a different experiment"
                    )
            else:
                started = time.monotonic()
                response = call_model(settings, prompts[index]["messages"])
                record = {
                    "plan_hash": plan["plan_hash"],
                    "sample_id": sample["id"],
                    "response": response,
                    "elapsed_seconds": time.monotonic() - started,
                }
                write_json(path, record)
                new_requests += 1
            for key, value in record["response"].get("usage", {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[key] = usage.get(key, 0) + value
            review = printer.review_response(record["response"], sample)
            reviews[sample["id"]] = review
            for kind, items in review["accepted"].items():
                evidence_records.extend(
                    {
                        "sample_id": sample["id"],
                        "kind": kind,
                        "record": item,
                        "source_metadata": source_metadata(sample),
                    }
                    for item in items
                )
            print(
                f"Extraction progress: {index + 1}/{len(samples)}; new requests={new_requests}",
                flush=True,
            )
        rejected = sum(len(r["rejected"]) for r in reviews.values())
        parse_errors = sum(r["sample_error"] is not None for r in reviews.values())
        empty = sum(not r["accepted"]["entities"] for r in reviews.values())
        summary = {
            "plan_hash": plan["plan_hash"],
            "contract_version": printer.VERSION,
            "provider": public_config(settings),
            "successful_responses": len(samples),
            "new_requests_this_run": new_requests,
            "usage": usage,
            "accepted_entities": sum(
                len(r["accepted"]["entities"]) for r in reviews.values()
            ),
            "accepted_relationships": sum(
                len(r["accepted"]["relationships"]) for r in reviews.values()
            ),
            "rejected_records": rejected,
            "sample_parse_failures": parse_errors,
            "empty_samples": empty,
            "semantic_review_required": True,
            "status": "reviewed_with_rejections"
            if rejected or parse_errors or empty
            else "needs_semantic_review",
            "embedding_requests": 0,
            "answer_generation_requests": 0,
        }
        write_json(
            args.output / "extractions.json",
            {k: r["accepted"] for k, r in reviews.items()},
        )
        write_json(args.output / "validation.json", reviews)
        write_json(args.output / "evidence_records.json", evidence_records)
        write_json(args.output / "summary.json", summary)
        write_report(args.output, reviews, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 2 if rejected or parse_errors or empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
