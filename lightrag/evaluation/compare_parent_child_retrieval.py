"""Run an isolated dense-vector retrieval comparison, without answer generation.

Remote embedding is opt-in. Input/oracle text never enters the document index;
only query strings are embedded separately. Existing stores are read-only.
Each output directory has one frozen plan and a resumable embedding cache.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from urllib import error, parse, request

import numpy as np

from lightrag.document_structure.capabilities import query_features
from lightrag.document_structure.children import load_local_tokenizer
from lightrag.document_structure.parents import MODEL_RE, file_hash
from lightrag.document_structure.retrieval import ParentChildIndex
from lightrag.document_structure.table_notes import evidence_text, with_table_notes

VERSION = "printer-dense-retrieval-v3"


def provider_identity(config: dict) -> dict:
    return {
        **{k: config[k] for k in ("base_url", "model", "dimension")},
        "send_dimensions": config.get("send_dimensions", False),
    }


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def normalized(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text)).casefold()


def models(text: str) -> list[str]:
    return sorted(set(MODEL_RE.findall(text.upper().replace("_", " "))))


def embedding_config(env_file: Path) -> dict:
    from dotenv import dotenv_values

    values = {**dotenv_values(env_file), **os.environ}
    if str(values.get("EMBEDDING_ASYMMETRIC_ENABLED", "false")).lower() in {
        "1",
        "true",
        "yes",
    }:
        raise ValueError(
            "This comparison requires symmetric embedding inputs; asymmetric document/query roles need separate cache identities"
        )
    if values.get("EMBEDDING_BINDING") != "openai":
        raise ValueError(
            "This experiment requires the configured OpenAI-compatible embedding binding"
        )
    base = (values.get("EMBEDDING_BINDING_HOST") or "https://api.openai.com/v1").rstrip(
        "/"
    )
    parts = parse.urlsplit(base)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError(
            "Embedding endpoint must not contain credentials/query/fragment"
        )
    key = values.get("EMBEDDING_BINDING_API_KEY") or values.get("OPENAI_API_KEY")
    model = values.get("EMBEDDING_MODEL")
    dimension = int(values.get("EMBEDDING_DIM") or 2048)
    if not model or dimension < 1:
        raise ValueError("Embedding model and positive dimension are required")
    return {
        "base_url": base,
        "model": model,
        "dimension": dimension,
        "api_key": key or "",
        "send_dimensions": str(values.get("EMBEDDING_SEND_DIM", "false")).lower()
        in {"1", "true", "yes"},
        "token_limit": int(values.get("EMBEDDING_TOKEN_LIMIT") or 3072),
    }


def validate_questions(questions: list[dict]) -> None:
    seen = set()
    for row in questions:
        identifier, query = row.get("id"), row.get("query")
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier)
            or identifier in seen
        ):
            raise ValueError("Question IDs must be nonempty and unique")
        seen.add(identifier)
        if (
            not isinstance(query, str)
            or not query.strip()
            or row.get("test_kind") not in {"retrieval", "boundary"}
        ):
            raise ValueError(
                "Question requires a query and retrieval/boundary test_kind"
            )
        anchors = row.get("evidence", [])
        if not isinstance(anchors, list) or (
            row["test_kind"] == "retrieval" and not anchors
        ):
            raise ValueError("Retrieval questions require source evidence")
        for anchor in anchors:
            if (
                anchor.get("document") not in {"A", "B", "C"}
                or not isinstance(anchor.get("quote"), str)
                or not normalized(anchor["quote"])
            ):
                raise ValueError(
                    "Source evidence requires a document and nonempty quote"
                )
    if not any(row["test_kind"] == "retrieval" for row in questions):
        raise ValueError("At least one retrieval question is required")


def load_data(
    legacy: Path, bundles: list[Path], question_file: Path
) -> tuple[list[dict], list[dict], ParentChildIndex]:
    all_parents, all_children, rows, filenames = [], [], [], {}
    for bundle in bundles:
        index = ParentChildIndex.from_directory(bundle)
        letter = bundle.name
        if letter not in {"A", "B", "C"}:
            raise ValueError("Printer bundle directories must be named A, B, C")
        for p in index.parents.values():
            filenames[p["document_name"]] = letter
        all_parents.extend(index.parents.values())
        all_children.extend(index.children.values())
    combined = ParentChildIndex(all_parents, all_children)
    for key, item in sorted(json.loads(legacy.read_text(encoding="utf-8")).items()):
        if not isinstance(item.get("content"), str) or not item["content"].strip():
            raise ValueError("Legacy chunk requires nonempty text")
        name = str(item["file_path"]).replace("\\", "/").split("/")[-1]
        if name not in filenames:
            raise ValueError("Legacy chunk does not belong to a selected manual")
        rows.append(
            {
                "id": key,
                "arm": "legacy",
                "text": item["content"],
                "content": item["content"],
                "document": filenames[name],
                "document_name": name,
                "models": models(name),
                "parent_id": None,
            }
        )
    for child in all_children:
        rows.append(
            {
                "id": child["child_id"],
                "arm": "child",
                "text": child["retrieval_text"],
                "content": with_table_notes(
                    child["content"], child.get("table_notes", [])
                ),
                "document": filenames[child["document_name"]],
                "document_name": child["document_name"],
                "feature_domain": child["feature_domain"],
                "models": child["scope"]["applicable_models"],
                "parent_id": child["parent_id"],
            }
        )
    questions = [
        json.loads(line)
        for line in question_file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    validate_questions(questions)
    return rows, questions, combined


def make_plan(
    rows: list[dict], questions: list[dict], config: dict, inputs: dict, tokenizer
) -> tuple[dict, list[dict]]:
    # The jobs below contain no answer text, oracle quotes or expected labels.
    texts = [row["text"] for row in rows] + [q["query"] for q in questions]
    jobs = [
        {
            "key": digest(
                {
                    "provider": provider_identity(config),
                    "text": text,
                }
            ),
            "text": text,
        }
        for text in dict.fromkeys(texts)
    ]
    token_counts = [len(tokenizer.encode(job["text"])) for job in jobs]
    if max(token_counts, default=0) > config.get("token_limit", 3072):
        raise ValueError(
            "An experiment embedding input exceeds the conservative local token check"
        )
    plan = {
        "version": VERSION,
        "provider": provider_identity(config),
        "local_input_token_limit": config.get("token_limit", 3072),
        "documents": dict(Counter(r["arm"] for r in rows)),
        "queries": len(questions),
        "unique_embedding_inputs": len(jobs),
        "local_cl100k_input_tokens": sum(token_counts),
        "max_local_input_tokens": max(token_counts, default=0),
        "inputs": inputs,
        "job_keys": [job["key"] for job in jobs],
        "generation_model_calls": 0,
    }
    plan["plan_hash"] = digest(plan)
    return plan, jobs


def embed_batch(
    config: dict, texts: list[str], timeout: int = 60
) -> tuple[np.ndarray, dict]:
    class NoRedirect(request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    parameters = {"model": config["model"], "input": texts}
    if config.get("send_dimensions", False):
        parameters["dimensions"] = config["dimension"]
    body = json.dumps(parameters).encode()
    req = request.Request(
        config["base_url"] + "/embeddings",
        data=body,
        headers={
            "Authorization": "Bearer " + config["api_key"],
            "Content-Type": "application/json",
        },
        method="POST",
    )
    opener = request.build_opener(NoRedirect())
    for attempt in range(4):
        try:
            with opener.open(req, timeout=timeout) as response:
                payload = json.load(response)
            break
        except error.HTTPError as exc:
            code = exc.code
            exc.close()
            if code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise RuntimeError(
                    f"Embedding API HTTP {code}; batch not cached"
                ) from None
            time.sleep(min(2**attempt, 8))
    items = sorted(payload["data"], key=lambda row: row["index"])
    if [row["index"] for row in items] != list(range(len(texts))):
        raise ValueError("Embedding API returned missing/duplicate indices")
    matrix = np.asarray([row["embedding"] for row in items], dtype="<f4")
    if (
        matrix.shape != (len(texts), config["dimension"])
        or not np.isfinite(matrix).all()
        or np.any(np.linalg.norm(matrix, axis=1) == 0)
    ):
        raise ValueError("Embedding API returned invalid shape/values")
    return matrix, payload.get("usage", {})


def read_embedding_cache(path: Path, config: dict) -> dict:
    cached = {}
    if path.exists():
        # An incomplete append is not silently ignored: preserve the file and fail.
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            vector = np.frombuffer(
                base64.b64decode(item["vector_f32"], validate=True), dtype="<f4"
            )
            if (
                vector.shape != (config["dimension"],)
                or not np.isfinite(vector).all()
                or np.linalg.norm(vector) == 0
            ):
                raise ValueError("Invalid cached vector")
            if item["key"] in cached and not np.array_equal(
                cached[item["key"]], vector
            ):
                raise ValueError("Conflicting cached vectors for one input")
            cached[item["key"]] = vector.copy()
    return cached


def reusable_embeddings(directory: Path | None, config: dict) -> dict:
    if directory is None:
        return {}
    plan = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    if plan["provider"] != provider_identity(config):
        raise ValueError("Reusable cache embedding provider/configuration mismatch")
    cached = read_embedding_cache(directory / "embedding_cache.jsonl", config)
    if not set(cached).issubset(plan["job_keys"]):
        raise ValueError("Reusable cache contains inputs outside its frozen plan")
    return cached


def cached_embeddings(
    jobs: list[dict],
    config: dict,
    output: Path,
    batch_size: int,
    reuse_directory: Path | None = None,
) -> tuple[dict, dict]:
    path = output / "embedding_cache.jsonl"
    cached, requests_made, usage = read_embedding_cache(path, config), 0, Counter()
    imported = reusable_embeddings(reuse_directory, config)
    reused = [
        job for job in jobs if job["key"] not in cached and job["key"] in imported
    ]
    if reused:
        with path.open("a", encoding="utf-8") as stream:
            for job in reused:
                value = imported[job["key"]]
                stream.write(
                    json.dumps(
                        {
                            "key": job["key"],
                            "vector_f32": base64.b64encode(value.tobytes()).decode(
                                "ascii"
                            ),
                        }
                    )
                    + "\n"
                )
                cached[job["key"]] = value.copy()
        (output / "embedding_reuse.json").write_text(
            json.dumps(
                {
                    "source_directory": str(reuse_directory.resolve()),
                    "source_plan_sha256": file_hash(
                        reuse_directory / "experiment.json"
                    ),
                    "source_cache_sha256": file_hash(
                        reuse_directory / "embedding_cache.jsonl"
                    ),
                    "imported_job_keys": [job["key"] for job in reused],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    pending = [job for job in jobs if job["key"] not in cached]
    with path.open("a", encoding="utf-8") as stream:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            matrix, batch_usage = embed_batch(config, [job["text"] for job in batch])
            requests_made += 1
            with (output / "embedding_requests.jsonl").open(
                "a", encoding="utf-8"
            ) as journal:
                journal.write(
                    json.dumps(
                        {
                            "inputs": len(batch),
                            "job_keys": [job["key"] for job in batch],
                            "usage": batch_usage,
                        }
                    )
                    + "\n"
                )
            for k, value in batch_usage.items():
                if isinstance(value, (int, float)):
                    usage[k] += value
            for job, vector in zip(batch, matrix, strict=True):
                stream.write(
                    json.dumps(
                        {
                            "key": job["key"],
                            "vector_f32": base64.b64encode(vector.tobytes()).decode(
                                "ascii"
                            ),
                        }
                    )
                    + "\n"
                )
                cached[job["key"]] = vector.copy()
            stream.flush()
            print(
                f"Embedding progress: {min(start + len(batch), len(pending))}/{len(pending)} new inputs; requests={requests_made}",
                flush=True,
            )
    return cached, {
        "new_inputs": len(pending),
        "cache_hits": len(jobs) - len(pending),
        "imported_inputs_this_run": len(reused),
        "requests": requests_made,
        "provider_usage": dict(usage),
    }


def recorded_requests(output: Path) -> dict:
    path, usage, inputs, count = output / "embedding_requests.jsonl", Counter(), 0, 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            count += 1
            inputs += row["inputs"]
            for key, value in row["usage"].items():
                if isinstance(value, (int, float)):
                    usage[key] += value
    return {
        "successful_requests": count,
        "inputs": inputs,
        "provider_usage": dict(usage),
    }


@contextmanager
def experiment_lock(output: Path):
    path = output / "embedding.lock"
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError:
        raise ValueError(
            "Experiment is locked; ensure the previous process has stopped before removing embedding.lock"
        ) from None
    try:
        with stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink()


def evidence_metrics(question: dict, evidence: list[dict]) -> dict:
    anchors = question.get("evidence", [])
    covered = [
        any(
            row["document"] == anchor["document"]
            and normalized(anchor["quote"]) in normalized(row["content"])
            for row in evidence
        )
        for anchor in anchors
    ]
    return {
        "anchors": len(anchors),
        "covered_anchors": sum(covered),
        "coverage": sum(covered) / len(covered) if covered else None,
        "full_coverage": all(covered) if covered else None,
    }


def evaluate(
    rows: list[dict],
    questions: list[dict],
    index: ParentChildIndex,
    vectors: dict,
    config: dict,
    tokenizer,
    top_k: int,
    context_budget: int,
) -> tuple[list[dict], dict]:
    def vector(text):
        key = digest(
            {
                "provider": provider_identity(config),
                "text": text,
            }
        )
        value = vectors[key]
        return value / np.linalg.norm(value)

    matrix = np.stack([vector(row["text"]) for row in rows])
    names = {
        p["document_name"]: r["document"]
        for r in rows
        if r["arm"] == "child"
        for p in [index.parents[r["parent_id"]]]
    }
    results = []
    for question in questions:
        scores = matrix @ vector(question["query"])
        query_models = models(question["query"])
        for arm, filtered in [
            ("legacy", False),
            ("legacy", True),
            ("child", False),
            ("child", True),
        ]:
            eligible = [
                i
                for i, row in enumerate(rows)
                if row["arm"] == arm
                and (
                    not filtered
                    or not query_models
                    or set(query_models).intersection(row["models"])
                )
            ]
            ranked = sorted(eligible, key=lambda i: (-float(scores[i]), rows[i]["id"]))[
                :top_k
            ]
            selected = [rows[i] for i in ranked]
            evidence, groups, remaining = [], [], context_budget
            candidate_rejections = []
            capability_checks = (
                index.capabilities.check_query(question["query"])
                if arm == "child" and filtered
                else []
            )
            declarations = {
                item["id"]: item
                for check in capability_checks
                for item in check["evidence"]
            }
            for item in declarations.values():
                cost = len(tokenizer.encode(item["content"]))
                included = cost <= remaining
                if included:
                    evidence.append(
                        {
                            **item,
                            "document": names[item["document_name"]],
                            "kind": "capability_declaration",
                        }
                    )
                    remaining -= cost
                groups.append(
                    {"hit_id": item["id"], "included": included, "tokens": cost}
                )
            for row in selected:
                if arm == "legacy":
                    group = [
                        {
                            "id": row["id"],
                            "document": row["document"],
                            "content": row["content"],
                            "document_name": row.get("document_name"),
                            "pages": None,
                        }
                    ]
                else:
                    if filtered and query_models:
                        resolved = None
                        for model in sorted(
                            set(query_models).intersection(row["models"])
                        ):
                            candidate = index.resolve(
                                [row["id"]],
                                model=model,
                                required_features=query_features(question["query"]),
                            )
                            candidate_rejections.extend(
                                {**item, "model": model}
                                for item in candidate["rejected"]
                            )
                            if candidate["accepted_child_ids"]:
                                resolved = candidate
                                break
                        parents = resolved["parents"] if resolved else []
                    else:
                        parent = index.parents[row["parent_id"]]
                        parent_ids = list(
                            dict.fromkeys(
                                [*parent["context_parent_ids"], parent["parent_id"]]
                            )
                        )
                        parents = [index.parents[key] for key in parent_ids]
                    group = [
                        {
                            "id": p["parent_id"],
                            "document": names[p["document_name"]],
                            "content": evidence_text(p),
                            "table_notes": p.get("table_notes", []),
                            "document_name": p["document_name"],
                            "pages": p.get("pages", []),
                        }
                        for p in parents
                    ]
                existing = {e["id"] for e in evidence}
                group = [item for item in group if item["id"] not in existing]
                cost = sum(len(tokenizer.encode(item["content"])) for item in group)
                included = bool(group) and cost <= remaining
                if included:
                    evidence.extend(group)
                    remaining -= cost
                groups.append(
                    {"hit_id": row["id"], "included": included, "tokens": cost}
                )
            metrics = evidence_metrics(question, evidence)
            child_metrics = evidence_metrics(question, selected)
            results.append(
                {
                    "id": question["id"],
                    "query": question["query"],
                    "test_kind": question["test_kind"],
                    "arm": arm + ("_model_filter" if filtered else "_raw"),
                    "selected": [
                        {
                            "id": rows[i]["id"],
                            "document": rows[i]["document"],
                            "score": float(scores[i]),
                            "parent_id": rows[i]["parent_id"],
                            "models": rows[i]["models"],
                            "feature_domain": rows[i].get("feature_domain", "unknown"),
                        }
                        for i in ranked
                    ],
                    "query_models": query_models,
                    "capability_checks": capability_checks,
                    "candidate_rejections": candidate_rejections,
                    "groups": groups,
                    "context_tokens": context_budget - remaining,
                    "evidence": evidence,
                    "metrics": metrics,
                    "hit_text_metrics": child_metrics,
                }
            )
    summary = {}
    for arm in sorted({r["arm"] for r in results}):
        evaluated = [
            r
            for r in results
            if r["arm"] == arm
            and r["test_kind"] == "retrieval"
            and r["metrics"]["coverage"] is not None
        ]
        if not evaluated:
            raise ValueError("No retrieval questions with source evidence to evaluate")
        summary[arm] = {
            "queries": len(evaluated),
            "mean_anchor_coverage": sum(r["metrics"]["coverage"] for r in evaluated)
            / len(evaluated),
            "full_coverage_queries": sum(
                r["metrics"]["full_coverage"] for r in evaluated
            ),
            "zero_coverage_queries": sum(
                r["metrics"]["covered_anchors"] == 0 for r in evaluated
            ),
            "mean_hit_text_anchor_coverage": sum(
                r["hit_text_metrics"]["coverage"] for r in evaluated
            )
            / len(evaluated),
        }
    return results, summary


def write_report(output: Path, summary: dict, results: list[dict]) -> None:
    labels = {
        "legacy_raw": "旧切块：直接召回",
        "legacy_model_filter": "旧切块：文件名型号过滤",
        "child_raw": "新子块：召回后取父块",
        "child_model_filter": "新子块：型号过滤、功能核验及父块回取",
    }
    lines = [
        "# 父子块检索对照结果",
        "",
        "本报告仅评价检索证据，不调用回答生成模型。引文覆盖按文档来源与规范化原文匹配计算，不等同于语义正确率或答案评分。",
        "",
        f"Embedding 模型：{summary['provider']['model']}；top-k：{summary['top_k']}；上下文预算：{summary['context_budget']}（本地 cl100k token）。",
        "",
        "| 方案 | 检索题数 | 平均引文覆盖 | 全覆盖题数 | 零覆盖题数 | 原始命中片段覆盖 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, metrics in summary["retrieval_metrics"].items():
        lines.append(
            f"| {labels[key]} | {metrics['queries']} | {metrics['mean_anchor_coverage']:.1%} | {metrics['full_coverage_queries']} | {metrics['zero_coverage_queries']} | {metrics['mean_hit_text_anchor_coverage']:.1%} |"
        )
    lines.extend(
        [
            "",
            "旧版与新版使用相同 embedding 模型和问题。新版同时改变了解析、切分与索引文本元数据，因此差异不能全部归因于父块扩展。原始片段覆盖与取回父块后的覆盖分别报告。v3 的新版过滤组还加入明确功能限制声明的核验与证据，不能将其变化全部归因于向量召回或父块扩展。",
            "",
            "## 逐题对照",
            "",
            "| 问题 | 旧版直接召回 | 新版型号过滤及父块回取 | 差值 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    by_question = {}
    for row in results:
        by_question.setdefault(row["id"], {})[row["arm"]] = row
    for key, arms in by_question.items():
        old, new = arms["legacy_raw"], arms["child_model_filter"]
        if old["test_kind"] == "retrieval" and old["metrics"]["coverage"] is not None:
            left, right = old["metrics"]["coverage"], new["metrics"]["coverage"]
            lines.append(f"| {key} | {left:.1%} | {right:.1%} | {right - left:+.1%} |")
    usage = summary["recorded_embedding_totals"]
    lines.extend(
        [
            "",
            "## 请求记录与限制",
            "",
            f"成功请求累计 {usage['successful_requests']} 次，输入累计 {usage['inputs']} 条。服务端 usage：`{json.dumps(usage['provider_usage'], ensure_ascii=False)}`。",
            "",
            "6 道证据边界题保留逐题检索记录，不纳入普通召回汇总。当前没有验证答案生成、追问策略或图谱消歧。父块、表格附加脚注与前置说明按完整组纳入预算，超出预算时整组跳过，不静默截断条件。功能核验仅识别明确的排他功能清单及查询中的功能词；未知保持未知，多型号和多步骤召回策略尚未修复。",
            "",
            "完整命中、分数、父块、预算跳过项与原文见 results.jsonl。原知识库只读；本目录保存独立缓存。",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-chunks", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--tokenizer-file", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-remote-embedding", action="store_true")
    parser.add_argument("--reuse-embedding-directory", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--context-budget", type=int, default=8000)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.top_k, args.context_budget) < 1:
        raise ValueError("Batch size, top-k and context budget must be positive")
    config = embedding_config(args.env_file)
    if args.allow_remote_embedding and not config["api_key"]:
        raise ValueError("No configured embedding API key; no requests made")
    tokenizer = load_local_tokenizer(args.tokenizer_file)
    rows, questions, index = load_data(args.legacy_chunks, args.bundle, args.questions)
    inputs = {
        "legacy_sha256": file_hash(args.legacy_chunks),
        "questions_sha256": file_hash(args.questions),
        "bundles": {
            str(d.resolve()): file_hash(d / "summary.json") for d in args.bundle
        },
        "top_k": args.top_k,
        "context_budget": args.context_budget,
    }
    reusable = reusable_embeddings(args.reuse_embedding_directory, config)
    if args.reuse_embedding_directory is not None:
        inputs["reuse_embedding_source"] = {
            "directory": str(args.reuse_embedding_directory.resolve()),
            "plan_sha256": file_hash(
                args.reuse_embedding_directory / "experiment.json"
            ),
            "cache_sha256": file_hash(
                args.reuse_embedding_directory / "embedding_cache.jsonl"
            ),
        }
    plan, jobs = make_plan(rows, questions, config, inputs, tokenizer)
    if args.output.exists():
        if (
            not args.resume
            or json.loads((args.output / "experiment.json").read_text(encoding="utf-8"))
            != plan
        ):
            raise ValueError("Output exists or resume plan changed")
    else:
        args.output.mkdir(parents=True)
        (args.output / "experiment.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                k: plan[k]
                for k in (
                    "documents",
                    "queries",
                    "unique_embedding_inputs",
                    "local_cl100k_input_tokens",
                    "max_local_input_tokens",
                )
            }
        ),
        flush=True,
    )
    available = set(reusable) | set(
        read_embedding_cache(args.output / "embedding_cache.jsonl", config)
    )
    print(
        json.dumps(
            {
                "reusable_inputs": sum(job["key"] in available for job in jobs),
                "remaining_embedding_inputs": sum(
                    job["key"] not in available for job in jobs
                ),
            }
        ),
        flush=True,
    )
    if not args.allow_remote_embedding:
        print("Preflight only; no embedding requests made.", flush=True)
        return 0
    with experiment_lock(args.output):
        vectors, calls = cached_embeddings(
            jobs, config, args.output, args.batch_size, args.reuse_embedding_directory
        )
    results, summary = evaluate(
        rows,
        questions,
        index,
        vectors,
        config,
        tokenizer,
        args.top_k,
        args.context_budget,
    )
    (args.output / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    run_summary = {
        "plan_hash": plan["plan_hash"],
        "provider": plan["provider"],
        "calls_this_run": calls,
        "recorded_embedding_totals": recorded_requests(args.output),
        "retrieval_metrics": summary,
        "top_k": args.top_k,
        "context_budget": args.context_budget,
        "generation_model_calls": 0,
        "limitations": [
            "Exact normalized source-quote coverage, not semantic answer grading.",
            "Boundary questions are reported but excluded from retrieval aggregates.",
            "Parsing, chunking and metadata differ between arms; this does not isolate parent expansion alone.",
            "The child model-filter arm also checks explicit exclusive feature lists; this is no longer dense retrieval plus model filtering alone.",
            "Unknown capability remains unknown; cross-section procedure completion and per-model query decomposition are not implemented.",
        ],
    }
    write_report(args.output, run_summary, results)
    (args.output / "summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
