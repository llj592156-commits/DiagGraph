"""Collect comparable answers or retrieval evidence from a running LightRAG API.

Run this file directly; only the Python standard library is required. Each
output directory belongs to one immutable experiment configuration. Resume
with the same arguments; use a new directory after changing server models,
knowledge, prompts, or other settings not visible in the request payload.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
import re
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request

MODES = ("naive", "local", "global", "hybrid", "mix")
DEFAULT_HTTP_TIMEOUT = 600
DEFAULT_PARAMETERS = {
    "top_k": 20,
    "chunk_top_k": 5,
    "max_entity_tokens": 2000,
    "max_relation_tokens": 2000,
    "max_total_tokens": 8000,
    "enable_rerank": False,
    "conversation_history": [],
    "stream": False,
    "include_references": True,
    "include_chunk_content": True,
}


def write_json(path: Path, value: Any) -> None:
    """Publish a complete UTF-8 JSON file; never leave a partial resume record."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_questions(path: Path, ids: list[str] | None) -> list[dict[str, Any]]:
    questions = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        identifier, query = item.get("id"), item.get("query")
        if not isinstance(identifier, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", identifier
        ):
            raise ValueError(
                f"Line {number}: id must contain only letters, digits, _ or -"
            )
        if identifier in seen:
            raise ValueError(f"Duplicate question id: {identifier}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Line {number}: query must be a nonempty string")
        seen.add(identifier)
        questions.append(item)
    if ids:
        unknown = set(ids) - seen
        if unknown:
            raise ValueError(f"Unknown question ids: {', '.join(sorted(unknown))}")
        questions = [item for item in questions if item["id"] in ids]
    if not questions:
        raise ValueError("The selected question set is empty")
    return questions


class NoRedirect(request.HTTPRedirectHandler):
    """Refuse redirects so credentials cannot leave the configured API URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class APIClient:
    def __init__(
        self, base_url: str, timeout: float, api_key: str = "", token: str = ""
    ):
        parts = parse.urlsplit(base_url)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
        ):
            raise ValueError(
                "API URL must be http(s), without credentials, query or fragment"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self.token = token
        self.opener = request.build_opener(NoRedirect())

    def post(self, endpoint: str, payload: dict, form: bool = False) -> tuple[int, Any]:
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
            if form
            else "application/json"
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = (
            parse.urlencode(payload).encode("utf-8")
            if form
            else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        )
        req = request.Request(
            self.base_url + endpoint, data=data, headers=headers, method="POST"
        )
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                status, body = response.status, response.read()
        except error.HTTPError as exc:
            with exc:
                status, body = exc.code, exc.read()
        decoded = body.decode("utf-8", errors="replace")
        try:
            return status, json.loads(decoded)
        except ValueError:
            return status, {"unparsed_body": decoded}

    def login(self, username: str, password: str) -> None:
        status, body = self.post(
            "/login", {"username": username, "password": password}, form=True
        )
        if status != 200 or not isinstance(body, dict) or not body.get("access_token"):
            # Login responses are never persisted or echoed: they contain credentials.
            raise ValueError(
                f"Login failed (HTTP {status}); check credentials and authentication mode"
            )
        self.token = body["access_token"]


def outcome(stage: str, http_status: int | None, body: Any) -> str:
    """Separate transport/service errors from valid retrieval with no evidence."""
    if http_status is None or not 200 <= http_status < 300:
        return "error"
    if not isinstance(body, dict):
        return "error"
    if stage == "answer":
        if not isinstance(body.get("response"), str):
            return "error"
        # /query can hide a failed core query behind a canned HTTP-200 answer.
        # Treat no-generation as unresolved rather than guessing why it happened.
        return "success" if body.get("llm_generated") is True else "unverified"
    if body.get("status") == "success" and isinstance(body.get("data"), dict):
        return "success"
    metadata = body.get("metadata")
    if isinstance(metadata, dict) and metadata.get("failure_reason") == "no_results":
        return "no_results"
    return "error"


def dict_items(value: Any) -> list[dict]:
    """Keep malformed API fields reviewable without breaking the whole report."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def collect_one(
    client: APIClient,
    stage: str,
    question: dict,
    mode: str,
    repeat: int,
    attempt: int,
    parameters: dict,
) -> dict:
    # Explicit allowlist: gold answers, evidence pages and rubric never reach the server.
    payload = {**parameters, "query": question["query"], "mode": mode}
    endpoint = "/query" if stage == "answer" else "/query/data"
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    status, body, failure = None, None, None
    try:
        status, body = client.post(endpoint, payload)
    except TimeoutError:
        failure = (
            "TimeoutError: HTTP request timed out; the server may still be processing. "
            "Check server logs/queues before retrying."
        )
    except (error.URLError, OSError) as exc:
        # Do not echo request headers, URL exception representations or credentials.
        failure = f"{type(exc).__name__}: request failed; check server/network/timeout"
    return {
        "question_id": question["id"],
        "query": question["query"],
        "group": question.get("group", "ungrouped"),
        "test_kind": question.get("test_kind", "retrieval"),
        "mode": mode,
        "repeat": repeat,
        "attempt": attempt,
        "stage": stage,
        "started_at_utc": started,
        "endpoint": endpoint,
        "request": payload,
        "http_status": status,
        "http_timeout_seconds": getattr(client, "timeout", None),
        "latency_ms": round((time.perf_counter() - clock) * 1000, 2),
        "outcome": outcome(stage, status, body),
        "error": failure,
        "response": body,
        "cache_state": "unknown",
        "token_usage": None,
        "manual_review": {
            "coverage_0_to_2": None,
            "applicability_pass": None,
            "citation_support_pass": None,
            "notes": "",
        },
    }


@contextmanager
def request_progress(key: str, interval: float = 15):
    """Show elapsed waiting time without issuing another server request.

    A heartbeat only reports that the HTTP call is outstanding; it does not
    claim that the server or upstream model is making progress.
    """
    stopped = threading.Event()
    started = time.perf_counter()

    def heartbeat() -> None:
        while not stopped.wait(interval):
            print(
                f"  Waiting for {key}: {time.perf_counter() - started:.0f}s elapsed; "
                "HTTP response not received yet.",
                flush=True,
            )

    worker = threading.Thread(target=heartbeat, name="batch-progress", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()


def make_jobs(questions: list[dict], modes: list[str], repeats: int):
    for repeat in range(1, repeats + 1):
        # Rotate mode order across questions and repetitions to reduce a fixed-order bias.
        for index, question in enumerate(questions):
            offset = (index + repeat - 1) % len(modes)
            for mode in modes[offset:] + modes[:offset]:
                yield question, mode, repeat


def latest_records(directory: Path) -> dict[str, dict]:
    latest = {}
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = f"{record['question_id']}_{record['mode']}_r{record['repeat']}"
        if key not in latest or record["attempt"] > latest[key]["attempt"]:
            latest[key] = {**record, "record_file": path.name}
    return latest


def render_report(output: Path, config: dict) -> dict:
    records = list(latest_records(output / "records").values())
    records.sort(key=lambda r: (r["question_id"], r["mode"], r["repeat"]))
    lines = [
        "# LightRAG 批量测试结果",
        "",
        f"阶段：{config['stage']}；计划请求：{config['planned_requests']}；已记录：{len(records)}。",
        "",
        "质量评分尚未完成。成功请求不等于答案正确；没有自动评出最佳模式。",
        "",
        "answer 阶段保存同一次请求的答案、引用和引用原文；retrieval 阶段保存结构化检索结果。",
        "两阶段分别运行会重新检索，不能假设它们使用相同证据。",
        "",
        "耗时为客户端测量值；缓存状态与真实 token 用量未知，不作零值处理。",
        "",
        "## 运行统计",
        "",
        "| 模式 | 成功 | 无结果 | 未验证 | 错误 | 成功请求耗时中位数 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    summary = {
        "planned_requests": config["planned_requests"],
        "recorded_jobs": len(records),
        "modes": {},
    }
    for mode in config["modes"]:
        rows = [r for r in records if r["mode"] == mode]
        counts = {
            status: sum(r["outcome"] == status for r in rows)
            for status in ("success", "no_results", "unverified", "error")
        }
        times = [r["latency_ms"] for r in rows if r["outcome"] == "success"]
        median = round(statistics.median(times), 2) if times else None
        summary["modes"][mode] = {**counts, "median_success_latency_ms": median}
        lines.append(
            f"| {mode} | {counts['success']} | {counts['no_results']} | "
            f"{counts['unverified']} | {counts['error']} | {median if median is not None else '未知'} |"
        )
    for row in records:
        body = row["response"] if isinstance(row["response"], dict) else {}
        data = body.get("data") if isinstance(body.get("data"), dict) else {}
        lines += [
            "",
            f"## {row['question_id']} · {row['mode']} · 第 {row['repeat']} 次",
            "",
            row["query"],
            "",
            f"状态：{row['outcome']}；耗时：{row['latency_ms']} ms。",
            "",
            f"[原始结果](records/{row['record_file']})",
            "",
            "人工评分：证据覆盖 __/2；型号与条件 __；引用支持 __；备注 __。",
            "",
        ]
        if config["stage"] == "answer":
            lines += ["### 答案", "", str(body.get("response", "无可用答案")), ""]
            refs = dict_items(body.get("references"))
        else:
            lines += [
                f"实体 {len(dict_items(data.get('entities')))}；关系 {len(dict_items(data.get('relationships')))}；"
                f"片段 {len(dict_items(data.get('chunks')))}。",
                "",
            ]
            refs = dict_items(data.get("references"))
        lines += ["### 引用与原文", ""]
        for ref in refs:
            lines += [
                f"- [{ref.get('reference_id', '')}] {ref.get('file_path', '')}",
                "",
            ]
            content = ref.get("content")
            content = (
                content
                if isinstance(content, list)
                else [content]
                if isinstance(content, str)
                else []
            )
            for text in content:
                lines += ["> " + str(text).replace("\n", "\n> "), ""]
        for chunk in dict_items(data.get("chunks")):
            lines += [
                f"片段 `{chunk.get('chunk_id', '')}` · {chunk.get('file_path', '')}",
                "",
                "> " + str(chunk.get("content", "")).replace("\n", "\n> "),
                "",
            ]
        if row["error"]:
            lines += [row["error"], ""]
    # This view is rebuilt from immutable attempt files; edit a separate review file.
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    write_json(output / "summary.json", summary)
    return summary


def run_batch(
    client: APIClient,
    output: Path,
    config: dict,
    questions: list[dict],
    retry_errors: bool = False,
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "experiment.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != config:
            raise ValueError(
                "Output configuration differs; choose a new --output directory"
            )
    else:
        if any(output.iterdir()):
            raise ValueError(
                "Output directory is not empty and has no experiment manifest"
            )
        write_json(manifest, config)
    directory = output / "records"
    directory.mkdir(exist_ok=True)
    latest = latest_records(directory)
    pending = []
    for question, mode, repeat in make_jobs(
        questions, config["modes"], config["repeats"]
    ):
        key = f"{question['id']}_{mode}_r{repeat}"
        old = latest.get(key)
        if old and not (retry_errors and old["outcome"] in {"error", "unverified"}):
            continue
        pending.append((question, mode, repeat, key, old))
    print(f"Planned: {config['planned_requests']}; pending: {len(pending)}", flush=True)
    try:
        for index, (question, mode, repeat, key, old) in enumerate(pending, 1):
            attempt = old["attempt"] + 1 if old else 1
            print(
                f"[{index}/{len(pending)}] Starting {key} "
                f"(HTTP socket timeout: {getattr(client, 'timeout', 'unknown')}s)",
                flush=True,
            )
            with request_progress(key):
                row = collect_one(
                    client,
                    config["stage"],
                    question,
                    mode,
                    repeat,
                    attempt,
                    config["parameters"],
                )
            write_json(directory / f"{key}_a{attempt}.json", row)
            print(
                f"[{index}/{len(pending)}] {key}: {row['outcome']} ({row['latency_ms']} ms)",
                flush=True,
            )
            if row["http_status"] in {401, 403}:
                print(
                    "Authentication rejected; stopped. Fix credentials and use --retry-errors.",
                    flush=True,
                )
                break
            if row["http_status"] == 429:
                print(
                    "Rate limited; stopped. Wait before resuming with --retry-errors.",
                    flush=True,
                )
                break
            if row["http_status"] is None:
                print(
                    "Connection/timeout failure; stopped. The server may still be processing. "
                    "Check its logs/queues before using --retry-errors.",
                    flush=True,
                )
                break
    finally:
        summary = render_report(output, config)
    failures = sum(
        stats["error"] + stats["unverified"] for stats in summary["modes"].values()
    )
    return (
        1 if failures or summary["recorded_jobs"] != config["planned_requests"] else 0
    )


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, required=True, help="UTF-8 JSONL, id/query required"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="One directory per experiment"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:9621",
        help="Include any API mount prefix",
    )
    parser.add_argument("--stage", choices=("answer", "retrieval"), default="answer")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--ids", nargs="+", help="Question ids; omitted means all")
    parser.add_argument(
        "--question-kind",
        choices=("all", "retrieval", "boundary"),
        default="all",
        help="Separate normal retrieval questions from clarification/evidence-boundary cases",
    )
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=DEFAULT_HTTP_TIMEOUT,
        help="HTTP socket timeout in seconds (default: 600); includes waiting for server work",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Record models/knowledge/prompt revision without secrets",
    )
    parser.add_argument(
        "--rerank", action="store_true", help="Enable configured server reranker"
    )
    for key in (
        "top_k",
        "chunk_top_k",
        "max_entity_tokens",
        "max_relation_tokens",
        "max_total_tokens",
    ):
        parser.add_argument(
            "--" + key.replace("_", "-"),
            type=positive_int,
            default=DEFAULT_PARAMETERS[key],
        )
    parser.add_argument("--api-key-env", default="LIGHTRAG_API_KEY")
    parser.add_argument("--token-env", default="LIGHTRAG_TEST_TOKEN")
    parser.add_argument(
        "--username", help="Log in; password is prompted or read from --password-env"
    )
    parser.add_argument("--password-env", default="LIGHTRAG_TEST_PASSWORD")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry errors/unverified; keep earlier attempts",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan; no requests or output files"
    )
    args = parser.parse_args(argv)
    try:
        questions = load_questions(args.dataset, args.ids)
        if args.question_kind != "all":
            questions = [
                question
                for question in questions
                if question.get("test_kind", "retrieval") == args.question_kind
            ]
            if not questions:
                raise ValueError("No questions match --question-kind")
        modes = list(dict.fromkeys(args.modes))
        parameters = {**DEFAULT_PARAMETERS, "enable_rerank": args.rerank}
        for key in (
            "top_k",
            "chunk_top_k",
            "max_entity_tokens",
            "max_relation_tokens",
            "max_total_tokens",
        ):
            parameters[key] = getattr(args, key)
        client = APIClient(
            args.base_url,
            args.timeout,
            os.environ.get(args.api_key_env, ""),
            os.environ.get(args.token_env, ""),
        )
        config = {
            "schema_version": 1,
            "base_url": client.base_url,
            "stage": args.stage,
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "question_ids": [q["id"] for q in questions],
            "modes": modes,
            "repeats": args.repeats,
            "parameters": parameters,
            "label": args.label,
            "planned_requests": len(questions) * len(modes) * args.repeats,
        }
        if args.dry_run:
            print(json.dumps(config, ensure_ascii=False, indent=2))
            return 0
        if args.username:
            password = os.environ.get(args.password_env) or getpass.getpass(
                "LightRAG password: "
            )
            client.login(args.username, password)
        return run_batch(client, args.output, config, questions, args.retry_errors)
    except KeyboardInterrupt:
        print(
            "Interrupted. Completed request records are saved; resume with the same arguments.",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
