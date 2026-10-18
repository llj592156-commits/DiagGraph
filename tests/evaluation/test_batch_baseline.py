"""Offline contracts for batch collection, experiment identity and recovery."""

import builtins
import json
import threading
from urllib import error

import pytest

from lightrag.evaluation import batch_baseline as batch

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("applied,expected", [(True, "success"), (False, "unverified")])
def test_requested_rerank_fallback_is_not_reported_as_success(applied, expected):
    body = {
        "status": "success",
        "data": {"chunks": []},
        "metadata": {
            "retrieval": {
                "rerank_requested": True,
                "rerank_applied": applied,
                "candidate_count": 20,
            }
        },
    }
    assert batch.outcome("retrieval", 200, body) == expected


@pytest.mark.parametrize(
    "metadata", [None, "invalid", [], {"retrieval": None}, {"retrieval": "invalid"}]
)
def test_malformed_optional_rerank_metadata_does_not_break_collection(metadata):
    assert (
        batch.outcome(
            "retrieval", 200, {"status": "success", "data": {}, "metadata": metadata}
        )
        == "success"
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def post(self, endpoint, payload):
        self.calls.append((endpoint, payload))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def question(identifier="Q01"):
    return {
        "id": identifier,
        "query": "L3251 用什么墨水？",
        "group": "facts",
        "expected_points": ["004"],
        "evidence": [{"quote": "GOLD MUST NOT BE SENT"}],
    }


def config(stage="answer", modes=None, questions=None):
    modes = modes or ["naive"]
    questions = questions or [question()]
    return {
        "stage": stage,
        "modes": modes,
        "repeats": 1,
        "parameters": batch.DEFAULT_PARAMETERS,
        "planned_requests": len(questions) * len(modes),
    }


def answer(text="答案"):
    return 200, {
        "response": text,
        "llm_generated": True,
        "references": [
            {"reference_id": "1", "file_path": "手册.pdf", "content": ["原文"]}
        ],
    }


def test_only_query_and_parameters_are_sent_to_api():
    client = FakeClient([answer()])
    row = batch.collect_one(
        client, "answer", question(), "mix", 1, 1, batch.DEFAULT_PARAMETERS
    )
    endpoint, payload = client.calls[0]
    assert endpoint == "/query"
    assert set(payload) == set(batch.DEFAULT_PARAMETERS) | {"query", "mode"}
    assert "GOLD" not in json.dumps(payload)
    assert row["response"]["references"][0]["content"] == ["原文"]
    assert row["manual_review"]["coverage_0_to_2"] is None


@pytest.mark.parametrize(
    "stage,status,body,expected",
    [
        ("answer", 200, {"response": "答案", "llm_generated": True}, "success"),
        ("answer", 200, {"response": "fallback", "llm_generated": False}, "unverified"),
        ("answer", 200, {"response": "old server"}, "unverified"),
        ("answer", 200, {"detail": "bad"}, "error"),
        ("answer", 500, {"response": "failure", "llm_generated": True}, "error"),
        ("retrieval", 200, {"status": "success", "data": {}}, "success"),
        (
            "retrieval",
            200,
            {"status": "failure", "metadata": {"failure_reason": "no_results"}},
            "no_results",
        ),
        ("retrieval", 200, {"status": "failure", "metadata": None}, "error"),
        ("retrieval", 200, {"status": "failure", "message": "backend error"}, "error"),
        ("retrieval", 200, [], "error"),
        ("retrieval", None, None, "error"),
    ],
)
def test_no_evidence_and_service_failures_remain_distinct(
    stage, status, body, expected
):
    assert batch.outcome(stage, status, body) == expected


def test_resume_skips_completed_requests_and_preserves_attempts(tmp_path):
    client = FakeClient([answer()])
    assert batch.run_batch(client, tmp_path, config(), [question()]) == 0
    assert batch.run_batch(client, tmp_path, config(), [question()]) == 0
    assert len(client.calls) == 1
    assert len(list((tmp_path / "records").glob("*.json"))) == 1
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "答案" in text and "手册.pdf" in text and "原文" in text
    assert "没有自动评出最佳模式" in text


def test_retry_errors_preserves_failed_attempt_and_replaces_latest_view(tmp_path):
    client = FakeClient([(500, {"detail": "failed"}), answer("recovered")])
    assert batch.run_batch(client, tmp_path, config(), [question()]) == 1
    assert batch.run_batch(client, tmp_path, config(), [question()]) == 1
    assert len(client.calls) == 1
    assert (
        batch.run_batch(client, tmp_path, config(), [question()], retry_errors=True)
        == 0
    )
    assert len(list((tmp_path / "records").glob("*.json"))) == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["attempt"] == 2
    assert rows[0]["response"]["response"] == "recovered"


def test_slow_request_reports_start_and_wait_before_result(
    tmp_path, monkeypatch, capsys
):
    observed_wait = threading.Event()
    original_print = builtins.print
    original_progress = batch.request_progress

    def observe_print(*args, **kwargs):
        original_print(*args, **kwargs)
        if args and "Waiting for Q01_naive_r1" in str(args[0]):
            observed_wait.set()

    class SlowClient:
        timeout = 600

        def post(self, endpoint, payload):
            assert observed_wait.wait(2), (
                "No waiting message while request was outstanding"
            )
            return answer()

    monkeypatch.setattr(builtins, "print", observe_print)
    monkeypatch.setattr(
        batch, "request_progress", lambda key: original_progress(key, interval=0.005)
    )
    assert batch.run_batch(SlowClient(), tmp_path, config(), [question()]) == 0
    text = capsys.readouterr().out
    assert (
        text.index("Starting Q01_naive_r1")
        < text.index("Waiting for Q01_naive_r1")
        < text.index("success")
    )
    assert not any(worker.name == "batch-progress" for worker in threading.enumerate())


def test_timeout_record_retains_effective_timeout_and_warns_about_server_work(tmp_path):
    client = FakeClient([TimeoutError()])
    client.timeout = 600
    assert batch.run_batch(client, tmp_path, config(), [question()]) == 1
    record = next(iter(batch.latest_records(tmp_path / "records").values()))
    assert record["http_timeout_seconds"] == 600
    assert "server may still be processing" in record["error"]
    assert record["outcome"] == "error"


def test_cli_defaults_to_longer_whole_query_wait(tmp_path, monkeypatch):
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(json.dumps(question()), encoding="utf-8")

    def inspect_run(client, output, settings, questions, retry_errors):
        assert client.timeout == 600
        return 0

    monkeypatch.setattr(batch, "run_batch", inspect_run)
    assert (
        batch.main(["--dataset", str(dataset), "--output", str(tmp_path / "output")])
        == 0
    )


def test_valid_no_results_is_completed_and_not_automatically_retried(tmp_path):
    client = FakeClient(
        [(200, {"status": "failure", "metadata": {"failure_reason": "no_results"}})]
    )
    settings = config(stage="retrieval")
    assert batch.run_batch(client, tmp_path, settings, [question()]) == 0
    assert (
        batch.run_batch(client, tmp_path, settings, [question()], retry_errors=True)
        == 0
    )
    assert len(client.calls) == 1
    assert client.calls[0][0] == "/query/data"


def test_changed_configuration_does_not_mix_results(tmp_path):
    client = FakeClient([answer()])
    batch.run_batch(client, tmp_path, config(), [question()])
    with pytest.raises(ValueError, match="configuration differs"):
        batch.run_batch(
            client, tmp_path, {**config(), "label": "new models"}, [question()]
        )
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "failure", [(401, {}), (403, {}), (429, {}), error.URLError("SECRET")]
)
def test_auth_or_connection_failure_stops_without_losing_result(tmp_path, failure):
    questions = [question(), question("Q02")]
    client = FakeClient([failure])
    assert (
        batch.run_batch(client, tmp_path, config(questions=questions), questions) == 1
    )
    assert len(client.calls) == 1
    assert (tmp_path / "summary.json").exists()
    assert "SECRET" not in (tmp_path / "results.jsonl").read_text(encoding="utf-8")


def test_interruption_retains_finished_jobs_and_rebuilds_report(tmp_path):
    questions = [question(), question("Q02")]
    client = FakeClient([answer(), KeyboardInterrupt()])
    # KeyboardInterrupt is a BaseException, so inject it outside FakeClient.post.
    original = client.post

    def post(endpoint, payload):
        if len(client.calls) == 1:
            raise KeyboardInterrupt
        return original(endpoint, payload)

    client.post = post
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(client, tmp_path, config(questions=questions), questions)
    assert len(batch.latest_records(tmp_path / "records")) == 1
    assert (tmp_path / "report.md").exists()


def test_malformed_reference_fields_do_not_discard_other_answers(tmp_path):
    questions = [question(), question("Q02")]
    client = FakeClient(
        [
            (
                200,
                {
                    "response": "first",
                    "llm_generated": True,
                    "references": {"bad": True},
                },
            ),
            answer("second"),
        ]
    )
    assert (
        batch.run_batch(client, tmp_path, config(questions=questions), questions) == 0
    )
    assert len(batch.latest_records(tmp_path / "records")) == 2
    assert "second" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_dataset_validation_rejects_duplicate_unsafe_and_unknown_ids(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(question()) + "\n" + json.dumps(question()), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        batch.load_questions(path, None)
    path.write_text(json.dumps(question("../escape")), encoding="utf-8")
    with pytest.raises(ValueError, match="id must"):
        batch.load_questions(path, None)
    path.write_text(
        "\ufeff" + json.dumps(question(), ensure_ascii=False), encoding="utf-8"
    )
    assert batch.load_questions(path, ["Q01"])[0]["query"] == question()["query"]
    with pytest.raises(ValueError, match="Unknown"):
        batch.load_questions(path, ["Q99"])


def test_dry_run_never_logs_in_sends_requests_or_writes_output(
    tmp_path, monkeypatch, capsys
):
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(json.dumps(question()), encoding="utf-8")
    monkeypatch.setattr(
        batch.APIClient, "post", lambda *args: pytest.fail("network call")
    )
    monkeypatch.setattr(
        batch.getpass, "getpass", lambda *args: pytest.fail("password prompt")
    )
    output = tmp_path / "output"
    assert (
        batch.main(
            [
                "--dataset",
                str(dataset),
                "--output",
                str(output),
                "--dry-run",
                "--username",
                "demo",
                "--modes",
                "naive",
                "mix",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["planned_requests"] == 2
    assert not output.exists()


def test_boundary_questions_can_be_selected_separately(tmp_path, capsys):
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        json.dumps(question())
        + "\n"
        + json.dumps({**question("Q31"), "test_kind": "boundary"}),
        encoding="utf-8",
    )
    assert (
        batch.main(
            [
                "--dataset",
                str(dataset),
                "--output",
                str(tmp_path / "output"),
                "--question-kind",
                "boundary",
                "--dry-run",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["question_ids"] == ["Q31"]
    assert plan["planned_requests"] == 5


def test_http_client_uses_utf8_and_auth_headers_without_leaking_to_output(monkeypatch):
    captured = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return '{"response": "中文"}'.encode()

    class Opener:
        def open(self, req, timeout):
            captured.append(req)
            return Response()

    monkeypatch.setattr(batch.request, "build_opener", lambda *args: Opener())
    client = batch.APIClient(
        "http://localhost:9621/prefix/", 12, api_key="secret-key", token="secret-token"
    )
    assert client.post("/query", {"query": "中文"}) == (200, {"response": "中文"})
    req = captured[0]
    assert req.full_url == "http://localhost:9621/prefix/query"
    assert req.get_header("X-api-key") == "secret-key"
    assert req.get_header("Authorization") == "Bearer secret-token"
    assert json.loads(req.data) == {"query": "中文"}
    assert (
        batch.NoRedirect().redirect_request(None, None, 302, "", {}, "https://other")
        is None
    )


def test_mode_rotation_and_repetition_do_not_drop_jobs():
    questions = [question(), question("Q02")]
    jobs = list(batch.make_jobs(questions, ["naive", "mix"], 2))
    assert len(jobs) == 8
    assert len({(q["id"], mode, repeat) for q, mode, repeat in jobs}) == 8
    assert [mode for _, mode, _ in jobs[:4]] == ["naive", "mix", "mix", "naive"]


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "http://u:p@localhost", "http://localhost?key=secret"]
)
def test_base_url_disallows_embedded_secrets_and_non_http_protocols(url):
    with pytest.raises(ValueError, match="API URL"):
        batch.APIClient(url, 10)
