"""Dense experiments isolate corpora and cache only validated provider vectors."""

import base64
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from lightrag.evaluation import compare_parent_child_retrieval as compare

pytestmark = pytest.mark.offline


def config():
    return {
        "base_url": "https://example.test/v1",
        "model": "embed-test",
        "dimension": 2,
        "api_key": "must-not-be-persisted",
    }


def tokenizer():
    return SimpleNamespace(encode=lambda text: list(text))


def test_plan_has_no_keys_or_oracle_text_and_deduplicates_inputs():
    rows = [{"arm": "child", "text": "Manual"}, {"arm": "legacy", "text": "Manual"}]
    questions = [
        {
            "query": "Question",
            "expected_points": ["Secret answer"],
            "evidence": [{"quote": "Oracle quote"}],
        }
    ]
    plan, jobs = compare.make_plan(rows, questions, config(), {}, tokenizer())
    assert len(jobs) == 2
    encoded = json.dumps([plan, jobs])
    assert "must-not-be-persisted" not in encoded
    assert "Secret answer" not in encoded
    assert "Oracle quote" not in encoded


def test_cache_resume_makes_no_duplicate_calls(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        compare,
        "embed_batch",
        lambda config, texts: (
            calls.append(texts)
            or (np.ones((len(texts), 2), dtype="<f4"), {"prompt_tokens": len(texts)})
        ),
    )
    jobs = [{"key": str(i), "text": f"Text {i}"} for i in range(3)]
    first, stats = compare.cached_embeddings(jobs, config(), tmp_path, 2)
    second, resumed = compare.cached_embeddings(jobs, config(), tmp_path, 2)
    assert len(calls) == 2
    assert stats["requests"] == 2 and resumed["requests"] == 0
    assert resumed["cache_hits"] == 3
    assert all(np.array_equal(first[key], second[key]) for key in first)
    journal = [
        json.loads(x)
        for x in (tmp_path / "embedding_requests.jsonl").read_text().splitlines()
    ]
    assert sum(row["inputs"] for row in journal) == 3


def test_provider_response_order_and_shape_are_validated(monkeypatch):
    payload = {
        "data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}],
        "usage": {"prompt_tokens": 2},
    }
    captured = []

    def open_request(req, timeout):
        captured.append(json.loads(req.data))
        return io.StringIO(json.dumps(payload))

    monkeypatch.setattr(
        compare.request,
        "build_opener",
        lambda *args: SimpleNamespace(open=open_request),
    )
    matrix, usage = compare.embed_batch(config(), ["one", "two"])
    assert matrix.tolist() == [[1, 0], [0, 1]]
    assert captured[0]["input"] == ["one", "two"]
    assert "dimensions" not in captured[0]
    assert usage["prompt_tokens"] == 2
    payload["data"][1]["index"] = 1
    with pytest.raises(ValueError, match="indices"):
        compare.embed_batch(config(), ["one", "two"])


def test_invalid_cache_fails_instead_of_silently_reembedding(tmp_path):
    (tmp_path / "embedding_cache.jsonl").write_text('{"key":', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        compare.cached_embeddings([], config(), tmp_path, 2)


@pytest.mark.parametrize(
    "kind",
    ["incomplete_read", "ssl_eof", "wrapped_ssl_eof", "timeout", "reset", "http_503"],
)
def test_embedding_transport_retry_repeats_exact_batch_and_returns_only_complete_vectors(
    monkeypatch, kind
):
    from http.client import IncompleteRead
    from ssl import SSLEOFError

    failures = {
        "incomplete_read": IncompleteRead(b"partial private body", 100),
        "ssl_eof": SSLEOFError("private reason"),
        "wrapped_ssl_eof": compare.error.URLError(SSLEOFError("private reason")),
        "timeout": TimeoutError(),
        "reset": ConnectionResetError(),
        "http_503": compare.error.HTTPError(
            "https://example.test", 503, "private", {}, None
        ),
    }
    calls, sleeps = [], []
    payload = {
        "data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]
    }

    class BrokenResponse(io.StringIO):
        def read(self, *args):
            raise failures[kind]

    def open_request(req, timeout):
        calls.append((req.data, timeout))
        if len(calls) == 1:
            return BrokenResponse("")
        return io.StringIO(json.dumps(payload))

    monkeypatch.setattr(
        compare.request,
        "build_opener",
        lambda *args: SimpleNamespace(open=open_request),
    )
    monkeypatch.setattr(compare.time, "sleep", sleeps.append)
    matrix, _ = compare.embed_batch(config(), ["one", "two"])
    assert matrix.tolist() == [[1, 0], [0, 1]]
    assert len(calls) == 2 and calls[0] == calls[1]
    assert sleeps == [1]


@pytest.mark.parametrize(
    "kind", ["transient", "certificate", "wrapped_certificate", "auth", "bad_request"]
)
def test_embedding_exhaustion_or_permanent_failure_never_returns_placeholder_vectors(
    monkeypatch, kind
):
    from http.client import IncompleteRead
    from ssl import SSLCertVerificationError

    failure = {
        "transient": IncompleteRead(b"partial private body", 100),
        "certificate": SSLCertVerificationError("private"),
        "wrapped_certificate": compare.error.URLError(
            SSLCertVerificationError("private")
        ),
        "auth": compare.error.HTTPError(
            "https://example.test", 401, "private", {}, None
        ),
        "bad_request": compare.error.HTTPError(
            "https://example.test", 400, "private", {}, None
        ),
    }[kind]
    calls, sleeps = [], []

    def open_request(*args, **kwargs):
        calls.append(1)
        raise failure

    monkeypatch.setattr(
        compare.request,
        "build_opener",
        lambda *args: SimpleNamespace(open=open_request),
    )
    monkeypatch.setattr(compare.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError) as caught:
        compare.embed_batch(config(), ["input"])
    assert len(calls) == (4 if kind == "transient" else 1)
    assert sleeps == ([1, 2, 4] if kind == "transient" else [])
    assert "private" not in str(caught.value)


def test_source_anchor_score_requires_both_document_and_quote():
    q = {"evidence": [{"document": "A", "quote": "按 OK 按钮。"}]}
    assert (
        compare.evidence_metrics(q, [{"document": "B", "content": "按OK按钮"}])[
            "coverage"
        ]
        == 0
    )
    assert (
        compare.evidence_metrics(q, [{"document": "A", "content": "请按 OK 按钮。"}])[
            "coverage"
        ]
        == 1
    )


@pytest.mark.parametrize("key", ["", "must-not-be-persisted"])
def test_preflight_does_not_call_provider(monkeypatch, tmp_path, key):
    legacy, questions = tmp_path / "old.json", tmp_path / "questions.jsonl"
    legacy.write_text("{}", encoding="utf-8")
    questions.write_text("{}", encoding="utf-8")
    bundle = tmp_path / "A"
    bundle.mkdir()
    (bundle / "summary.json").write_text("{}")
    monkeypatch.setattr(
        compare, "embedding_config", lambda path: {**config(), "api_key": key}
    )
    monkeypatch.setattr(compare, "load_local_tokenizer", lambda path: tokenizer())
    monkeypatch.setattr(
        compare,
        "load_data",
        lambda *args: (
            [{"arm": "legacy", "text": "Manual"}],
            [{"query": "Question"}],
            None,
        ),
    )
    monkeypatch.setattr(
        compare,
        "cached_embeddings",
        lambda *args: pytest.fail("No provider access in preflight"),
    )
    args = [
        "--legacy-chunks",
        str(legacy),
        "--questions",
        str(questions),
        "--bundle",
        str(bundle),
        "--tokenizer-file",
        "unused",
        "--output",
        str(tmp_path / "output"),
    ]
    assert compare.main(args) == 0
    assert (tmp_path / "output" / "experiment.json").exists()
    with pytest.raises(ValueError, match="Output exists"):
        compare.main(args)


def test_dimension_policy_changes_cache_identity_but_api_key_does_not():
    rows, questions = [{"arm": "child", "text": "manual"}], [{"query": "question"}]
    base, _ = compare.make_plan(rows, questions, config(), {}, tokenizer())
    changed_key, _ = compare.make_plan(
        rows, questions, {**config(), "api_key": "different"}, {}, tokenizer()
    )
    dimensions, _ = compare.make_plan(
        rows, questions, {**config(), "send_dimensions": True}, {}, tokenizer()
    )
    assert base == changed_key
    assert base["job_keys"] != dimensions["job_keys"]


def test_experiment_lock_prevents_duplicate_provider_runs_and_releases(tmp_path):
    with compare.experiment_lock(tmp_path):
        assert (tmp_path / "embedding.lock").exists()
        with (
            pytest.raises(ValueError, match="locked"),
            compare.experiment_lock(tmp_path),
        ):
            pytest.fail("Second experiment acquired the same lock")
    assert not (tmp_path / "embedding.lock").exists()


def test_request_totals_survive_cache_only_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(
        compare,
        "embed_batch",
        lambda cfg, texts: (
            np.ones((len(texts), 2), dtype="<f4"),
            {"prompt_tokens": 9},
        ),
    )
    jobs = [{"key": "one", "text": "manual"}]
    compare.cached_embeddings(jobs, config(), tmp_path, 1)
    compare.cached_embeddings(jobs, config(), tmp_path, 1)
    assert compare.recorded_requests(tmp_path) == {
        "successful_requests": 1,
        "inputs": 1,
        "provider_usage": {"prompt_tokens": 9},
    }


def example_data():
    from lightrag.document_structure.children import heading_conditions
    from lightrag.document_structure.retrieval import ParentChildIndex

    parents, children, rows = [], [], []
    for key, model, letter, text in [
        ("right", "L3266", "A", "Supporting quote and context"),
        ("wrong", "L4263", "B", "Irrelevant evidence"),
    ]:
        parent = {
            "parent_id": key,
            "doc_id": "doc-" + key,
            "document_name": model + ".pdf",
            "heading_path": ["Guide"],
            "content": text,
            "content_role": "body",
            "scope": {"status": "filename", "applicable_models": [model]},
            "feature_domain": "unknown",
            "context_parent_ids": [],
            "review_flags": [],
        }
        parent["conditions"] = heading_conditions(parent)
        content = "Supporting" if key == "right" else "Irrelevant"
        child = {
            "child_id": "child-" + key,
            "parent_id": key,
            "content": content,
            "parent_span": {"start": 0, "end": len(content)},
            "review_flags": [],
            **{
                k: parent[k]
                for k in (
                    "doc_id",
                    "document_name",
                    "heading_path",
                    "scope",
                    "conditions",
                    "feature_domain",
                    "context_parent_ids",
                )
            },
        }
        parents.append(parent)
        children.append(child)
        rows.extend(
            [
                {
                    "id": "legacy-" + key,
                    "arm": "legacy",
                    "text": text,
                    "content": text,
                    "document": letter,
                    "models": [model],
                    "parent_id": None,
                },
                {
                    "id": child["child_id"],
                    "arm": "child",
                    "text": "Indexed " + text,
                    "content": content,
                    "document": letter,
                    "models": [model],
                    "parent_id": key,
                },
            ]
        )
    questions = [
        {
            "id": "Q1",
            "query": "L3266 question",
            "test_kind": "retrieval",
            "evidence": [{"document": "A", "quote": "Supporting quote"}],
        },
        {
            "id": "Q2",
            "query": "Boundary query",
            "test_kind": "boundary",
            "evidence": [],
        },
    ]
    vectors = {}
    for text in [r["text"] for r in rows] + [q["query"] for q in questions]:
        key = compare.digest(
            {"provider": compare.provider_identity(config()), "text": text}
        )
        vectors[key] = np.array(
            [0.8, 0.2] if "Supporting" in text else [1.0, 0], dtype=np.float32
        )
    return rows, questions, ParentChildIndex(parents, children), vectors


def test_dense_comparison_filters_models_and_scores_parent_expansion(tmp_path):
    rows, questions, index, vectors = example_data()
    results, summary = compare.evaluate(
        rows, questions, index, vectors, config(), tokenizer(), 1, 1000
    )
    assert len(results) == 8
    assert all(s["queries"] == 1 for s in summary.values())
    assert summary["legacy_raw"]["mean_anchor_coverage"] == 0
    assert summary["legacy_model_filter"]["mean_anchor_coverage"] == 1
    assert summary["child_model_filter"]["mean_anchor_coverage"] == 1
    assert summary["child_model_filter"]["mean_hit_text_anchor_coverage"] == 0
    report = {
        "provider": compare.provider_identity(config()),
        "top_k": 1,
        "context_budget": 1000,
        "retrieval_metrics": summary,
        "recorded_embedding_totals": {
            "successful_requests": 0,
            "inputs": 0,
            "provider_usage": {},
        },
    }
    compare.write_report(tmp_path, report, results)
    assert "Q1" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_context_budget_does_not_truncate_evidence_groups():
    rows, questions, index, vectors = example_data()
    results, summary = compare.evaluate(
        rows, questions, index, vectors, config(), tokenizer(), 1, 1
    )
    assert summary["child_model_filter"]["mean_anchor_coverage"] == 0
    selected = next(
        r for r in results if r["id"] == "Q1" and r["arm"] == "child_model_filter"
    )
    assert selected["evidence"] == []
    assert not selected["groups"][0]["included"]


@pytest.mark.parametrize("embedding", [[1], [0, 0], [float("nan"), 1]])
def test_invalid_provider_vectors_are_rejected(monkeypatch, embedding):
    payload = {"data": [{"index": 0, "embedding": embedding}]}
    monkeypatch.setattr(
        compare.request,
        "build_opener",
        lambda *args: SimpleNamespace(
            open=lambda *args, **kwargs: io.StringIO(json.dumps(payload))
        ),
    )
    with pytest.raises(ValueError, match="shape/values"):
        compare.embed_batch(config(), ["input"])


def test_dimension_parameter_can_be_explicitly_enabled(monkeypatch):
    captured = []

    def response(req, timeout):
        captured.append(json.loads(req.data))
        return io.StringIO(json.dumps({"data": [{"index": 0, "embedding": [1, 0]}]}))

    monkeypatch.setattr(
        compare.request, "build_opener", lambda *args: SimpleNamespace(open=response)
    )
    compare.embed_batch({**config(), "send_dimensions": True}, ["input"])
    assert captured[0]["dimensions"] == 2


def test_missing_key_blocks_real_run_before_any_io(monkeypatch, tmp_path):
    monkeypatch.setattr(
        compare, "embedding_config", lambda path: {**config(), "api_key": ""}
    )
    monkeypatch.setattr(
        compare,
        "load_data",
        lambda *args: pytest.fail("Should fail before loading experiment inputs"),
    )
    with pytest.raises(ValueError, match="API key"):
        compare.main(
            [
                "--legacy-chunks",
                "unused",
                "--bundle",
                "unused",
                "--questions",
                "unused",
                "--tokenizer-file",
                "unused",
                "--output",
                str(tmp_path / "out"),
                "--allow-remote-embedding",
            ]
        )
    assert not (tmp_path / "out").exists()


def test_embedding_config_preflight_accepts_no_key_and_defaults_to_no_dimensions(
    monkeypatch, tmp_path
):
    envfile = tmp_path / "experiment.env"
    envfile.write_text(
        "EMBEDDING_BINDING=openai\nEMBEDDING_MODEL=test\nEMBEDDING_DIM=2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(compare.os, "environ", {})
    actual = compare.embedding_config(envfile)
    assert actual["api_key"] == ""
    assert not actual["send_dimensions"]
    assert actual["dimension"] == 2


@pytest.mark.parametrize(
    "damage", ["duplicate", "no_query", "no_evidence", "invalid_kind", "empty_quote"]
)
def test_invalid_question_sets_fail_before_paid_work(damage):
    case = {
        "id": "Q1",
        "query": "Question",
        "test_kind": "retrieval",
        "evidence": [{"document": "A", "quote": "Manual quote"}],
    }
    cases = [case]
    if damage == "duplicate":
        cases.append(case.copy())
    elif damage == "no_query":
        case["query"] = " "
    elif damage == "no_evidence":
        case["evidence"] = []
    elif damage == "invalid_kind":
        case["test_kind"] = "unknown"
    else:
        case["evidence"][0]["quote"] = "..."
    with pytest.raises(ValueError):
        compare.validate_questions(cases)


def test_cross_experiment_reuse_only_requests_changed_text(monkeypatch, tmp_path):
    source, output = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    output.mkdir()
    old_rows = [{"arm": "child", "text": "unchanged"}]
    plan, jobs = compare.make_plan(old_rows, [], config(), {}, tokenizer())
    (source / "experiment.json").write_text(json.dumps(plan), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        compare,
        "embed_batch",
        lambda cfg, texts: (
            calls.append(texts) or np.ones((len(texts), 2), dtype="<f4"),
            {},
        ),
    )
    compare.cached_embeddings(jobs, config(), source, 16)
    _, new_jobs = compare.make_plan(
        old_rows + [{"arm": "child", "text": "changed"}], [], config(), {}, tokenizer()
    )
    calls.clear()
    _, stats = compare.cached_embeddings(new_jobs, config(), output, 16, source)
    assert calls == [["changed"]]
    assert stats["imported_inputs_this_run"] == 1
    assert stats["new_inputs"] == 1
    assert compare.recorded_requests(output)["inputs"] == 1
    assert (output / "embedding_reuse.json").exists()
    compare.cached_embeddings(new_jobs, config(), output, 16, source)
    assert calls == [["changed"]]


def test_reuse_refuses_different_provider_and_unplanned_inputs(tmp_path):
    plan, _jobs = compare.make_plan(
        [{"arm": "child", "text": "manual"}], [], config(), {}, tokenizer()
    )
    (tmp_path / "experiment.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration mismatch"):
        compare.reusable_embeddings(tmp_path, {**config(), "model": "different"})
    (tmp_path / "embedding_cache.jsonl").write_text(
        json.dumps(
            {
                "key": "outside-plan",
                "vector_f32": base64.b64encode(
                    np.ones(2, dtype="<f4").tobytes()
                ).decode("ascii"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside"):
        compare.reusable_embeddings(tmp_path, config())


def test_conflicting_duplicate_cache_rows_fail(tmp_path):
    path = tmp_path / "embedding_cache.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "key": "same",
                    "vector_f32": base64.b64encode(
                        np.array(v, dtype="<f4").tobytes()
                    ).decode("ascii"),
                }
            )
            + "\n"
            for v in ([1, 0], [0, 1])
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Conflicting"):
        compare.read_embedding_cache(path, config())


def test_capability_gate_uses_source_declaration_not_oracle_labels():
    from lightrag.document_structure.children import build_children
    from lightrag.document_structure.retrieval import ParentChildIndex
    from tests.document_structure.test_children import parent

    tok = SimpleNamespace(model_name="characters", encode=lambda text: list(text))
    parents, children = build_children(
        [
            parent(
                "intro", "## Guide\n以下功能仅适用于 L5298。\nO ADF（自动文稿送纸器）"
            ),
            parent("load", "ADF loading instructions"),
        ],
        tok,
    )
    index = ParentChildIndex(parents, children)
    rows = [
        {
            "id": c["child_id"],
            "arm": "child",
            "text": c["retrieval_text"],
            "content": c["content"],
            "models": c["scope"]["applicable_models"],
            "parent_id": c["parent_id"],
            "document": "B",
        }
        for c in children
    ]
    questions = [
        {
            "id": "Q",
            "query": "L3266 如何使用 ADF？",
            "test_kind": "retrieval",
            "evidence": [{"document": "B", "quote": "以下功能仅适用于 L5298。"}],
        }
    ]
    vectors = {
        compare.digest(
            {"provider": compare.provider_identity(config()), "text": text}
        ): np.array([1.0, 0.0])
        for text in [r["text"] for r in rows] + [questions[0]["query"]]
    }
    results, _ = compare.evaluate(
        rows, questions, index, vectors, config(), tok, 5, 1000
    )
    result = next(r for r in results if r["arm"] == "child_model_filter")
    assert result["capability_checks"][0]["status"] == "unsupported"
    assert all(
        r["reason"] == "feature_capability_not_supported"
        for r in result["candidate_rejections"]
    )
    assert result["metrics"]["coverage"] == 1
    assert all(e["kind"] == "capability_declaration" for e in result["evidence"])
    assert "loading instructions" not in "".join(
        e["content"] for e in result["evidence"]
    )
