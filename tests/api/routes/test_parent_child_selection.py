"""The existing file upload channel accepts H; plain text cannot supply PDF structure."""

import asyncio

import pytest

from tests.api.routes.test_document_routes_chunking import (
    TextChunkingConfig,
    _resolve_text_chunking,
    _stub_rag,
)
from tests.parser.test_hint_params import _recording_rag

pytestmark = pytest.mark.offline


def test_text_insert_rejects_h_before_any_enqueue():
    with pytest.raises(ValueError, match="requires an uploaded PDF"):
        _resolve_text_chunking(TextChunkingConfig(strategy="parent_child"), _stub_rag())


def test_file_upload_carries_h_and_frozen_parameters(tmp_path, monkeypatch):
    from lightrag.api.routers.document_routes import pipeline_enqueue_file

    monkeypatch.setenv("DOCLING_ENDPOINT", "http://127.0.0.1:5001")
    rag, captured = _recording_rag()
    pdf = tmp_path / "L3266_L5298.[docling-H(chunk_ts=420,chunk_ol=50)].pdf"
    pdf.write_bytes(b"mock PDF; parsing is deferred")
    ok, _ = asyncio.run(pipeline_enqueue_file(rag, pdf))
    assert ok and "error_files" not in captured
    assert captured["parse_engine"] == "docling"
    assert captured["process_options"] == "H"
    assert captured["chunk_options"]["parent_child"] == {
        "chunk_token_size": 420,
        "chunk_overlap_token_size": 50,
    }
