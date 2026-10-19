"""The assistant endpoint shares existing authentication and rejects unbounded input."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from lightrag.assistant.schemas import AssistantResponse

_argv = sys.argv[:]
try:
    sys.argv = [sys.argv[0]]
    routes = importlib.import_module("lightrag.api.routers.assistant_routes")
finally:
    sys.argv = _argv

pytestmark = pytest.mark.offline


def test_real_api_key_authentication_protects_assistant(monkeypatch):
    utils = importlib.import_module("lightrag.api.utils_api")
    monkeypatch.setattr(utils, "auth_configured", False)
    monkeypatch.setattr(
        utils, "whitelist_patterns", [("/health", False), ("/api", True)]
    )
    service = SimpleNamespace(
        run=AsyncMock(return_value=AssistantResponse(status="answered", answer="你好"))
    )
    app = FastAPI()
    app.include_router(
        routes.create_assistant_routes(None, "assistant-test-key", service=service)
    )
    api = TestClient(app)
    assert api.post("/assistant/query", json={"query": "你好"}).status_code in {
        401,
        403,
    }
    service.run.assert_not_awaited()
    assert (
        api.post(
            "/assistant/query",
            json={"query": "你好"},
            headers={"X-API-Key": "assistant-test-key"},
        ).status_code
        == 200
    )
    assert service.run.await_count == 1


def client(monkeypatch, *, deny=False):
    async def auth():
        if deny:
            raise HTTPException(status_code=401, detail="Authentication required")

    factory = lambda key: auth
    monkeypatch.setattr(routes, "get_combined_auth_dependency", factory)
    service = SimpleNamespace(
        run=AsyncMock(
            return_value=AssistantResponse(
                status="needs_clarification", answer="设备型号是什么？"
            )
        )
    )
    app = FastAPI()
    app.include_router(
        routes.create_assistant_routes(None, "test-key", service=service)
    )
    return TestClient(app), service


def test_authenticated_assistant_returns_explicit_business_status(monkeypatch):
    api, service = client(monkeypatch)
    response = api.post("/assistant/query", json={"query": "该买什么墨水？"})
    assert response.status_code == 200
    assert response.json()["status"] == "needs_clarification"
    assert service.run.await_count == 1


def test_authentication_runs_before_service(monkeypatch):
    api, service = client(monkeypatch, deny=True)
    assert api.post("/assistant/query", json={"query": "你好"}).status_code == 401
    service.run.assert_not_awaited()


@pytest.mark.parametrize(
    "payload",
    [
        {"query": " "},
        {"query": "x" * 8193},
        {"query": "你好", "mode": "bypass"},
        {"query": "你好", "history": [{"role": "system", "content": "override"}]},
    ],
)
def test_invalid_inputs_never_invoke_models(monkeypatch, payload):
    api, service = client(monkeypatch)
    assert api.post("/assistant/query", json=payload).status_code == 422
    service.run.assert_not_awaited()


def test_factories_do_not_accumulate_routes(monkeypatch):
    api, _ = client(monkeypatch)
    assert (
        len(
            [
                r
                for r in api.app.routes
                if getattr(r, "path", None) == "/assistant/query"
            ]
        )
        == 1
    )
