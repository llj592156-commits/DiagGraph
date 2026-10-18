"""Paratera GLM-Rerank uses its documented endpoint and standard result indices."""

from unittest.mock import AsyncMock

import pytest

from lightrag.rerank import cohere_rerank

pytestmark = pytest.mark.offline


@pytest.mark.asyncio
async def test_paratera_glm_request_and_index_score_response(monkeypatch):
    from lightrag import rerank

    response = AsyncMock()
    response.status = 200
    response.json.return_value = {
        "results": [
            {"index": 1, "relevance_score": 0.92, "document": {"text": "source two"}},
            {"index": 0, "relevance_score": 0.11},
        ]
    }
    session = AsyncMock()
    session.__aenter__.return_value = session
    response_context = AsyncMock()
    response_context.__aenter__.return_value = response
    from unittest.mock import Mock

    session.post = Mock(return_value=response_context)
    monkeypatch.setattr(rerank.aiohttp, "ClientSession", lambda: session)
    url = "https://llmapi.paratera.com/v1/p002/rerank"
    result = await cohere_rerank(
        "L4263 ink",
        ["source one", "source two"],
        top_n=5,
        model="GLM-Rerank",
        base_url=url,
        api_key="test-only",
    )
    assert result == [
        {"index": 1, "relevance_score": 0.92},
        {"index": 0, "relevance_score": 0.11},
    ]
    assert session.post.call_args.args == (url,)
    payload = session.post.call_args.kwargs["json"]
    assert payload == {
        "model": "GLM-Rerank",
        "query": "L4263 ink",
        "documents": ["source one", "source two"],
        "top_n": 5,
    }
