"""Authenticated, nonstreaming assistant endpoint alongside the existing query API."""

from fastapi import APIRouter, Depends

from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.assistant import AssistantRequest, AssistantResponse, AssistantService


def create_assistant_routes(rag, api_key: str | None = None, *, service=None):
    router = APIRouter(tags=["assistant"])
    assistant = service if service is not None else AssistantService(rag)
    auth = get_combined_auth_dependency(api_key)

    @router.post(
        "/assistant/query",
        response_model=AssistantResponse,
        dependencies=[Depends(auth)],
        description="LLM routing, bounded mix retrieval and evidence-checked answers. Inspect status: HTTP 200 alone does not mean an answer was supported. Stateless conversation: send user/assistant history on each request.",
    )
    async def assistant_query(request: AssistantRequest) -> AssistantResponse:
        return await assistant.run(request)

    return router
