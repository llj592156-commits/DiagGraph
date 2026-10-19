"""Resolve explicit conversation controls without treating old replies as evidence."""

from __future__ import annotations

import re

from .schemas import AssistantRequest


def is_regeneration_request(text: str) -> bool:
    """Match standalone retry commands only; requests adding business details stay intact."""
    normalized = re.sub(r"[\s。.!！?？,，;；:：]+", "", text.casefold())
    return normalized in {
        "regenerate",
        "pleaseregenerate",
        "regeneratetheanswer",
        "tryagain",
        "pleasetryagain",
        "answeragain",
        "pleaseansweragain",
    } or bool(
        re.fullmatch(
            r"(?:(?:请|麻烦)(?:你)?(?:帮我)?|帮我)?"
            r"(?:(?:再|重新)(?:生成|回答|回复|答复|说)(?:一遍|一次|一下)?"
            r"|再来(?:一遍|一次)|换(?:个|一种)说法)"
            r"(?:吧|吗|好吗)?",
            normalized,
        )
    )


def conversation_context(request: AssistantRequest) -> dict:
    """Return routing hints and source indexes, never trusted facts or cached evidence."""
    if not is_regeneration_request(request.query):
        return {"action": "answer"}
    for index in range(len(request.history) - 2, -1, -1):
        user, reply = request.history[index : index + 2]
        if (
            user.role != "user"
            or reply.role != "assistant"
            or is_regeneration_request(user.content)
        ):
            continue
        return {
            "action": "regenerate",
            "target_user_message_index": index,
            "target_assistant_message_index": index + 1,
            # A marker is only a reason to recheck sources, never proof of support.
            "requires_evidence_check": bool(re.search(r"\[\d+\]", reply.content)),
        }
    return {
        "action": "regenerate",
        "target_user_message_index": None,
        "target_assistant_message_index": None,
        "requires_evidence_check": False,
    }
