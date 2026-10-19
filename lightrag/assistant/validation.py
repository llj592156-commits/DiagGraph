"""Conservative guards for known diagnosis gap errors, not a semantic classifier."""

import re

from .schemas import RouteDecision


def asks_user_observation(text: str) -> bool:
    """Recognize explicit user-addressed questions that cannot be answered by manuals."""
    return bool(
        re.search(
            r"(?:您|你).{0,24}(?:是否|有没有|已经|曾经|试过|多少|几次|什么型号|哪种|什么方向)"
            r"|条纹.{0,8}(?:是|为).{0,30}(?:水平|横向).{0,12}还是.{0,12}(?:垂直|纵向)"
            r"|\b(?:have|did)\s+you\b|\bare (?:the|your) (?:stripes|lines)\b",
            text,
            re.IGNORECASE,
        )
    )


def validate_clarification(question: str | None, route: RouteDecision) -> None:
    """Reject blank questions and clear requests to repeat an explicit action count."""
    if question is None:
        return
    if not question.strip():
        raise ValueError("Use null when no clarification is necessary")
    if not re.search(
        r"几次|多少次|几遍|多少遍|次数.{0,8}(?:多少|几)|how many times|number of times",
        question,
        re.IGNORECASE,
    ):
        return
    for fact in route.extracted_context:
        if fact.field != "attempted_action":
            continue
        # Use only source-validated user facts, never assistant history or documents.
        text = fact.value + " " + fact.quote
        if not re.search(
            r"[一二三四五六七八九十两零百\d]+\s*(?:次|遍)"
            r"|\b(?:once|twice|\d+\s+times)\b",
            text,
            re.IGNORECASE,
        ):
            continue
        actions = re.findall(
            r"清洗|清洁|重启|重置|校准|更换|clean|restart|reset|calibrat|replac",
            text,
            re.IGNORECASE,
        )
        if any(action.casefold() in question.casefold() for action in actions):
            raise ValueError(
                "The user's attempted_action already states this action count. "
                "Ask only for a genuinely unknown detail, or use null."
            )
