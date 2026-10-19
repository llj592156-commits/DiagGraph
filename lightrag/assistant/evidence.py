"""Model-facing evidence without indexing coordinates; full attribution stays intact."""

from .schemas import EvidenceItem


def model_evidence(item: EvidenceItem) -> dict:
    """Keep exact source text and separate applicability scopes in every model stage."""
    result = item.model_dump()
    for source in result["sources"]:
        pc = source.get("parent_child")
        if pc:
            # These are indexing locations/counters, never instructions or limits.
            for key in (
                "source_positions",
                "source_spans",
                "parent_span",
                "tokens",
                "retrieval_tokens",
                "tokenizer",
            ):
                pc.pop(key, None)
    return result
