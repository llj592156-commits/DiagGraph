"""Extract explicit exclusive feature lists without rewriting parent scope.

Only an explicit declaration plus a contiguous bullet list is recognized.
Unknown wording remains unknown; absence of a feature is never non-support.
See ``docs/ParentChildEvidenceRepair-zh.md`` for supported syntax and limits.
"""

from __future__ import annotations

import re
from copy import deepcopy

from .parents import MODEL_RE, stable_id

DECLARATION_RE = re.compile(
    r"以下功能(?:仅适用于?|只适用于?)\s*"
    r"(?P<models>[A-Za-z0-9 /、,，和及]+)[。.]\s*$"
)
BULLET_RE = re.compile(r"^\s*(?:[Oo•●·]|[-*])\s+(.+?)\s*$")
FEATURE_LABELS = {
    "fax": {"传真", "传真功能"},
    "adf": {"ADF", "ADF（自动文稿送纸器）", "ADF(自动文稿送纸器)"},
    "ethernet": {"以太网", "以太网连接", "以太网接口"},
}
FEATURE_PATTERNS = {
    "fax": re.compile(r"传真"),
    "adf": re.compile(r"(?<![A-Za-z])ADF(?![A-Za-z])|自动文稿送纸器", re.IGNORECASE),
    "ethernet": re.compile(r"以太网|(?<![A-Za-z])Ethernet(?![A-Za-z])", re.IGNORECASE),
}


def query_features(query: str) -> list[str]:
    """Recognize literal feature names, not inferred intents or capabilities."""
    return sorted(
        key for key, pattern in FEATURE_PATTERNS.items() if pattern.search(query)
    )


def extract_capabilities(parents: list[dict]) -> list[dict]:
    from .children import source_refs

    records = []
    for parent in parents:
        if parent["content_role"] != "body":
            continue
        family = sorted(
            set(MODEL_RE.findall(parent["document_name"].upper().replace("_", " ")))
        )
        text = parent["content"]
        lines = list(re.finditer(r"[^\n]+", text))
        for position, line in enumerate(lines):
            declaration = DECLARATION_RE.search(line[0])
            if not declaration:
                continue
            allowed = sorted(set(MODEL_RE.findall(declaration["models"].upper())))
            if not allowed or not family or not set(allowed).issubset(family):
                continue
            start = line.start() + declaration.start()
            for bullet in lines[position + 1 :]:
                match = BULLET_RE.fullmatch(bullet[0])
                if not match:
                    break
                features = [
                    key for key, labels in FEATURE_LABELS.items() if match[1] in labels
                ]
                if not features:
                    continue
                end = bullet.end()
                refs, positions = source_refs(parent, start, end)
                for feature in features:
                    record = {
                        "feature": feature,
                        "family_models": family,
                        "supported_models": allowed,
                        "doc_id": parent["doc_id"],
                        "document_name": parent["document_name"],
                        "parent_id": parent["parent_id"],
                        "parent_span": {"start": start, "end": end},
                        "content": text[start:end],
                        "source_spans": refs,
                        "source_positions": positions,
                        "rule": "explicit_exclusive_feature_list_v1",
                    }
                    record["id"] = stable_id("capability-", record)
                    records.append(record)
    return records


class CapabilityIndex:
    """Check a literal feature against explicit declarations in its document family."""

    def __init__(self, parents: list[dict]):
        self.records = extract_capabilities(parents)

    def check(self, model: str, feature: str) -> dict:
        if feature not in FEATURE_LABELS or not model.strip():
            raise ValueError("A known feature and device model are required")
        model = model.strip().upper()
        records = [
            record
            for record in self.records
            if record["feature"] == feature and model in record["family_models"]
        ]
        claims = {model in record["supported_models"] for record in records}
        status = (
            "unknown"
            if not claims
            else "conflict"
            if len(claims) > 1
            else "supported"
            if True in claims
            else "unsupported"
        )
        return {
            "model": model,
            "feature": feature,
            "status": status,
            "evidence": deepcopy(records),
            "requires_review": status in {"unknown", "conflict"},
        }

    def check_query(self, query: str) -> list[dict]:
        models = sorted(set(MODEL_RE.findall(query.upper().replace("_", " "))))
        return [
            self.check(model, feature)
            for model in models
            for feature in query_features(query)
        ]
