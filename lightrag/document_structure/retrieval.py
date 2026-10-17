"""Resolve selected child IDs to applicable parents without search or LLMs.

This is an offline mapping check, not a production retrieval backend or a
relevance score. Callers must provide the device model and already selected
child IDs. Unknown model scope is refused; retained local review flags travel
with the evidence. See ``docs/ParentChildOfflinePreparation-zh.md``.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from .capabilities import CapabilityIndex
from .children import VERSION, heading_conditions
from .parents import file_hash
from .table_notes import (
    evidence_text,
    notes_for_child,
    notes_for_parent,
    validate_notes,
    with_table_notes,
)
from .tables import validate_table_part


def normalize_platform(value: str | None) -> str | None:
    if value is None:
        return None
    key = value.casefold().replace(" ", "")
    aliases = {
        "mac": "macos",
        "macos": "macos",
        "windows": "windows",
        "ios": "ios",
        "android": "android",
    }
    if key not in aliases:
        raise ValueError("Unsupported platform filter")
    return aliases[key]


class ParentChildIndex:
    """Resolve immutable bundle IDs with model, platform and domain checks."""

    def __init__(self, parents: list[dict], children: list[dict]):
        self.parents = {p["parent_id"]: deepcopy(p) for p in parents}
        self.children = {c["child_id"]: deepcopy(c) for c in children}
        if len(self.parents) != len(parents) or len(self.children) != len(children):
            raise ValueError("Duplicate parent/child ID")
        for parent in self.parents.values():
            validate_notes(parent)
            parent.setdefault("conditions", heading_conditions(parent))
            for key in parent["context_parent_ids"]:
                ancestor = self.parents.get(key)
                if (
                    ancestor is None
                    or ancestor["doc_id"] != parent["doc_id"]
                    or ancestor["content_role"] != "body"
                ):
                    raise ValueError("Invalid context parent identity")
                path, prefix = parent["heading_path"], ancestor["heading_path"]
                if len(prefix) >= len(path) or path[: len(prefix)] != prefix:
                    raise ValueError("Context parent must be a strict heading ancestor")
        for child in self.children.values():
            parent = self.parents.get(child["parent_id"])
            if parent is None or parent["content_role"] != "body":
                raise ValueError("Orphan/non-body child")
            if child.get("table_notes", []) != notes_for_child(parent, child):
                raise ValueError("Child table note metadata mismatch")
            if with_table_notes(
                child.get("retrieval_text", ""), child.get("table_notes", [])
            ) != child.get("retrieval_text", ""):
                raise ValueError("Table note missing from child embedding text")
            for field in [
                "doc_id",
                "document_name",
                "scope",
                "conditions",
                "heading_path",
                "feature_domain",
                "context_parent_ids",
            ]:
                if child[field] != parent[field]:
                    raise ValueError(f"Child/parent metadata mismatch: {field}")
            span = child["parent_span"]
            if not 0 <= span["start"] < span["end"] <= len(parent["content"]):
                raise ValueError("Invalid child parent span")
            if child.get("transformation"):
                validate_table_part(
                    parent["content"][span["start"] : span["end"]], child
                )
            elif parent["content"][span["start"] : span["end"]] != child["content"]:
                raise ValueError("Invalid child parent span")
        self.capabilities = CapabilityIndex(list(self.parents.values()))

    @classmethod
    def from_directory(cls, directory: Path) -> ParentChildIndex:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary["schema_version"] not in {
            VERSION,
            "parent-child-draft-v1",
            "parent-child-draft-v2",
        }:
            raise ValueError("Unsupported child draft version")
        rows = {}
        for name in ("parents.jsonl", "children.jsonl"):
            if file_hash(directory / name) != summary["outputs"][name]:
                raise ValueError("Child bundle artifact hash mismatch")
            rows[name] = [
                json.loads(line)
                for line in (directory / name).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if (
            len(rows["parents.jsonl"]) != summary["parents"]
            or len(rows["children.jsonl"]) != summary["children"]
        ):
            raise ValueError("Child bundle count mismatch")
        if any(
            p["doc_id"] != summary["doc_id"]
            or p["document_name"] != summary["document_name"]
            for p in rows["parents.jsonl"]
        ):
            raise ValueError("Child bundle document identity mismatch")
        if summary["schema_version"] == VERSION:
            inputs = summary["inputs"]
            source = inputs.get("table_sidecar")
            if source:
                path = Path(source)
                source_hash = file_hash(path)
                if source_hash != inputs["table_sidecar_sha256"]:
                    raise ValueError("Table note sidecar hash mismatch")
                tables = json.loads(path.read_text(encoding="utf-8"))["tables"]
                for parent in rows["parents.jsonl"]:
                    if parent.get("table_notes", []) != notes_for_parent(
                        parent, tables, source_hash
                    ):
                        raise ValueError("Table notes differ from source sidecar")
            elif any(p.get("table_notes") for p in rows["parents.jsonl"]):
                raise ValueError("Table notes have no source sidecar")
        return cls(rows["parents.jsonl"], rows["children.jsonl"])

    def resolve(
        self,
        child_ids: list[str],
        *,
        model: str,
        platform: str | None = None,
        feature_domain: str | None = None,
        required_features: list[str] | None = None,
    ) -> dict:
        """Return deduplicated parents and required ancestors, with rejections.

        No token budgeting or ranking is performed. A production adapter must
        budget complete evidence groups and honor each returned review flag.
        """
        if not model.strip():
            raise ValueError("A device model is required")
        model, platform = model.strip().upper(), normalize_platform(platform)
        capability_checks = [
            self.capabilities.check(model, feature)
            for feature in dict.fromkeys(required_features or [])
        ]
        blocked = any(
            c["status"] in {"unsupported", "conflict"} for c in capability_checks
        )
        accepted, rejected, selected, context_ids = [], [], {}, set()

        def reason(parent: dict, *, context: bool = False) -> str | None:
            if blocked:
                return "feature_capability_not_supported"
            scope = parent["scope"]
            if (
                scope["status"] in {"unknown", "conflict"}
                or not scope["applicable_models"]
            ):
                return "model_scope_unresolved"
            if model not in scope["applicable_models"]:
                return "model_mismatch"
            platforms = parent["conditions"]["platforms"]
            if platform and platforms and platform not in platforms:
                return "platform_mismatch"
            if (
                feature_domain
                and parent["feature_domain"] != feature_domain
                and not (context and parent["feature_domain"] == "unknown")
            ):
                return "domain_mismatch"
            return None

        def ancestors(parent: dict) -> list[dict]:
            found = {}

            def visit(node: dict) -> None:
                for key in node["context_parent_ids"]:
                    if key not in found:
                        found[key] = self.parents[key]
                        visit(self.parents[key])

            visit(parent)
            return sorted(found.values(), key=lambda p: len(p["heading_path"]))

        for key in dict.fromkeys(child_ids):
            if key not in self.children:
                rejected.append({"child_id": key, "reason": "child_not_found"})
                continue
            child = self.children[key]
            parent = self.parents[child["parent_id"]]
            rejection = reason(parent)
            context = ancestors(parent)
            if rejection is None:
                for ancestor in context:
                    mismatch = reason(ancestor, context=True)
                    if mismatch:
                        rejection = "required_context_" + mismatch
                        break
            if rejection:
                rejected.append({"child_id": key, "reason": rejection})
                continue
            accepted.append(key)
            for ancestor in context:
                selected[ancestor["parent_id"]] = ancestor
                context_ids.add(ancestor["parent_id"])
            selected[parent["parent_id"]] = parent
        direct_ids = {self.children[key]["parent_id"] for key in accepted}
        return {
            "accepted_child_ids": accepted,
            "rejected": rejected,
            "parents": [
                {**deepcopy(p), "evidence_content": evidence_text(p)}
                for p in selected.values()
            ],
            "context_only_parent_ids": sorted(context_ids - direct_ids),
            "capability_checks": capability_checks,
            "capability_evidence": list(
                {
                    record["id"]: record
                    for check in capability_checks
                    for record in check["evidence"]
                }.values()
            ),
            "requires_review": any(c["requires_review"] for c in capability_checks)
            or any(p["review_flags"] for p in selected.values())
            or any(self.children[key]["review_flags"] for key in accepted),
            "review_flags": sorted(
                {flag for p in selected.values() for flag in p["review_flags"]}
                | {
                    flag
                    for key in accepted
                    for flag in self.children[key]["review_flags"]
                }
            ),
            "mapping_only": True,
        }
