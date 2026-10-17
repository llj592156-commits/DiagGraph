"""Audit supplemental evidence and feature gates locally, without embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lightrag.document_structure.capabilities import query_features
from lightrag.document_structure.retrieval import ParentChildIndex
from lightrag.document_structure.table_notes import evidence_text
from lightrag.evaluation.compare_parent_child_retrieval import normalized


def audit(indexes: dict[str, ParentChildIndex], questions: list[dict]) -> dict:
    parents = [
        parent for index in indexes.values() for parent in index.parents.values()
    ]
    children = [
        child for index in indexes.values() for child in index.children.values()
    ]
    combined = ParentChildIndex(parents, children)
    inventory, gates = [], []
    for question in questions:
        anchors = []
        for anchor in question.get("evidence", []):
            candidates = indexes[anchor["document"]].parents.values()
            needle = normalized(anchor["quote"])
            anchors.append(
                {
                    "document": anchor["document"],
                    "quote": anchor["quote"],
                    "source_body_contains": any(
                        needle in normalized(p["content"]) for p in candidates
                    ),
                    "evidence_contains": any(
                        needle in normalized(evidence_text(p)) for p in candidates
                    ),
                }
            )
        inventory.append({"id": question["id"], "anchors": anchors})
        for check in combined.capabilities.check_query(question["query"]):
            candidate_ids = [
                c["child_id"]
                for c in children
                if check["model"] in c["scope"]["applicable_models"]
                and check["feature"] in query_features(c["content"])
            ]
            resolved = combined.resolve(
                candidate_ids,
                model=check["model"],
                required_features=[check["feature"]],
            )
            gates.append(
                {
                    "id": question["id"],
                    **check,
                    "candidate_children": len(candidate_ids),
                    "accepted_children": len(resolved["accepted_child_ids"]),
                    "explanation_retained": bool(resolved["capability_evidence"]),
                }
            )
    return {
        "mapping_only": True,
        "embedding_requests": 0,
        "generation_model_calls": 0,
        "table_notes": sum(len(p.get("table_notes", [])) for p in parents),
        "capability_declarations": combined.capabilities.records,
        "anchor_inventory": inventory,
        "feature_gate_checks": gates,
        "limitation": "Corpus presence and deterministic gates only; not retrieval accuracy or answer correctness.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Output already exists")
    indexes = {path.name: ParentChildIndex.from_directory(path) for path in args.bundle}
    if len(indexes) != len(args.bundle):
        raise ValueError("Duplicate bundle labels")
    questions = [
        json.loads(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = audit(indexes, questions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "table_notes": result["table_notes"],
                "capability_declarations": len(result["capability_declarations"]),
                "embedding_requests": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
