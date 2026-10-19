"""Audit frozen source, parent/child, persisted and retrieved evidence offline.

Inputs are read directly and never initialized as storage backends. Page checks
establish consistency with parser positions, not independent PDF/OCR accuracy.
The output directory must be new; failures produce a report and exit code 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from lightrag.chunker.parent_child import child_evidence_content
from lightrag.document_structure.children import (
    ChildConfig,
    load_local_tokenizer,
    validate_children,
)
from lightrag.document_structure.parents import read_sidecar, validate_coverage
from lightrag.document_structure.retrieval import ParentChildIndex
from lightrag.document_structure.table_notes import evidence_text


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return "sha256:" + result.hexdigest()


class Audit:
    """Collect check-level results and detect inputs changed during the run."""

    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.inputs: dict[str, str] = {}

    def track(self, path: Path, expected: str | None = None) -> Path:
        path = path.resolve()
        observed = digest(path)
        require(
            expected is None or observed == expected, f"Input hash mismatch: {path}"
        )
        previous = self.inputs.setdefault(str(path), observed)
        require(previous == observed, f"Input changed while reading: {path}")
        return path

    def json(self, path: Path) -> dict:
        return json.loads(self.track(path).read_text(encoding="utf-8"))

    def rows(self, path: Path) -> list[dict]:
        return [
            json.loads(line)
            for line in self.track(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def check(self, category: str, identity: str, operation) -> None:
        item = {"category": category, "id": identity, "passed": True}
        try:
            detail = operation()
            if detail is not None:
                item["detail"] = detail
        except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
            item.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        self.checks.append(item)

    def finish(self) -> dict:
        for path, expected in list(self.inputs.items()):
            self.check(
                "input_unchanged",
                path,
                lambda p=path, h=expected: require(
                    digest(Path(p)) == h, "Input changed"
                ),
            )
        counts = {}
        for category in sorted({c["category"] for c in self.checks}):
            rows = [c for c in self.checks if c["category"] == category]
            counts[category] = {
                "checks": len(rows),
                "passed": sum(c["passed"] for c in rows),
                "failed": sum(not c["passed"] for c in rows),
            }
        return {
            "schema_version": "parent-child-integrity-audit-v1",
            "passed": all(c["passed"] for c in self.checks),
            "counts": counts,
            "input_hashes": self.inputs,
            "checks": self.checks,
            "limitations": [
                "Pages are checked against parsed source positions, not PDF visual truth.",
                "Structural coverage does not establish retrieval relevance or answer accuracy.",
                "Saved retrievals are replayed offline; no model or service is called.",
            ],
        }


def check_pages(row: dict, page_count: int) -> None:
    positions = [p for s in row["source_positions"] for p in s["positions"]]
    require(bool(positions), "No source positions")
    require(
        all(str(p.get("anchor", "")).isdigit() for p in positions),
        "Missing/non-numeric page anchor",
    )
    pages = sorted({int(p["anchor"]) for p in positions})
    require(all(1 <= p <= page_count for p in pages), "Page outside document bounds")
    require(row["pages"] == pages, "Pages disagree with source positions")


def check_parent_positions(parent: dict, blocks: dict) -> None:
    spans = parent["source_spans"]
    refs = parent["source_positions"]
    require(len(spans) == len(refs), "Source position/span count mismatch")
    for span, ref in zip(spans, refs, strict=True):
        require(
            all(span[k] == ref[k] for k in ("block_id", "fragment_id")),
            "Source position/span identity mismatch",
        )
        original = blocks[ref["block_id"]]["positions"]
        require(
            all(p in original for p in ref["positions"]),
            "Position absent from original source block",
        )


def check_position_coverage(parents: list[dict], blocks: list[dict]) -> None:
    observed: dict[str, list] = {}
    for parent in parents:
        for ref in parent["source_positions"]:
            observed.setdefault(ref["block_id"], []).extend(ref["positions"])
    for block in blocks:
        require(
            all(p in observed.get(block["blockid"], []) for p in block["positions"]),
            f"Dropped source position: {block['blockid']}",
        )


def check_stored_parent(parent: dict, stored: dict, children: list[dict]) -> None:
    require(all(stored[k] == v for k, v in parent.items()), "Stored parent differs")
    require(stored["source_doc_id"] == parent["doc_id"], "Stored parent owner differs")
    expected = [f"{c['doc_id']}-pc-{c['child_id']}" for c in children]
    require(stored["indexed_chunk_ids"] == expected, "Stored parent child links differ")


def check_stored_child(child: dict, stored: dict, key: str) -> None:
    # The namespace key includes doc_id; the row retains the local chunk ID.
    require(
        key == f"{child['doc_id']}-pc-{child['child_id']}",
        "Stored namespace key differs",
    )
    require(stored["chunk_id"] == "pc-" + child["child_id"], "Stored chunk ID differs")
    require(stored["full_doc_id"] == child["doc_id"], "Stored child owner differs")
    require(
        stored["file_path"] == child["document_name"], "Stored source filename differs"
    )
    require(
        stored["content"] == child_evidence_content(child), "Stored child text differs"
    )
    meta = stored["parent_child"]
    require(meta["source_doc_id"] == child["doc_id"], "Stored metadata owner differs")
    for field, value in child.items():
        if field not in {"content", "retrieval_text"}:
            require(meta[field] == value, f"Stored child metadata differs: {field}")


def check_evidence(chunk: dict, mappings: list[dict], docs: dict, rows: dict) -> None:
    key = chunk["chunk_id"]
    selected = [m for m in mappings if m["source_chunk_id"] == key]
    require(bool(selected), "Evidence has no source metadata")
    first = rows[key]
    doc_id = first["full_doc_id"]
    meta = first["parent_child"]
    parents = docs[doc_id]["parent_child"]["parents"]
    ids = list(dict.fromkeys([meta["parent_id"], *meta["context_parent_ids"]]))
    require([m["parent_id"] for m in selected] == ids, "Missing/reordered prerequisite")
    support = selected[0]["source_chunk_ids"]
    require(bool(support) and support[0] == key, "Invalid primary supporting child")
    require(len(support) == len(set(support)), "Duplicate supporting child")
    level = selected[0]["evidence_level"]
    require(
        level in {"full_parent", "child_with_prerequisites"}, "Unknown evidence level"
    )
    for mapping in selected:
        require(mapping["doc_id"] == doc_id, "Evidence source document differs")
        require(
            mapping["pages"] == parents[mapping["parent_id"]]["pages"],
            "Evidence pages differ",
        )
        require(
            mapping["source_chunk_ids"] == support, "Inconsistent supporting children"
        )
        require(mapping["evidence_level"] == level, "Inconsistent evidence level")
    prerequisites = [evidence_text(parents[p]) for p in ids if p != meta["parent_id"]]
    child_text = []
    for child_id in support:
        row = rows[child_id]
        require(row["full_doc_id"] == doc_id, "Cross-document supporting child")
        child_meta = row["parent_child"]
        require(
            child_meta["parent_id"] == meta["parent_id"],
            "Cross-parent supporting child",
        )
        require(
            child_meta["context_parent_ids"] == meta["context_parent_ids"],
            "Prerequisites differ",
        )
        child_text.append(row["content"])
    expected = (
        [evidence_text(parents[p]) for p in ids]
        if level == "full_parent"
        else [*child_text, *prerequisites]
    )
    require(
        chunk["content"] == "\n\n".join(expected),
        "Evidence body is truncated or altered",
    )
    require(
        chunk["file_path"] == first["file_path"], "Evidence source filename differs"
    )


def audit_bundle(audit: Audit, directory: Path, tokenizer) -> tuple[list, list]:
    summary = audit.json(directory / "summary.json")
    for name, expected in summary["outputs"].items():
        audit.track(directory / name, expected)
    parent_dir = Path(summary["inputs"]["parent_dir"])
    audit.track(parent_dir / "parents.jsonl", summary["inputs"]["parents_sha256"])
    audit.track(parent_dir / "summary.json", summary["inputs"]["parent_summary_sha256"])
    parent_summary = audit.json(parent_dir / "summary.json")
    inputs = parent_summary["inputs"]
    audit.track(Path(inputs["pdf"]), inputs["pdf_sha256"])
    source = audit.track(Path(inputs["sidecar"]), inputs["sidecar_sha256"])
    if summary["inputs"].get("table_sidecar"):
        audit.track(
            Path(summary["inputs"]["table_sidecar"]),
            summary["inputs"]["table_sidecar_sha256"],
        )
    index = ParentChildIndex.from_directory(directory)
    parents, children = list(index.parents.values()), list(index.children.values())
    _, blocks = read_sidecar(source)
    block_map = {b["blockid"]: b for b in blocks}
    identity = summary["doc_id"]
    audit.check(
        "source_content_coverage", identity, lambda: validate_coverage(blocks, parents)
    )
    audit.check(
        "source_position_coverage",
        identity,
        lambda: check_position_coverage(parents, blocks),
    )
    audit.check(
        "child_structure",
        identity,
        lambda: validate_children(
            parents, children, tokenizer, ChildConfig(**summary["config"])
        ),
    )
    for parent in parents:
        audit.check(
            "parent_source_positions",
            parent["parent_id"],
            lambda p=parent: check_parent_positions(p, block_map),
        )
    for kind, items in (("parent", parents), ("child", children)):
        for row in items:
            audit.check(
                f"{kind}_pages",
                row[f"{kind}_id"],
                lambda r=row: check_pages(r, parent_summary["pdf_pages"]),
            )
    return parents, children


def run_audit(args, tokenizer=None) -> dict:
    audit = Audit()
    if tokenizer is None:
        tokenizer = load_local_tokenizer(audit.track(args.tokenizer))
    parents, children = [], []
    for directory in args.bundle:

        def load(directory=directory):
            ps, cs = audit_bundle(audit, directory, tokenizer)
            parents.extend(ps)
            children.extend(cs)

        audit.check("bundle_artifacts_and_ancestry", str(directory), load)
    docs, rows = audit.json(args.full_docs), audit.json(args.text_chunks)
    grouped: dict[str, list] = {}
    for child in children:
        grouped.setdefault(child["parent_id"], []).append(child)
    for parent in parents:

        def stored_parent(p=parent):
            doc = docs[p["doc_id"]]
            require(
                doc["file_path"] == p["document_name"],
                "Document source filename differs",
            )
            require(
                doc["parent_child"]["doc_id"] == p["doc_id"],
                "Document bundle owner differs",
            )
            check_stored_parent(
                p,
                doc["parent_child"]["parents"][p["parent_id"]],
                grouped.get(p["parent_id"], []),
            )

        audit.check("stored_parent", parent["parent_id"], stored_parent)
    for child in children:
        key = f"{child['doc_id']}-pc-{child['child_id']}"
        audit.check(
            "stored_child",
            key,
            lambda c=child, k=key: check_stored_child(c, rows[k], k),
        )
    expected_parents = {p["parent_id"] for p in parents}
    actual_parents = {
        k for d in docs.values() for k in d.get("parent_child", {}).get("parents", {})
    }
    audit.check(
        "stored_parent_inventory",
        "all",
        lambda: require(
            actual_parents == expected_parents, "Stored parent inventory differs"
        ),
    )
    expected_children = {f"{c['doc_id']}-pc-{c['child_id']}" for c in children}
    actual_children = {k for k, r in rows.items() if r.get("parent_child")}
    audit.check(
        "stored_child_inventory",
        "all",
        lambda: require(
            actual_children == expected_children, "Stored child inventory differs"
        ),
    )
    records = audit.rows(args.retrieval_results)
    audit.check(
        "saved_retrieval_inventory",
        "all",
        lambda: require(bool(records), "No saved retrieval responses"),
    )
    for number, record in enumerate(records, 1):

        def replay(record=record, number=number):
            response = record["response"]
            require(response["status"] == "success", "Saved retrieval did not succeed")
            chunks = response["data"]["chunks"]
            mappings = response["metadata"]["parent_child"]["parents"]
            keys = [c["chunk_id"] for c in chunks]
            require(
                bool(keys) and len(keys) == len(set(keys)),
                "Empty/duplicate evidence groups",
            )
            require(
                set(keys) == {m["source_chunk_id"] for m in mappings},
                "Unmapped evidence or dangling source",
            )
            for chunk in chunks:
                audit.check(
                    "retrieved_evidence",
                    f"{number}:{chunk['chunk_id']}",
                    lambda c=chunk: check_evidence(c, mappings, docs, rows),
                )

        audit.check("saved_retrieval_response", str(number), replay)
    report = audit.finish()
    report["corpus"] = {
        "documents": len({p["doc_id"] for p in parents}),
        "parents": len(parents),
        "body_parents": sum(p["content_role"] == "body" for p in parents),
        "children": len(children),
        "transformed_table_children": sum(
            bool(c.get("transformation")) for c in children
        ),
        "required_note_occurrences": sum(
            len(c.get("required_notes", [])) for c in children
        ),
        "table_note_occurrences": sum(len(c.get("table_notes", [])) for c in children),
        "parent_review_flags": dict(
            Counter(f for p in parents for f in p.get("review_flags", []))
        ),
        "saved_responses": len(records),
    }
    return report


def write_report(output: Path, report: dict) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Parent/child integrity audit",
        "",
        f"Passed: {report['passed']}",
        "",
        "| Check | Total | Passed | Failed |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {k} | {v['checks']} | {v['passed']} | {v['failed']} |"
        for k, v in report["counts"].items()
    )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            *[f"- {s}" for s in report["limitations"]],
            "",
            "## Failures",
            "",
        ]
    )
    lines.extend(
        f"- {c['category']} / {c['id']}: {c['error']}"
        for c in report["checks"]
        if not c["passed"]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--full-docs", type=Path, required=True)
    parser.add_argument("--text-chunks", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; choose a new audit directory")
    try:
        report = run_audit(args)
    except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
        failed = Audit()
        message = f"Audit could not complete: {type(exc).__name__}: {exc}"

        def raise_input_error():
            raise ValueError(message)

        failed.check("audit_input", "incomplete_run", raise_input_error)
        report = failed.finish()
    write_report(args.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
