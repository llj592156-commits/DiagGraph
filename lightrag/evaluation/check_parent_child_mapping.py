"""Check selected printer-manual child-to-parent mappings, without searching.

These are deterministic structure checks on manual B, not retrieval-quality
scores. Expected observations belong in this report, never in an index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lightrag.document_structure.retrieval import ParentChildIndex
from lightrag.utils import logger


def check_printer_mappings(index: ParentChildIndex) -> list[dict]:
    parents = list(index.parents.values())

    def choose(predicate) -> dict:
        matched = [p for p in parents if predicate(p)]
        if len(matched) != 1:
            raise ValueError(
                f"Expected one printer example parent, found {len(matched)}"
            )
        return matched[0]

    network = choose(
        lambda p: p["feature_domain"] == "network" and p["heading_path"][-1] == "E-12"
    )
    display = choose(
        lambda p: (
            p["feature_domain"] == "device_display_errors" and "E-12" in p["content"]
        )
    )
    adf_specs = choose(lambda p: p["heading_path"][-1] == "ADF 规格")
    adf_loading = choose(
        lambda p: p["heading_path"] == ["放置原稿", "将原稿放在 ADF 上"]
    )
    base = ["故障排除", "本产品未能正常工作", "无法打印"]
    windows = choose(
        lambda p: (
            p["heading_path"] == [*base, "无法从 Windows 进行打印", "无法连接到网络"]
        )
    )
    mac = choose(
        lambda p: p["heading_path"] == [*base, "无法从 Mac OS 打印", "无法连接到网络"]
    )

    def child(parent: dict, text: str | None = None) -> str:
        matches = [
            c
            for c in index.children.values()
            if c["parent_id"] == parent["parent_id"]
            and (text is None or text in c["content"])
        ]
        if not matches:
            raise ValueError("No child with the expected example content")
        return matches[0]["child_id"]

    cases = [
        (
            "network-E12-domain",
            [child(network, "E-12"), child(display, "E-12")],
            {"model": "L3266", "feature_domain": "network"},
            [network],
            1,
        ),
        (
            "display-E12-domain",
            [child(network, "E-12"), child(display, "E-12")],
            {"model": "L3266", "feature_domain": "device_display_errors"},
            [display],
            1,
        ),
        (
            "ADF-explicit-model-accepted",
            [child(adf_specs)],
            {"model": "L5298"},
            [adf_specs],
            0,
        ),
        ("ADF-explicit-model-refused", [child(adf_specs)], {"model": "L3266"}, [], 1),
        (
            "ADF-filename-default-preserved",
            [child(adf_loading)],
            {"model": "L3266"},
            [adf_loading],
            0,
        ),
        (
            "Windows-context-and-platform",
            [child(windows), child(mac)],
            {"model": "L3266", "platform": "Windows"},
            [windows],
            1,
        ),
        (
            "Mac-context-and-platform",
            [child(windows), child(mac)],
            {"model": "L3266", "platform": "Mac OS"},
            [mac],
            1,
        ),
    ]
    results = []
    for name, ids, filters, expected, rejected_count in cases:
        result = index.resolve(ids, **filters)
        expected_ids = {p["parent_id"] for p in expected}
        direct_ids = {
            index.children[key]["parent_id"] for key in result["accepted_child_ids"]
        }
        returned = {p["parent_id"]: p for p in result["parents"]}
        expected_context = {key for p in expected for key in p["context_parent_ids"]}
        passed = (
            direct_ids == expected_ids
            and len(result["rejected"]) == rejected_count
            and expected_ids | expected_context == set(returned)
            and all(
                returned[key]["content"] == index.parents[key]["content"]
                for key in returned
            )
        )
        results.append(
            {
                "case": name,
                "passed": passed,
                "filters": filters,
                "selected_child_ids": ids,
                "accepted_child_ids": result["accepted_child_ids"],
                "returned_parents": [
                    {
                        "parent_id": p["parent_id"],
                        "heading_path": p["heading_path"],
                        "pages": p.get("pages", []),
                    }
                    for p in result["parents"]
                ],
                "context_only_parent_ids": result["context_only_parent_ids"],
                "rejected": result["rejected"],
                "requires_review": result["requires_review"],
            }
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Output already exists")
    results = check_printer_mappings(ParentChildIndex.from_directory(args.bundle_dir))
    passed = sum(row["passed"] for row in results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "mapping_only": True,
                "passed": passed,
                "total": len(results),
                "cases": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Printer child-parent mappings: %s/%s", passed, len(results))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
