"""Offline CLI for the shared document organizer; no model calls or indexing."""

import argparse
from pathlib import Path

from lightrag.document_structure.parents import prepare


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--parsed-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scope-rules", type=Path)
    parser.add_argument("--structure-rules", type=Path)
    parser.add_argument("--max-parent-chars", type=int, default=10000)
    args = parser.parse_args(argv)
    from lightrag.utils import logger

    summary = prepare(
        args.pdf,
        args.parsed_dir,
        args.output,
        args.scope_rules,
        args.max_parent_chars,
        args.structure_rules,
    )
    logger.info(
        "Parent candidates: %s; alignment=%s; output=%s",
        summary["parents"],
        summary["alignment_counts"],
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
