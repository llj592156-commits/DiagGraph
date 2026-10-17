"""Generate offline children from an existing completed parent draft."""

from __future__ import annotations

import argparse
from pathlib import Path

from lightrag.document_structure.children import (
    VOCAB_SHA256,
    ChildConfig,
    load_local_tokenizer,
    prepare_children,
)
from lightrag.utils import logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer-file", required=True, type=Path)
    parser.add_argument("--target-tokens", type=int, default=500)
    parser.add_argument("--overlap-tokens", type=int, default=60)
    parser.add_argument("--table-header-overrides", type=Path)
    args = parser.parse_args(argv)
    config = ChildConfig(args.target_tokens, args.overlap_tokens)
    summary = prepare_children(
        args.parent_dir,
        args.output,
        load_local_tokenizer(args.tokenizer_file),
        config,
        tokenizer_sha256=VOCAB_SHA256,
        table_header_overrides=args.table_header_overrides,
    )
    logger.info("Offline children: %s; output=%s", summary["children"], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
