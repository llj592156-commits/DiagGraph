"""H is an explicit peer strategy with a durable per-document parameter snapshot."""

import pytest

from lightrag.parser.param_schema import parse_chunk_params
from lightrag.parser.routing import (
    canonicalize_parser_hinted_basename,
    parse_process_options,
    resolve_chunk_options,
    resolve_parser_directives,
    slim_chunk_options,
    validate_process_options,
)

pytestmark = pytest.mark.offline


def test_h_filename_routes_docling_and_preserves_canonical_source():
    name = "L3266_L5298.[docling-H(chunk_ts=420,chunk_ol=50)].pdf"
    result = resolve_parser_directives(
        name, parser_rules="", require_external_endpoint=False
    )
    assert result.engine == "docling"
    assert result.process_options == "H"
    assert result.chunk_params == {
        "H": {"chunk_token_size": 420, "chunk_overlap_token_size": 50}
    }
    assert canonicalize_parser_hinted_basename(name) == "L3266_L5298.pdf"


def test_h_options_default_snapshot_and_aliases():
    assert parse_process_options("H!").chunking == "H"
    assert parse_process_options("H!").skip_kg
    default = resolve_chunk_options({}, process_options="H")
    assert default["parent_child"] == {
        "chunk_token_size": 500,
        "chunk_overlap_token_size": 60,
    }
    assert "fixed_token" not in default
    assert slim_chunk_options({"parent_child": {"chunk_token_size": 420}}, "H")[
        "parent_child"
    ] == {
        "chunk_token_size": 420,
        "chunk_overlap_token_size": 60,
    }
    params, errors = parse_chunk_params(
        "chunk_ts=420,chunk_ol=50", selector="H", label="test"
    )
    assert not errors and params == {
        "chunk_token_size": 420,
        "chunk_overlap_token_size": 50,
    }


@pytest.mark.parametrize("value", ["HF", "HR", "HV", "HP", "HC", "Hi", "Ht", "He"])
def test_h_rejects_competing_chunkers_and_unmapped_analysis(value):
    assert validate_process_options(value)
