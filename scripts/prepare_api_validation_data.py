"""Create and verify a sorted, small-row-group API copy of validation data.

The source file is never modified. Run from any directory with:
    python scripts/prepare_api_validation_data.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "processed" / "ieee_val.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "ieee_val_api.parquet"


def prepare(source: Path, output: Path, row_group_size: int) -> None:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("Output must differ from the original validation file.")
    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive")

    original = pq.read_table(source)
    if "TransactionID" not in original.column_names:
        raise ValueError("Source Parquet has no TransactionID column")

    ids = original["TransactionID"].to_pylist()
    counts = Counter(ids)
    duplicate_ids = sum(count > 1 for count in counts.values())
    duplicate_extra_rows = sum(count - 1 for count in counts.values() if count > 1)

    # Add a temporary source ordinal as a tie-breaker, preserving source order
    # for duplicate TransactionIDs (and therefore preserving lookup behavior).
    with_ordinal = original.append_column(
        "__fraudlens_source_ordinal__",
        pa.array(range(original.num_rows), type=pa.int64()),
    )
    order = pc.sort_indices(
        with_ordinal,
        sort_keys=[("TransactionID", "ascending"), ("__fraudlens_source_ordinal__", "ascending")],
        null_placement="at_end",
    )
    sorted_table = with_ordinal.take(order).drop(["__fraudlens_source_ordinal__"])

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        pq.write_table(
            sorted_table,
            temporary,
            row_group_size=row_group_size,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        optimized = pq.read_table(temporary)
        if optimized.num_rows != original.num_rows:
            raise RuntimeError(f"Row count changed: {original.num_rows} -> {optimized.num_rows}")
        if not optimized.schema.equals(original.schema):
            raise RuntimeError("Output schema differs from source schema")
        if not optimized.equals(sorted_table):
            raise RuntimeError("Output values or nulls differ from the sorted source")

        # Only replace the generated output after all checks pass.
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    metadata = pq.ParquetFile(output).metadata
    print(f"Source rows: {original.num_rows:,}")
    print(f"Output rows: {metadata.num_rows:,}")
    print(f"Duplicate TransactionID values: {duplicate_ids:,}")
    print(f"Additional rows retained for duplicate IDs: {duplicate_extra_rows:,}")
    print(f"Output row groups: {metadata.num_row_groups:,}")
    print(f"Output: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--row-group-size", type=int, default=2048)
    args = parser.parse_args()
    prepare(args.source, args.output, args.row_group_size)


if __name__ == "__main__":
    main()
