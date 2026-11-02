"""Build a compact Arrow context artifact for cloud investigation requests.

This is an offline preprocessing step. It reads the original train and
validation Parquet files, retains only fields used by the evidence tools, and
writes a separate artifact. Neither source file is modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "TransactionID", "TransactionDT", "card1", "addr1",
    "P_emaildomain", "TransactionAmt", "isFraud",
]


def prepare(train_path: Path, validation_path: Path, output_path: Path, row_group_size: int) -> None:
    train_path = train_path.resolve()
    validation_path = validation_path.resolve()
    output_path = output_path.resolve()
    if output_path in {train_path, validation_path}:
        raise ValueError("Output must not overwrite either source dataset")
    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive")

    train = pq.read_table(train_path, columns=FIELDS)
    validation = pq.read_table(validation_path, columns=FIELDS)
    if not train.schema.equals(validation.schema):
        unified = pa.unify_schemas([train.schema, validation.schema])
        train = train.cast(unified)
        validation = validation.cast(unified)
    combined = pa.concat_tables([train, validation])

    # Keep duplicate transaction rows and maintain source order for ties.
    combined = combined.append_column(
        "__fraudlens_source_ordinal__",
        pa.array(range(combined.num_rows), type=pa.int64()),
    )
    indices = pc.sort_indices(
        combined,
        sort_keys=[
            ("card1", "ascending"),
            ("addr1", "ascending"),
            ("TransactionDT", "ascending"),
            ("__fraudlens_source_ordinal__", "ascending"),
        ],
        null_placement="at_end",
    )
    sorted_table = combined.take(indices).drop(["__fraudlens_source_ordinal__"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        pq.write_table(
            sorted_table,
            temporary,
            row_group_size=row_group_size,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        prepared = pq.read_table(temporary)
        if prepared.num_rows != train.num_rows + validation.num_rows:
            raise RuntimeError("Prepared context row count does not match train + validation")
        if not prepared.schema.equals(sorted_table.schema) or not prepared.equals(sorted_table):
            raise RuntimeError("Prepared context schema, values, or nulls differ from source rows")
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    metadata = pq.ParquetFile(output_path).metadata
    print(f"Rows retained: {metadata.num_rows:,}")
    print(f"Columns retained: {', '.join(FIELDS)}")
    print(f"Row groups: {metadata.num_row_groups:,}")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/ieee_train.parquet")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/processed/ieee_val.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/ieee_investigation_context.parquet")
    parser.add_argument("--row-group-size", type=int, default=2048)
    args = parser.parse_args()
    prepare(args.train, args.validation, args.output, args.row_group_size)


if __name__ == "__main__":
    main()
