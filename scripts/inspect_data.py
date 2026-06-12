"""Quick schema inspection of the competition parquet files."""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

DATA = "Dataset"
files = [
    "X_train.parquet",
    "y_train.parquet",
    "y_train_index.parquet",
    "X_test.reduced.parquet",
    "y_test.reduced.parquet",
    "y_test_index.reduced.parquet",
]

for f in files:
    path = f"{DATA}/{f}"
    pf = pq.ParquetFile(path)
    md = pf.metadata
    print("=" * 70)
    print(f"FILE: {f}")
    print(f"  rows={md.num_rows:,}  cols={md.num_columns}  row_groups={md.num_row_groups}")
    print(f"  schema: {[c for c in pf.schema_arrow.names]}")
    # read a small head
    head = pf.read_row_group(0).slice(0, 8).to_pandas()
    print(head.to_string())
    print()
