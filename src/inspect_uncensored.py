import os
import pandas as pd
import pyarrow.parquet as pq

TCC_DIR = "/content/drive/MyDrive/TCC"
years = [2015, 2016, 2017, 2018, 2019]

print("=== INSPECTING UNCENSORED YEARS (2015-2019) ===")
for y in years:
    fpath = os.path.join(TCC_DIR, f"MICRODADOS_ENEM_{y}.parquet")
    if os.path.exists(fpath):
        schema = pq.read_schema(fpath)
        meta = pq.read_metadata(fpath)
        print(f"\n--- Year {y} ---")
        print(f"Num rows: {meta.num_rows:,}, Num columns: {meta.num_columns}")
        print(f"Columns: {schema.names}")
        sample = pd.read_parquet(fpath).head(5)
        school_cols = [
            c for c in sample.columns if "ESC" in c or "ESCOLA" in c
        ]
        print(f"School columns in {y}: {school_cols}")
        print("Sample head school cols:")
        print(sample[school_cols].to_string())
