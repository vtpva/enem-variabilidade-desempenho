import os
import pandas as pd
import pyarrow.parquet as pq

TCC_DIR = "/content/drive/MyDrive/TCC"
years = [2018, 2019, 2020, 2021, 2022]

print("=== INSPECTING ENEM PARQUET DATA ON GOOGLE DRIVE ===")
for y in years:
    path = os.path.join(TCC_DIR, f"MICRODADOS_ENEM_{y}.parquet")
    if os.path.exists(path):
        schema = pq.read_schema(path)
        meta = pq.read_metadata(path)
        print(f"\n--- Year {y} ---")
        print(f"Path: {path}")
        print(f"Num rows: {meta.num_rows:,}, Num columns: {meta.num_columns}")
        print(f"Columns: {schema.names}")

        # Read a tiny sample of 5 rows
        df_sample = pd.read_parquet(path).head(5)
        print("Sample head:")
        print(df_sample.iloc[:, :12].to_string())
        print(df_sample.iloc[:, 12:24].to_string())
        if df_sample.shape[1] > 24:
            print(df_sample.iloc[:, 24:].to_string())
