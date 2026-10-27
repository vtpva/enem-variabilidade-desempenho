import os
import pandas as pd

TCC_DIR = "/content/drive/MyDrive/TCC"
years = [2018, 2019, 2020, 2021, 2022]

for y in years:
    path = os.path.join(TCC_DIR, f"MICRODADOS_ENEM_{y}.parquet")
    df = pd.read_parquet(path)
    concluintes = df[
        (df["TP_ST_CONCLUSAO"] == 2) &
        df["NU_NOTA_MT"].notna() & (df["NU_NOTA_MT"] > 0) &
        df["NU_NOTA_CN"].notna() & (df["NU_NOTA_CN"] > 0) &
        df["NU_NOTA_CH"].notna() & (df["NU_NOTA_CH"] > 0) &
        df["NU_NOTA_LC"].notna() & (df["NU_NOTA_LC"] > 0) &
        df["NU_NOTA_REDACAO"].notna() & (df["NU_NOTA_REDACAO"] > 0) &
        df["CO_MUNICIPIO_ESC"].notna() &
        df["TP_DEPENDENCIA_ADM_ESC"].notna()
    ].copy()

    # Form cluster: UF_MUN_DEP_LOC
    concluintes["CLUSTER_ID"] = (
        concluintes["SG_UF_ESC"].astype(str) + "_" +
        concluintes["CO_MUNICIPIO_ESC"].astype(int).astype(str) + "_" +
        concluintes["TP_DEPENDENCIA_ADM_ESC"].astype(int).astype(str) + "_" +
        concluintes["TP_LOCALIZACAO_ESC"].fillna(1).astype(int).astype(str)
    )

    cluster_counts = concluintes["CLUSTER_ID"].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= 10].index
    filtered = concluintes[concluintes["CLUSTER_ID"].isin(valid_clusters)]

    print(f"\n--- Year {y} ---")
    print(f"Total eligible concluintes: {len(concluintes):,}")
    print(f"Total clusters: {concluintes['CLUSTER_ID'].nunique():,}")
    print(
        f"Filtered concluintes (nj >= 10): {len(filtered):,} "
        f"({len(filtered)/len(concluintes)*100:.1f}%)"
    )
    print(f"Clusters with nj >= 10: {len(valid_clusters):,}")
    print(
        f"Cluster size: mean={cluster_counts.loc[valid_clusters].mean():.1f}, "
        f"median={cluster_counts.loc[valid_clusters].median():.0f}, "
        f"max={cluster_counts.loc[valid_clusters].max():,}"
    )
