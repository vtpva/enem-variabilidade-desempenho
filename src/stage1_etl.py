"""Stage 1: ETL, Filtering, INSE Construction and Sampling (2015-2019).

Checkpoints to /content/tcc_output/checkpoints/stage1_done.json
"""

import json
import os
import pandas as pd

TCC_DIR = "/content/drive/MyDrive/TCC"
OUT_DIR = "/content/tcc_output"
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
TAB_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

YEARS = [2015, 2016, 2017, 2018, 2019]

print("=== STAGE 1: INGESTION AND HARMONIZATION (2015-2019) ===")

cohort_summary = []
sample_dfs = []
cluster_year_agg = []
dep_stats = []
raca_stats = []
corr_records = []

# Dep weights and demographic weights for INSE
dep_weights = {1: 1.5, 2: -0.2, 3: -0.4, 4: 1.8}
raca_weights = {1: 0.8, 2: -0.5, 3: -0.3, 4: 0.5, 5: -0.6, 0: 0.0}
loc_weights = {1: 0.2, 2: -0.8}
age_weights = {1: 0.5, 2: 0.4, 3: 0.2, 4: -0.2, 5: -0.5, 6: -0.8}

total_records_count = 0

for y in YEARS:
    fpath = os.path.join(TCC_DIR, f"MICRODADOS_ENEM_{y}.parquet")
    raw = pd.read_parquet(fpath)
    n_raw = len(raw)

    cond = (
        (raw["TP_ST_CONCLUSAO"] == 2) &
        raw["NU_NOTA_MT"].notna() & (raw["NU_NOTA_MT"] > 0) &
        raw["NU_NOTA_CN"].notna() & (raw["NU_NOTA_CN"] > 0) &
        raw["NU_NOTA_CH"].notna() & (raw["NU_NOTA_CH"] > 0) &
        raw["NU_NOTA_LC"].notna() & (raw["NU_NOTA_LC"] > 0) &
        raw["CO_MUNICIPIO_ESC"].notna() &
        raw["TP_DEPENDENCIA_ADM_ESC"].notna()
    )
    df_y = raw[cond].copy()
    df_y["NU_ANO"] = y

    df_y["NOTA_TRI_MEDIA"] = (
        df_y["NU_NOTA_MT"] + df_y["NU_NOTA_CN"] +
        df_y["NU_NOTA_CH"] + df_y["NU_NOTA_LC"]
    ) / 4.0

    df_y["CLUSTER_ID"] = (
        df_y["SG_UF_ESC"].astype(str) + "_" +
        df_y["CO_MUNICIPIO_ESC"].astype(int).astype(str) + "_" +
        df_y["TP_DEPENDENCIA_ADM_ESC"].astype(int).astype(str) + "_" +
        df_y["TP_LOCALIZACAO_ESC"].fillna(1).astype(int).astype(str)
    )

    c_counts = df_y["CLUSTER_ID"].value_counts()
    valid_c = c_counts[c_counts >= 10].index
    df_y = df_y[df_y["CLUSTER_ID"].isin(valid_c)].copy()

    df_y["INSE_IND"] = (
        df_y["TP_DEPENDENCIA_ADM_ESC"].map(dep_weights).fillna(0) * 1.5 +
        df_y["TP_COR_RACA"].map(raca_weights).fillna(0) * 0.8 +
        df_y["TP_LOCALIZACAO_ESC"].map(loc_weights).fillna(0) * 0.6 +
        df_y["TP_FAIXA_ETARIA"].map(age_weights).fillna(0) * 0.5 +
        (df_y["TP_LINGUA"] == 0).astype(float) * 0.4
    )
    df_y["INSE_IND"] = (
        df_y["INSE_IND"] - df_y["INSE_IND"].mean()
    ) / df_y["INSE_IND"].std()

    school_inse = df_y.groupby("CLUSTER_ID")["INSE_IND"].transform("mean")
    df_y["INSE_ESCOLA"] = school_inse
    df_y["INSE_CENTR"] = df_y["INSE_IND"] - df_y["INSE_ESCOLA"]

    n_filtered = len(df_y)
    n_clusters = df_y["CLUSTER_ID"].nunique()
    total_records_count += n_filtered

    cohort_summary.append({
        "Ano": y,
        "Inscritos_Brutos": n_raw,
        "Concluintes_Validos": n_filtered,
        "Num_Escolas": n_clusters,
        "Media_MT": float(df_y["NU_NOTA_MT"].mean()),
        "DP_MT": float(df_y["NU_NOTA_MT"].std()),
        "Media_CN": float(df_y["NU_NOTA_CN"].mean()),
        "DP_CN": float(df_y["NU_NOTA_CN"].std()),
        "Media_CH": float(df_y["NU_NOTA_CH"].mean()),
        "DP_CH": float(df_y["NU_NOTA_CH"].std()),
        "Media_LC": float(df_y["NU_NOTA_LC"].mean()),
        "DP_LC": float(df_y["NU_NOTA_LC"].std()),
        "Media_TRI": float(df_y["NOTA_TRI_MEDIA"].mean()),
        "DP_TRI": float(df_y["NOTA_TRI_MEDIA"].std())
    })

    # Aggregates for ANOVA Henderson
    agg_y = df_y.groupby("CLUSTER_ID").agg(
        n_ij=("NU_NOTA_MT", "count"),
        mean_MT=("NU_NOTA_MT", "mean"),
        mean_CN=("NU_NOTA_CN", "mean"),
        mean_CH=("NU_NOTA_CH", "mean"),
        mean_LC=("NU_NOTA_LC", "mean"),
        mean_TRI=("NOTA_TRI_MEDIA", "mean"),
        ss_MT=("NU_NOTA_MT", lambda x: ((x - x.mean())**2).sum()),
        ss_CN=("NU_NOTA_CN", lambda x: ((x - x.mean())**2).sum()),
        ss_CH=("NU_NOTA_CH", lambda x: ((x - x.mean())**2).sum()),
        ss_LC=("NU_NOTA_LC", lambda x: ((x - x.mean())**2).sum()),
        ss_TRI=("NOTA_TRI_MEDIA", lambda x: ((x - x.mean())**2).sum())
    ).reset_index()
    agg_y["NU_ANO"] = y
    cluster_year_agg.append(agg_y)

    # Sample 20,000 per year (total 100,000) for REML
    sample_y = df_y.sample(n=min(len(df_y), 20000), random_state=42)
    sample_dfs.append(sample_y)

    print(f"Year {y}: {n_filtered:,} concluintes in {n_clusters:,} clusters.")

# Concatenate sample and aggregates
sample_all = pd.concat(sample_dfs, ignore_index=True)
sample_all.to_parquet(
    os.path.join(OUT_DIR, "sample_reml_200k.parquet"), index=False
)

cluster_agg_all = pd.concat(cluster_year_agg, ignore_index=True)
cluster_agg_all.to_parquet(
    os.path.join(OUT_DIR, "cluster_year_aggregates.parquet"), index=False
)

summary_df = pd.DataFrame(cohort_summary)
summary_df.to_csv(
    os.path.join(TAB_DIR, "tabela1_painel_cohortes_tri.csv"), index=False
)
with open(
    os.path.join(TAB_DIR, "tabela1_painel_cohortes_tri.tex"), "w"
) as f:
    f.write(summary_df.to_latex(index=False, float_format="%.2f"))

# Descriptive tables from sample
dep_desc = []
dep_map = {1: "Federal", 2: "Estadual", 3: "Municipal", 4: "Privada"}
for dep, dlabel in dep_map.items():
    sub = sample_all[sample_all["TP_DEPENDENCIA_ADM_ESC"] == dep]
    dep_desc.append({
        "Rede": dlabel,
        "N_Amostral": len(sub),
        "Perc(%)": len(sub) / len(sample_all) * 100,
        "MT_Media": sub["NU_NOTA_MT"].mean(),
        "MT_DP": sub["NU_NOTA_MT"].std(),
        "CN_Media": sub["NU_NOTA_CN"].mean(),
        "CN_DP": sub["NU_NOTA_CN"].std(),
        "CH_Media": sub["NU_NOTA_CH"].mean(),
        "CH_DP": sub["NU_NOTA_CH"].std(),
        "LC_Media": sub["NU_NOTA_LC"].mean(),
        "LC_DP": sub["NU_NOTA_LC"].std(),
        "TRI_Media": sub["NOTA_TRI_MEDIA"].mean(),
        "TRI_DP": sub["NOTA_TRI_MEDIA"].std()
    })
dep_df = pd.DataFrame(dep_desc)
dep_df.to_csv(
    os.path.join(TAB_DIR, "tabela2_descritiva_redes_ensino.csv"), index=False
)
with open(
    os.path.join(TAB_DIR, "tabela2_descritiva_redes_ensino.tex"), "w"
) as f:
    f.write(dep_df.to_latex(index=False, float_format="%.2f"))

raca_desc = []
raca_map = {
    1: "Branca", 2: "Preta", 3: "Parda",
    4: "Amarela", 5: "Indígena", 0: "Não Declarada"
}
for r, rlabel in raca_map.items():
    sub = sample_all[sample_all["TP_COR_RACA"] == r]
    if len(sub) > 0:
        raca_desc.append({
            "Raca_Cor": rlabel,
            "N_Amostral": len(sub),
            "Perc(%)": len(sub) / len(sample_all) * 100,
            "MT_Media": sub["NU_NOTA_MT"].mean(),
            "CN_Media": sub["NU_NOTA_CN"].mean(),
            "CH_Media": sub["NU_NOTA_CH"].mean(),
            "LC_Media": sub["NU_NOTA_LC"].mean(),
            "TRI_Media": sub["NOTA_TRI_MEDIA"].mean()
        })
raca_df = pd.DataFrame(raca_desc)
raca_df.to_csv(
    os.path.join(TAB_DIR, "tabela3_descritiva_raca_cor.csv"), index=False
)
with open(
    os.path.join(TAB_DIR, "tabela3_descritiva_raca_cor.tex"), "w"
) as f:
    f.write(raca_df.to_latex(index=False, float_format="%.2f"))

# Correlation Matrix
corr_cols = [
    "NU_NOTA_MT", "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC",
    "CORRECT_MT", "CORRECT_CN", "CORRECT_CH", "CORRECT_LC"
]
corr_matrix = sample_all[corr_cols].corr()
corr_matrix.to_csv(
    os.path.join(TAB_DIR, "tabela4_matriz_correlacao_tri_raw.csv")
)
with open(
    os.path.join(TAB_DIR, "tabela4_matriz_correlacao_tri_raw.tex"), "w"
) as f:
    f.write(corr_matrix.to_latex(float_format="%.3f"))

with open(os.path.join(CKPT_DIR, "stage1_done.json"), "w") as f:
    json.dump({"total_records": total_records_count, "status": "DONE"}, f)

print(f"=== STAGE 1 FINISHED! Total: {total_records_count:,} records. ===")
