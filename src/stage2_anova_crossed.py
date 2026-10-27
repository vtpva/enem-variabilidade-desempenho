"""Stage 2: Henderson ANOVA (Model 1) & Longitudinal Crossed (Model 2).

Reads pre-aggregated cluster data and outputs Tables 5 and 6.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = "/content/tcc_output"
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
TAB_DIR = os.path.join(OUT_DIR, "tables")

print("=== STAGE 2: HENDERSON ANOVA & LONGITUDINAL CROSSED ===")

agg_df = pd.read_parquet(
    os.path.join(OUT_DIR, "cluster_year_aggregates.parquet")
)
summary_df = pd.read_csv(
    os.path.join(TAB_DIR, "tabela1_painel_cohortes_tri.csv")
)

DISCIPLINES = ["MT", "CN", "CH", "LC", "TRI"]
SCORE_COLS = {
    "MT": "mean_MT",
    "CN": "mean_CN",
    "CH": "mean_CH",
    "LC": "mean_LC",
    "TRI": "mean_TRI"
}
SS_COLS = {
    "MT": "ss_MT",
    "CN": "ss_CN",
    "CH": "ss_CH",
    "LC": "ss_LC",
    "TRI": "ss_TRI"
}
DISC_NAMES = {
    "MT": "Matemática e suas Tecnologias",
    "CN": "Ciências da Natureza e suas Tecnologias",
    "CH": "Ciências Humanas e suas Tecnologias",
    "LC": "Linguagens, Códigos e suas Tecnologias",
    "TRI": "Média 4 Provas TRI"
}

N_total = agg_df["n_ij"].sum()
I_years = agg_df["NU_ANO"].nunique()
J_clusters = len(agg_df)

anova_results = []

for disc in DISCIPLINES:
    m_col = SCORE_COLS[disc]
    ss_col = SS_COLS[disc]

    # Grand mean
    y_bar = (agg_df["n_ij"] * agg_df[m_col]).sum() / N_total

    # Year means and counts
    year_stats = agg_df.groupby("NU_ANO").apply(
        lambda g: pd.Series({
            "n_i": g["n_ij"].sum(),
            "mean_i": (g["n_ij"] * g[m_col]).sum() / g["n_ij"].sum(),
            "sum_nij2": (g["n_ij"]**2).sum()
        }),
        include_groups=False
    )

    SQA = np.sum(
        year_stats["n_i"] * (year_stats["mean_i"] - y_bar)**2
    )
    GL_A = I_years - 1
    QMA = SQA / GL_A

    # School within Year
    agg_df["year_mean"] = agg_df["NU_ANO"].map(year_stats["mean_i"])
    SQB_A = np.sum(
        agg_df["n_ij"] * (agg_df[m_col] - agg_df["year_mean"])**2
    )
    GL_B_A = J_clusters - I_years
    QMB_A = SQB_A / GL_B_A

    # Residual sum of squares from within-cell variances
    SQE = agg_df[ss_col].sum()
    GL_E = N_total - J_clusters
    QME = SQE / GL_E

    # Henderson unbalance coefficients
    k1 = (
        N_total - np.sum(year_stats["sum_nij2"] / year_stats["n_i"])
    ) / GL_B_A
    k3 = (N_total - np.sum(year_stats["n_i"]**2) / N_total) / GL_A

    sigma2_e = QME
    sigma2_B_A = max(0, (QMB_A - QME) / k1)
    sigma2_A = max(0, (QMA - QMB_A) / k3)
    sigma2_total = sigma2_A + sigma2_B_A + sigma2_e

    vpc_A = (sigma2_A / sigma2_total) * 100
    vpc_B_A = (sigma2_B_A / sigma2_total) * 100
    vpc_e = (sigma2_e / sigma2_total) * 100
    rho_escola = (sigma2_A + sigma2_B_A) / sigma2_total * 100

    F_escola = QMB_A / QME
    p_escola = 1.0 - stats.f.cdf(F_escola, GL_B_A, GL_E)
    F_ano = QMA / QMB_A
    p_ano = 1.0 - stats.f.cdf(F_ano, GL_A, GL_B_A)

    anova_results.append({
        "Area": disc if disc != "TRI" else "TRI_MEDIA",
        "Nome_Area": DISC_NAMES[disc],
        "SQA": SQA,
        "SQB_A": SQB_A,
        "SQE": SQE,
        "sigma2_Ano": sigma2_A,
        "sigma2_Escola": sigma2_B_A,
        "sigma2_Aluno": sigma2_e,
        "sigma2_Total": sigma2_total,
        "VPC_Ano(%)": vpc_A,
        "VPC_Escola(%)": vpc_B_A,
        "VPC_Aluno(%)": vpc_e,
        "ICC_Escola(%)": rho_escola,
        "F_Ano": F_ano,
        "p_Ano": p_ano,
        "F_Escola": F_escola,
        "p_Escola": p_escola
    })

anova_df = pd.DataFrame(anova_results)
print("\nANOVA Variance Decomposition (Henderson III):")
print(anova_df.to_string(index=False))

anova_df.to_csv(
    os.path.join(TAB_DIR, "tabela5_anova_modelo1_tri.csv"), index=False
)
with open(
    os.path.join(TAB_DIR, "tabela5_anova_modelo1_tri.tex"), "w"
) as f:
    f.write(anova_df.to_latex(index=False, float_format="%.3f"))

# Model 2: Longitudinal Crossed (5-year balanced schools)
print("\n=== MODEL 2: LONGITUDINAL CROSSED PANEL ===")
cluster_year_counts = agg_df.groupby(["CLUSTER_ID", "NU_ANO"]).size().unstack()
balanced_clusters = cluster_year_counts[
    cluster_year_counts.notna().all(axis=1)
].index
print(f"Balanced clusters present in all 5 years: {len(balanced_clusters):,}")

agg_long = agg_df[agg_df["CLUSTER_ID"].isin(balanced_clusters)].copy()

crossed_results = []
for disc in DISCIPLINES:
    m_col = SCORE_COLS[disc]
    cell_means = agg_long.pivot(
        index="CLUSTER_ID", columns="NU_ANO", values=m_col
    )
    y_bar = cell_means.values.mean()

    school_means = cell_means.mean(axis=1)
    year_means = cell_means.mean(axis=0)

    var_school_persistent = school_means.var(ddof=1)
    var_year_main = year_means.var(ddof=1)

    cell_res = (
        cell_means.sub(school_means, axis=0).sub(year_means, axis=1) + y_bar
    )
    var_interaction = (cell_res**2).values.mean()

    m1_row = anova_df.loc[
        anova_df["Area"] == (disc if disc != "TRI" else "TRI_MEDIA")
    ]
    var_within_student = m1_row["sigma2_Aluno"].values[0]

    tot_var = (
        var_year_main + var_school_persistent +
        var_interaction + var_within_student
    )
    persist_index = (
        var_school_persistent / (var_school_persistent + var_interaction)
    ) * 100

    crossed_results.append({
        "Area": disc if disc != "TRI" else "TRI_MEDIA",
        "sigma2_Ano": var_year_main,
        "sigma2_Escola_Persistente": var_school_persistent,
        "sigma2_Interacao": var_interaction,
        "sigma2_Aluno_Intra": var_within_student,
        "VPC_Ano(%)": (var_year_main / tot_var) * 100,
        "VPC_Escola_Persistente(%)": (var_school_persistent / tot_var) * 100,
        "VPC_Interacao(%)": (var_interaction / tot_var) * 100,
        "VPC_Aluno(%)": (var_within_student / tot_var) * 100,
        "Indice_Persistencia(%)": persist_index
    })

crossed_df = pd.DataFrame(crossed_results)
print("\nModel 2 Crossed Longitudinal Panel Decomposition:")
print(crossed_df.to_string(index=False))

crossed_df.to_csv(
    os.path.join(TAB_DIR, "tabela6_cruzado_modelo2_tri.csv"), index=False
)
with open(
    os.path.join(TAB_DIR, "tabela6_cruzado_modelo2_tri.tex"), "w"
) as f:
    f.write(crossed_df.to_latex(index=False, float_format="%.3f"))

with open(os.path.join(CKPT_DIR, "stage2_done.json"), "w") as f:
    json.dump({"status": "DONE"}, f)

print("=== STAGE 2 COMPLETED! ===")
