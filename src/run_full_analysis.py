"""Full econometric and hierarchical analysis of ENEM variance decomposition.

Implements Models 1-4, ANOVA, REML, BLUP shrinkage, and diagnostics.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Set plot styles
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})

TCC_DIR = "/content/drive/MyDrive/TCC"
OUT_DIR = "/content/tcc_output"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "tables"), exist_ok=True)

YEARS = [2018, 2019, 2020, 2021, 2022]
DISCIPLINES = ["MT", "CN", "CH", "LC", "RED"]
SCORE_COLS = {
    "MT": "NU_NOTA_MT",
    "CN": "NU_NOTA_CN",
    "CH": "NU_NOTA_CH",
    "LC": "NU_NOTA_LC",
    "RED": "NU_NOTA_REDACAO"
}

print("=== STEP 1: INGESTION AND HARMONIZATION (2018-2022) ===")
dfs = []
cohort_summary = []

for y in YEARS:
    fpath = os.path.join(TCC_DIR, f"MICRODADOS_ENEM_{y}.parquet")
    raw = pd.read_parquet(fpath)
    n_raw = len(raw)

    # Filter concluintes regulares with valid scores
    cond = (
        (raw["TP_ST_CONCLUSAO"] == 2) &
        raw["NU_NOTA_MT"].notna() & (raw["NU_NOTA_MT"] > 0) &
        raw["NU_NOTA_CN"].notna() & (raw["NU_NOTA_CN"] > 0) &
        raw["NU_NOTA_CH"].notna() & (raw["NU_NOTA_CH"] > 0) &
        raw["NU_NOTA_LC"].notna() & (raw["NU_NOTA_LC"] > 0) &
        raw["NU_NOTA_REDACAO"].notna() & (raw["NU_NOTA_REDACAO"] > 0) &
        raw["CO_MUNICIPIO_ESC"].notna() &
        raw["TP_DEPENDENCIA_ADM_ESC"].notna()
    )
    df_y = raw[cond].copy()

    # Harmonize features
    df_y["NU_ANO"] = y
    df_y["NOTA_GERAL"] = (
        df_y["NU_NOTA_MT"] + df_y["NU_NOTA_CN"] +
        df_y["NU_NOTA_CH"] + df_y["NU_NOTA_LC"] +
        df_y["NU_NOTA_REDACAO"]
    ) / 5.0

    df_y["CLUSTER_ID"] = (
        df_y["SG_UF_ESC"].astype(str) + "_" +
        df_y["CO_MUNICIPIO_ESC"].astype(int).astype(str) + "_" +
        df_y["TP_DEPENDENCIA_ADM_ESC"].astype(int).astype(str) + "_" +
        df_y["TP_LOCALIZACAO_ESC"].fillna(1).astype(int).astype(str)
    )

    # Filter clusters with >= 10 students
    c_counts = df_y["CLUSTER_ID"].value_counts()
    valid_c = c_counts[c_counts >= 10].index
    df_y = df_y[df_y["CLUSTER_ID"].isin(valid_c)].copy()

    # Socioeconomic proxy score from demographic variables
    # (Race, School Dep, Location, Age group, Language)
    dep_weights = {1: 1.5, 2: -0.2, 3: -0.4, 4: 1.8}
    raca_weights = {1: 0.8, 2: -0.5, 3: -0.3, 4: 0.5, 5: -0.6, 0: 0.0}
    loc_weights = {1: 0.2, 2: -0.8}
    age_weights = {1: 0.5, 2: 0.4, 3: 0.2, 4: -0.2, 5: -0.5, 6: -0.8}

    df_y["INSE_IND"] = (
        df_y["TP_DEPENDENCIA_ADM_ESC"].map(dep_weights).fillna(0) * 1.5 +
        df_y["TP_COR_RACA"].map(raca_weights).fillna(0) * 0.8 +
        df_y["TP_LOCALIZACAO_ESC"].map(loc_weights).fillna(0) * 0.6 +
        df_y["TP_FAIXA_ETARIA"].map(age_weights).fillna(0) * 0.5 +
        (df_y["TP_LINGUA"] == 0).astype(float) * 0.4
    )
    # Standardize INSE
    df_y["INSE_IND"] = (
        df_y["INSE_IND"] - df_y["INSE_IND"].mean()
    ) / df_y["INSE_IND"].std()

    # School mean INSE & Group-mean centered INSE
    school_inse = df_y.groupby("CLUSTER_ID")["INSE_IND"].transform("mean")
    df_y["INSE_ESCOLA"] = school_inse
    df_y["INSE_CENTR"] = df_y["INSE_IND"] - df_y["INSE_ESCOLA"]

    n_filtered = len(df_y)
    n_clusters = df_y["CLUSTER_ID"].nunique()

    cohort_summary.append({
        "NU_ANO": y,
        "Total_Raw": n_raw,
        "Filtered_Concluintes": n_filtered,
        "Num_Clusters": n_clusters,
        "Mean_Cluster_Size": n_filtered / n_clusters if n_clusters else 0,
        "Mean_MT": df_y["NU_NOTA_MT"].mean(),
        "SD_MT": df_y["NU_NOTA_MT"].std(),
        "Mean_CN": df_y["NU_NOTA_CN"].mean(),
        "SD_CN": df_y["NU_NOTA_CN"].std(),
        "Mean_CH": df_y["NU_NOTA_CH"].mean(),
        "SD_CH": df_y["NU_NOTA_CH"].std(),
        "Mean_LC": df_y["NU_NOTA_LC"].mean(),
        "SD_LC": df_y["NU_NOTA_LC"].std(),
        "Mean_RED": df_y["NU_NOTA_REDACAO"].mean(),
        "SD_RED": df_y["NU_NOTA_REDACAO"].std(),
        "Mean_GERAL": df_y["NOTA_GERAL"].mean(),
        "SD_GERAL": df_y["NOTA_GERAL"].std()
    })
    print(f"Year {y}: {n_filtered:,} concluintes in {n_clusters:,} clusters.")
    dfs.append(df_y)

df_all = pd.concat(dfs, ignore_index=True)
print(f"\nTotal harmonized dataset: {len(df_all):,} records.")

# Save cohort summary table
summary_df = pd.DataFrame(cohort_summary)
summary_df.to_csv(
    os.path.join(OUT_DIR, "tables", "tabela_resumo_cohortes.csv"), index=False
)
with open(
    os.path.join(OUT_DIR, "tables", "tabela_resumo_cohortes.tex"), "w"
) as f:
    f.write(summary_df.to_latex(index=False, float_format="%.2f"))

print("\n=== STEP 2: MODEL 1 - PURE NESTED RANDOM EFFECTS (ANOVA) ===")
# Y_ijk = mu + A_i + B_j(i) + e_ijk
anova_results = []
disciplines_all = ["MT", "CN", "CH", "LC", "RED", "GERAL"]
score_mapping = {
    "MT": "NU_NOTA_MT",
    "CN": "NU_NOTA_CN",
    "CH": "NU_NOTA_CH",
    "LC": "NU_NOTA_LC",
    "RED": "NU_NOTA_REDACAO",
    "GERAL": "NOTA_GERAL"
}

N_total = len(df_all)
I_years = df_all["NU_ANO"].nunique()
df_all["CLUSTER_YEAR"] = (
    df_all["NU_ANO"].astype(str) + "_" + df_all["CLUSTER_ID"]
)
J_clusters = df_all["CLUSTER_YEAR"].nunique()

for disc in disciplines_all:
    col = score_mapping[disc]
    y = df_all[col].values
    y_bar = y.mean()

    # Year means
    year_means = df_all.groupby("NU_ANO")[col].mean()
    year_counts = df_all.groupby("NU_ANO")[col].count()
    SQA = np.sum(year_counts * (year_means - y_bar)**2)
    GL_A = I_years - 1
    QMA = SQA / GL_A

    # School within Year means
    cluster_year_means = df_all.groupby("CLUSTER_YEAR")[col].mean()
    cluster_year_counts = df_all.groupby("CLUSTER_YEAR")[col].count()
    cy_df = pd.DataFrame({
        "mean_ij": cluster_year_means,
        "n_ij": cluster_year_counts
    })
    cy_df["year"] = cy_df.index.str.split("_").str[0].astype(int)
    cy_df["year_mean"] = cy_df["year"].map(year_means)

    SQB_A = np.sum(cy_df["n_ij"] * (cy_df["mean_ij"] - cy_df["year_mean"])**2)
    GL_B_A = J_clusters - I_years
    QMB_A = SQB_A / GL_B_A

    # Total and Residual
    SQT = np.sum((y - y_bar)**2)
    SQE = SQT - SQA - SQB_A
    GL_E = N_total - J_clusters
    QME = SQE / GL_E

    # Henderson unbalance coefficients
    sum_nij2 = (
        df_all.groupby(["NU_ANO", "CLUSTER_ID"])[col]
        .count()
        .pow(2)
        .groupby("NU_ANO")
        .sum()
    )
    n_i_dot = df_all.groupby("NU_ANO")[col].count()
    k1 = (N_total - np.sum(sum_nij2 / n_i_dot)) / GL_B_A

    sum_ni2 = np.sum(n_i_dot**2)
    k3 = (N_total - sum_ni2 / N_total) / GL_A

    # Variance components
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
        "Disciplina": disc,
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
    os.path.join(OUT_DIR, "tables", "tabela_anova_modelo1.csv"), index=False
)
with open(
    os.path.join(OUT_DIR, "tables", "tabela_anova_modelo1.tex"), "w"
) as f:
    f.write(anova_df.to_latex(index=False, float_format="%.3f"))

print("\n=== STEP 3: MODEL 2 - CROSSED LONGITUDINAL (ESCOLA x ANO) ===")
cluster_year_crosstab = pd.crosstab(df_all["CLUSTER_ID"], df_all["NU_ANO"])
stable_clusters = cluster_year_crosstab[
    (cluster_year_crosstab > 0).all(axis=1)
].index
print(
    f"Clusters continuously observed across all 5 years: "
    f"{len(stable_clusters):,}"
)

df_long = df_all[df_all["CLUSTER_ID"].isin(stable_clusters)].copy()
print(f"Longitudinal panel size: {len(df_long):,} students.")

crossed_results = []
for disc in disciplines_all:
    col = score_mapping[disc]
    y = df_long[col].values
    y_bar = y.mean()

    # Cell, school, and year means
    cell_means = (
        df_long.groupby(["CLUSTER_ID", "NU_ANO"])[col].mean().unstack()
    )
    school_means = df_long.groupby("CLUSTER_ID")[col].mean()
    year_means = df_long.groupby("NU_ANO")[col].mean()

    var_school_persistent = school_means.var(ddof=1)
    var_year_main = year_means.var(ddof=1)

    cell_res = (
        cell_means.sub(school_means, axis=0).sub(year_means, axis=1) + y_bar
    )
    var_interaction = (cell_res**2).values.mean()

    cell_std = df_long.groupby(["CLUSTER_ID", "NU_ANO"])[col].std()
    var_within_student = (cell_std**2).mean()

    tot_var = (
        var_year_main + var_school_persistent +
        var_interaction + var_within_student
    )

    crossed_results.append({
        "Disciplina": disc,
        "sigma2_Ano": var_year_main,
        "sigma2_Escola_Persistente": var_school_persistent,
        "sigma2_Interacao_Ano_Escola": var_interaction,
        "sigma2_Aluno_Intra": var_within_student,
        "VPC_Ano(%)": (var_year_main / tot_var) * 100,
        "VPC_Escola_Persistente(%)": (var_school_persistent / tot_var) * 100,
        "VPC_Interacao(%)": (var_interaction / tot_var) * 100,
        "VPC_Aluno(%)": (var_within_student / tot_var) * 100,
        "Persistencia_Escolar(%)": (
            var_school_persistent / (var_school_persistent + var_interaction)
        ) * 100
    })

crossed_df = pd.DataFrame(crossed_results)
print("\nLongitudinal Crossed Effects (Escola x Ano):")
print(crossed_df.to_string(index=False))

crossed_df.to_csv(
    os.path.join(OUT_DIR, "tables", "tabela_cruzado_modelo2.csv"), index=False
)
with open(
    os.path.join(OUT_DIR, "tables", "tabela_cruzado_modelo2.tex"), "w"
) as f:
    f.write(crossed_df.to_latex(index=False, float_format="%.3f"))

print("\n=== STEP 4: MODEL 3 & 4 - MIXED MODELS (REML / HLM) ===")
np.random.seed(42)
sample_size = 200000
sample_idx = df_all.groupby(
    ["NU_ANO", "TP_DEPENDENCIA_ADM_ESC"], group_keys=False
).apply(
    lambda x: x.sample(
        min(len(x), int(sample_size * len(x) / len(df_all))),
        random_state=42
    )
).index
df_sample = df_all.loc[sample_idx].copy()
print(f"Stratified sample for REML: {len(df_sample):,} students.")

reml_results = {}
m3_models = {}
m4_models = {}

for disc in ["MT", "CN", "CH", "LC", "RED", "GERAL"]:
    col = score_mapping[disc]
    print(f"\nFitting REML Mixed Models for {disc}...")

    # Model 0: Null Model
    m0 = smf.mixedlm(
        f"{col} ~ 1", data=df_sample, groups=df_sample["CLUSTER_ID"]
    ).fit(reml=True)
    sigma2_u_0 = m0.cov_re.iloc[0, 0]
    sigma2_e_0 = m0.scale
    icc_0 = sigma2_u_0 / (sigma2_u_0 + sigma2_e_0)

    # Model 3: Fixed Year + Random School
    m3 = smf.mixedlm(
        f"{col} ~ C(NU_ANO, Treatment(reference=2018))",
        data=df_sample,
        groups=df_sample["CLUSTER_ID"]
    ).fit(reml=True)
    m3_models[disc] = m3
    sigma2_u_3 = m3.cov_re.iloc[0, 0]
    sigma2_e_3 = m3.scale

    pred_fixed_3 = m3.predict(df_sample)
    var_fixed_3 = np.var(pred_fixed_3)
    denom_3 = var_fixed_3 + sigma2_u_3 + sigma2_e_3
    r2_marginal_3 = var_fixed_3 / denom_3
    r2_condicional_3 = (var_fixed_3 + sigma2_u_3) / denom_3

    # Model 4: Full HLM
    df_sample["TP_SEXO_BIN"] = (df_sample["TP_SEXO"] == "M").astype(int)
    formula_4 = (
        f"{col} ~ C(NU_ANO, Treatment(reference=2018)) + "
        f"TP_SEXO_BIN + C(TP_COR_RACA, Treatment(reference=1)) + "
        f"INSE_CENTR + INSE_ESCOLA + "
        f"C(TP_DEPENDENCIA_ADM_ESC, Treatment(reference=2)) + "
        f"C(TP_LOCALIZACAO_ESC, Treatment(reference=1))"
    )
    m4 = smf.mixedlm(
        formula_4, data=df_sample, groups=df_sample["CLUSTER_ID"]
    ).fit(reml=True)
    m4_models[disc] = m4
    sigma2_u_4 = m4.cov_re.iloc[0, 0]
    sigma2_e_4 = m4.scale

    pred_fixed_4 = m4.predict(df_sample)
    var_fixed_4 = np.var(pred_fixed_4)
    denom_4 = var_fixed_4 + sigma2_u_4 + sigma2_e_4
    r2_marginal_4 = var_fixed_4 / denom_4
    r2_condicional_4 = (var_fixed_4 + sigma2_u_4) / denom_4

    delta_escola = (sigma2_u_0 - sigma2_u_4) / sigma2_u_0 * 100.0
    delta_aluno = (sigma2_e_0 - sigma2_e_4) / sigma2_e_0 * 100.0

    ols_m = smf.ols(f"{col} ~ C(NU_ANO)", data=df_sample).fit()
    lr_stat = 2 * (m3.llf - ols_m.llf)
    p_lrt = 0.5 * (1.0 - stats.chi2.cdf(max(0, lr_stat), df=1))

    reml_results[disc] = {
        "sigma2_u_Nulo": sigma2_u_0,
        "sigma2_e_Nulo": sigma2_e_0,
        "ICC_Nulo(%)": icc_0 * 100,
        "sigma2_u_M3": sigma2_u_3,
        "sigma2_e_M3": sigma2_e_3,
        "R2_Marginal_M3(%)": r2_marginal_3 * 100,
        "R2_Condicional_M3(%)": r2_condicional_3 * 100,
        "LRT_Stat": lr_stat,
        "p_LRT": p_lrt,
        "sigma2_u_M4(Condicional)": sigma2_u_4,
        "sigma2_e_M4": sigma2_e_4,
        "R2_Marginal_M4(%)": r2_marginal_4 * 100,
        "R2_Condicional_M4(%)": r2_condicional_4 * 100,
        "Reducao_Efeito_Escola_Delta(%)": delta_escola,
        "Reducao_Efeito_Aluno(%)": delta_aluno,
        "Beta_INSE_Individual": m4.fe_params.get("INSE_CENTR", 0.0),
        "Gamma_INSE_Escola": m4.fe_params.get("INSE_ESCOLA", 0.0),
        "Beta_Sexo_M": m4.fe_params.get("TP_SEXO_BIN", 0.0)
    }

reml_df = pd.DataFrame(reml_results).T
print("\nREML & HLM Comprehensive Decomposition:")
print(reml_df.to_string())

reml_df.to_csv(
    os.path.join(OUT_DIR, "tables", "tabela_reml_hlm_modelos3_4.csv")
)
with open(
    os.path.join(OUT_DIR, "tables", "tabela_reml_hlm_modelos3_4.tex"), "w"
) as f:
    f.write(reml_df.to_latex(float_format="%.3f"))

# Extract M4 Coefficients Table for Mathematics
m4_mt = m4_models["MT"]
coef_df_mt = pd.DataFrame({
    "Estimate": m4_mt.params,
    "Std_Err": m4_mt.bse,
    "z_value": m4_mt.tvalues,
    "p_value": m4_mt.pvalues
})
coef_df_mt.to_csv(
    os.path.join(OUT_DIR, "tables", "coeficientes_hlm_matematica.csv")
)
with open(
    os.path.join(OUT_DIR, "tables", "coeficientes_hlm_matematica.tex"), "w"
) as f:
    f.write(coef_df_mt.to_latex(float_format="%.4f"))

print("\n=== STEP 5: BLUP PREDICTION & SHRINKAGE DIAGNOSTICS ===")
blup_dict = m4_mt.random_effects
blup_df = pd.DataFrame.from_dict(blup_dict, orient="index")
blup_df.columns = ["u_hat_BLUP"]
blup_df["CLUSTER_ID"] = blup_df.index

cluster_stats = df_sample.groupby("CLUSTER_ID").agg(
    n_students=("NU_NOTA_MT", "count"),
    raw_mean=("NU_NOTA_MT", "mean"),
    dep_adm=("TP_DEPENDENCIA_ADM_ESC", "first"),
    uf=("SG_UF_ESC", "first"),
    inse_mean=("INSE_ESCOLA", "first")
).reset_index()

blup_merged = pd.merge(cluster_stats, blup_df, on="CLUSTER_ID")
sigma2_u_est = m4_models["MT"].cov_re.iloc[0, 0]
sigma2_e_est = m4_models["MT"].scale
blup_merged["lambda_shrinkage"] = (
    (blup_merged["n_students"] * sigma2_u_est) /
    (blup_merged["n_students"] * sigma2_u_est + sigma2_e_est)
)
blup_merged.to_csv(
    os.path.join(OUT_DIR, "tables", "blup_escolas_matematica.csv"),
    index=False
)

print(f"BLUP extracted for {len(blup_merged):,} school clusters.")
blup_merged["size_bin"] = pd.qcut(
    blup_merged["n_students"], q=5, duplicates="drop"
)
print("Shrinkage summary:")
print(
    blup_merged.groupby("size_bin")["lambda_shrinkage"]
    .agg(["mean", "min", "max", "count"])
)

print("\n=== STEP 6: PUBLICATION-QUALITY VECTOR PLOTS (PDF & PNG) ===")

# Plot 1: Variance Decomposition across Disciplines
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
disciplines = anova_df["Disciplina"].tolist()
x = np.arange(len(disciplines))
width = 0.55

ax.bar(
    x, anova_df["VPC_Ano(%)"], width,
    label="Ano (Interanual)", color="#34495e"
)
ax.bar(
    x, anova_df["VPC_Escola(%)"], width,
    bottom=anova_df["VPC_Ano(%)"],
    label="Escola (Interinstitucional)", color="#e74c3c"
)
ax.bar(
    x, anova_df["VPC_Aluno(%)"], width,
    bottom=anova_df["VPC_Ano(%)"] + anova_df["VPC_Escola(%)"],
    label="Aluno (Intraescolar)", color="#3498db"
)

ax.set_ylabel(
    "Participação na Variância Total (%)", fontsize=12, fontweight="bold"
)
ax.set_title(
    "Decomposição da Variabilidade das Notas do ENEM\n"
    "(Modelo 1: ANOVA Henderson III / Aninhado Puro)",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xticks(x)
ax.set_xticklabels(
    [f"{d}\n({SCORE_COLS.get(d, 'GERAL')})" for d in disciplines],
    fontsize=10
)
ax.legend(loc="upper right", frameon=True, framealpha=0.95)
ax.set_ylim(0, 105)

for i in range(len(disciplines)):
    v_ano = anova_df.loc[i, "VPC_Ano(%)"]
    v_esc = anova_df.loc[i, "VPC_Escola(%)"]
    v_alu = anova_df.loc[i, "VPC_Aluno(%)"]
    ax.text(
        i, v_ano / 2, f"{v_ano:.1f}%",
        ha="center", va="center", color="white", fontsize=9, fontweight="bold"
    )
    ax.text(
        i, v_ano + v_esc / 2, f"{v_esc:.1f}%",
        ha="center", va="center", color="white", fontsize=9, fontweight="bold"
    )
    ax.text(
        i, v_ano + v_esc + v_alu / 2, f"{v_alu:.1f}%",
        ha="center", va="center", color="white", fontsize=9, fontweight="bold"
    )

plt.tight_layout()
fig.savefig(
    os.path.join(OUT_DIR, "figures", "fig1_decomposicao_variancia_areas.pdf"),
    format="pdf"
)
fig.savefig(
    os.path.join(OUT_DIR, "figures", "fig1_decomposicao_variancia_areas.png"),
    format="png", dpi=300
)
plt.close(fig)

# Plot 2: Longitudinal Stability vs Volatility
fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
x = np.arange(len(crossed_df))
ax.bar(
    x - 0.15, crossed_df["VPC_Escola_Persistente(%)"], 0.3,
    label="Efeito Escola Persistente", color="#27ae60"
)
ax.bar(
    x + 0.15, crossed_df["VPC_Interacao(%)"], 0.3,
    label="Volatilidade Anual (Escola × Ano)", color="#f39c12"
)
ax.set_ylabel("% da Variância Total", fontsize=11, fontweight="bold")
ax.set_title(
    "Estabilidade Institucional vs. Volatilidade Anual no ENEM\n"
    "(Modelo 2: Escolas Cruzadas ao Longo de 5 Anos)",
    fontsize=12, fontweight="bold"
)
ax.set_xticks(x)
ax.set_xticklabels(crossed_df["Disciplina"], fontsize=10)
ax.legend(frameon=True)
for i in range(len(crossed_df)):
    ax.text(
        i - 0.15, crossed_df.loc[i, "VPC_Escola_Persistente(%)"] + 0.5,
        f"{crossed_df.loc[i, 'VPC_Escola_Persistente(%)']:.1f}%",
        ha="center", fontsize=8, fontweight="bold"
    )
    ax.text(
        i + 0.15, crossed_df.loc[i, "VPC_Interacao(%)"] + 0.5,
        f"{crossed_df.loc[i, 'VPC_Interacao(%)']:.1f}%",
        ha="center", fontsize=8, fontweight="bold"
    )
plt.tight_layout()
fig.savefig(
    os.path.join(
        OUT_DIR, "figures", "fig2_estabilidade_vs_volatilidade_escolas.pdf"
    ),
    format="pdf"
)
fig.savefig(
    os.path.join(
        OUT_DIR, "figures", "fig2_estabilidade_vs_volatilidade_escolas.png"
    ),
    format="png", dpi=300
)
plt.close(fig)

# Plot 3: Composition Effect (Gross vs Net School Variance)
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
x = np.arange(len(reml_df))
w = 0.35
v_nulo = reml_df["sigma2_u_Nulo"].values
v_cond = reml_df["sigma2_u_M4(Condicional)"].values
reduc = reml_df["Reducao_Efeito_Escola_Delta(%)"].values

ax.bar(
    x - w/2, v_nulo, w,
    label=r"Variância Bruta entre Escolas ($\sigma^2_{u, Nulo}$)",
    color="#e74c3c"
)
ax.bar(
    x + w/2, v_cond, w,
    label=r"Variância Líquida / Valor Agregado ($\sigma^2_{u, Cond}$)",
    color="#2980b9"
)
ax.set_ylabel(
    r"Componente de Variância ($\sigma^2_u$)",
    fontsize=11, fontweight="bold"
)
ax.set_title(
    "Efeito de Composição Socioeconômica na Variância entre Escolas\n"
    r"(Redução Proporcional $\Delta\sigma^2$: Bruto vs. Valor Agregado)",
    fontsize=12, fontweight="bold"
)
ax.set_xticks(x)
ax.set_xticklabels(reml_df.index, fontsize=10)
ax.legend(frameon=True)

for i in range(len(reml_df)):
    ax.text(
        x[i], max(v_nulo[i], v_cond[i]) + 20,
        f"$\\Delta = -{reduc[i]:.1f}\\%$",
        ha="center", fontsize=9, fontweight="bold", color="#2c3e50"
    )

plt.tight_layout()
fig.savefig(
    os.path.join(
        OUT_DIR, "figures", "fig3_efeito_composicao_valor_agregado.pdf"
    ),
    format="pdf"
)
fig.savefig(
    os.path.join(
        OUT_DIR, "figures", "fig3_efeito_composicao_valor_agregado.png"
    ),
    format="png", dpi=300
)
plt.close(fig)

# Plot 4: BLUP Shrinkage Factor vs. School Size
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

ax1.scatter(
    blup_merged["n_students"], blup_merged["lambda_shrinkage"],
    c=blup_merged["dep_adm"], cmap="viridis", alpha=0.6, s=25
)
ax1.set_xscale("log")
ax1.set_xlabel(
    r"Número de Alunos na Escola ($n_j$ - log)", fontsize=11
)
ax1.set_ylabel(
    r"Fator de Contração ($\lambda_j$ - Shrinkage)", fontsize=11
)
ax1.set_title(
    "Fator de Contração Empírico de Bayes (Shrinkage) vs. Porte",
    fontsize=11, fontweight="bold"
)
ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
ax1.grid(True, which="both", ls=":")

ax2.scatter(
    blup_merged["raw_mean"],
    blup_merged["u_hat_BLUP"] + df_sample["NU_NOTA_MT"].mean(),
    c=blup_merged["n_students"], cmap="plasma", alpha=0.5, s=20
)
ax2.plot(
    [300, 800], [300, 800],
    color="black", linestyle="--", label="Identidade (Sem Shrinkage)"
)
ax2.set_xlabel(
    r"Média Amostral Bruta da Escola ($\overline{Y}_{j\cdot}$)",
    fontsize=11
)
ax2.set_ylabel(
    r"Efeito Ajustado BLUP ($\hat{\mu} + \hat{u}_j$)",
    fontsize=11
)
ax2.set_title(
    "Efeito do Shrinkage sobre Rankings (Matemática)",
    fontsize=11, fontweight="bold"
)
ax2.legend(loc="upper left")

plt.tight_layout()
fig.savefig(
    os.path.join(OUT_DIR, "figures", "fig4_blup_shrinkage_diagnostico.pdf"),
    format="pdf"
)
fig.savefig(
    os.path.join(OUT_DIR, "figures", "fig4_blup_shrinkage_diagnostico.png"),
    format="png", dpi=300
)
plt.close(fig)

# Plot 5: Residual Diagnostics (QQ-Plot)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

stats.probplot(m4_mt.resid, dist="norm", plot=ax1)
ax1.set_title(
    r"Q-Q Plot dos Resíduos Individuais ($\epsilon_{ijk}$ - Nível 1)",
    fontsize=11, fontweight="bold"
)
ax1.set_xlabel("Quantis Teóricos Normais")
ax1.set_ylabel("Quantis Amostrais dos Resíduos")

stats.probplot(blup_merged["u_hat_BLUP"], dist="norm", plot=ax2)
ax2.set_title(
    r"Q-Q Plot dos Efeitos de Escola ($\hat{u}_j$ - Nível 2)",
    fontsize=11, fontweight="bold"
)
ax2.set_xlabel("Quantis Teóricos Normais")
ax2.set_ylabel("Quantis Amostrais dos BLUPs")

plt.tight_layout()
fig.savefig(
    os.path.join(
        OUT_DIR, "figures", "fig5_diagnostico_residuos_qqplot.pdf"
    ),
    format="pdf"
)
fig.savefig(
    os.path.join(
        OUT_DIR, "figures", "fig5_diagnostico_residuos_qqplot.png"
    ),
    format="png", dpi=300
)
plt.close(fig)

# Plot 6: Evolution by Dependency
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
dep_labels = {1: "Federal", 2: "Estadual", 3: "Municipal", 4: "Privada"}
for dep, label in dep_labels.items():
    dep_data = (
        df_all[df_all["TP_DEPENDENCIA_ADM_ESC"] == dep]
        .groupby("NU_ANO")["NOTA_GERAL"]
        .mean()
    )
    ax.plot(
        dep_data.index, dep_data.values,
        marker="o", linewidth=2.2, label=label
    )

ax.set_title(
    "Evolução Temporal da Média Geral do ENEM por Rede (2018–2022)",
    fontsize=12, fontweight="bold"
)
ax.set_xlabel("Ano da Edição do ENEM", fontsize=11)
ax.set_ylabel("Nota Média Geral (TRI / Escala 0–1000)", fontsize=11)
ax.set_xticks(YEARS)
ax.legend(title="Rede de Ensino", frameon=True)
plt.tight_layout()
fig.savefig(
    os.path.join(OUT_DIR, "figures", "fig6_evolucao_temporal_redes.pdf"),
    format="pdf"
)
fig.savefig(
    os.path.join(OUT_DIR, "figures", "fig6_evolucao_temporal_redes.png"),
    format="png", dpi=300
)
plt.close(fig)

print("\n=== SCRIPT COMPLETED SUCCESSFULLY! ALL OUTPUTS GENERATED ===")
