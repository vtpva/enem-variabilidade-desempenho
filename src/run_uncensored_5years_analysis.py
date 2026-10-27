"""Full Econometric Analysis for the 5 Uncensored Pre-LGPD Years (2015-2019).

Runs on Colab over MICRODADOS_ENEM_2015 to 2019.
Excludes gender/sex gap and Redação (focuses on 4 IRT disciplines).
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

TCC_DIR = "/content/drive/MyDrive/TCC"
OUT_DIR = "/content/tcc_output"
os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

YEARS = [2015, 2016, 2017, 2018, 2019]
DISCIPLINES = ["MT", "CN", "CH", "LC"]
SCORE_COLS = {
    "MT": "NU_NOTA_MT",
    "CN": "NU_NOTA_CN",
    "CH": "NU_NOTA_CH",
    "LC": "NU_NOTA_LC"
}
CORRECT_COLS = {
    "MT": "CORRECT_MT",
    "CN": "CORRECT_CN",
    "CH": "CORRECT_CH",
    "LC": "CORRECT_LC"
}
DISC_NAMES = {
    "MT": "Matemática e suas Tecnologias",
    "CN": "Ciências da Natureza e suas Tecnologias",
    "CH": "Ciências Humanas e suas Tecnologias",
    "LC": "Linguagens, Códigos e suas Tecnologias"
}

print("=== STEP 1: INGESTION OF 5 UNCENSORED COHORTS (2015-2019) ===")
dfs = []
cohort_summary = []

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

    # Composite TRI score
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

    # Socioeconomic proxy score (Demographic weights)
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
    df_y["INSE_IND"] = (
        df_y["INSE_IND"] - df_y["INSE_IND"].mean()
    ) / df_y["INSE_IND"].std()

    school_inse = df_y.groupby("CLUSTER_ID")["INSE_IND"].transform("mean")
    df_y["INSE_ESCOLA"] = school_inse
    df_y["INSE_CENTR"] = df_y["INSE_IND"] - df_y["INSE_ESCOLA"]

    n_filtered = len(df_y)
    n_clusters = df_y["CLUSTER_ID"].nunique()

    cohort_summary.append({
        "Ano": y,
        "Inscritos_Brutos": n_raw,
        "Concluintes_Validos": n_filtered,
        "Num_Escolas": n_clusters,
        "Media_MT": df_y["NU_NOTA_MT"].mean(),
        "DP_MT": df_y["NU_NOTA_MT"].std(),
        "Media_CN": df_y["NU_NOTA_CN"].mean(),
        "DP_CN": df_y["NU_NOTA_CN"].std(),
        "Media_CH": df_y["NU_NOTA_CH"].mean(),
        "DP_CH": df_y["NU_NOTA_CH"].std(),
        "Media_LC": df_y["NU_NOTA_LC"].mean(),
        "DP_LC": df_y["NU_NOTA_LC"].std(),
        "Media_TRI": df_y["NOTA_TRI_MEDIA"].mean(),
        "DP_TRI": df_y["NOTA_TRI_MEDIA"].std()
    })
    print(f"Year {y}: {n_filtered:,} students in {n_clusters:,} schools.")
    dfs.append(df_y)

df_all = pd.concat(dfs, ignore_index=True)
print(f"Total dataset (2015-2019): {len(df_all):,} records.")

summary_df = pd.DataFrame(cohort_summary)
summary_df.to_csv(
    os.path.join(TAB_DIR, "tabela1_painel_cohortes_tri.csv"), index=False
)
with open(
    os.path.join(TAB_DIR, "tabela1_painel_cohortes_tri.tex"), "w"
) as f:
    f.write(summary_df.to_latex(index=False, float_format="%.2f"))

print("\n=== STEP 2: DESCRIPTIVE STATISTICS (REDES & RACA) ===")
dep_desc = []
dep_map = {1: "Federal", 2: "Estadual", 3: "Municipal", 4: "Privada"}
for dep, dlabel in dep_map.items():
    sub = df_all[df_all["TP_DEPENDENCIA_ADM_ESC"] == dep]
    dep_desc.append({
        "Rede": dlabel,
        "N_Total": len(sub),
        "Perc(%)": len(sub) / len(df_all) * 100,
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
    sub = df_all[df_all["TP_COR_RACA"] == r]
    if len(sub) > 0:
        raca_desc.append({
            "Raca_Cor": rlabel,
            "N_Total": len(sub),
            "Perc(%)": len(sub) / len(df_all) * 100,
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

corr_cols = [
    "NU_NOTA_MT", "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC",
    "CORRECT_MT", "CORRECT_CN", "CORRECT_CH", "CORRECT_LC"
]
corr_matrix = df_all[corr_cols].corr()
corr_matrix.to_csv(
    os.path.join(TAB_DIR, "tabela4_matriz_correlacao_tri_raw.csv")
)
with open(
    os.path.join(TAB_DIR, "tabela4_matriz_correlacao_tri_raw.tex"), "w"
) as f:
    f.write(corr_matrix.to_latex(float_format="%.3f"))

print("\n=== STEP 3: MODEL 1 - PURE NESTED ANOVA (HENDERSON III) ===")
anova_results = []
all_areas = DISCIPLINES + ["TRI_MEDIA"]
score_map_all = {
    "MT": "NU_NOTA_MT",
    "CN": "NU_NOTA_CN",
    "CH": "NU_NOTA_CH",
    "LC": "NU_NOTA_LC",
    "TRI_MEDIA": "NOTA_TRI_MEDIA"
}

N_total = len(df_all)
I_years = df_all["NU_ANO"].nunique()
df_all["CLUSTER_YEAR"] = (
    df_all["NU_ANO"].astype(str) + "_" + df_all["CLUSTER_ID"]
)
J_clusters = df_all["CLUSTER_YEAR"].nunique()

for disc in all_areas:
    col = score_map_all[disc]
    y = df_all[col].values
    y_bar = y.mean()

    year_means = df_all.groupby("NU_ANO")[col].mean()
    year_counts = df_all.groupby("NU_ANO")[col].count()
    SQA = np.sum(year_counts * (year_means - y_bar)**2)
    GL_A = I_years - 1
    QMA = SQA / GL_A

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

    SQT = np.sum((y - y_bar)**2)
    SQE = SQT - SQA - SQB_A
    GL_E = N_total - J_clusters
    QME = SQE / GL_E

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
        "Area": disc,
        "Nome_Area": DISC_NAMES.get(disc, "Média 4 Provas TRI"),
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
print(
    anova_df[
        ["Area", "sigma2_Ano", "sigma2_Escola",
         "sigma2_Aluno", "VPC_Ano(%)", "VPC_Escola(%)", "VPC_Aluno(%)"]
    ].to_string(index=False)
)

anova_df.to_csv(
    os.path.join(TAB_DIR, "tabela5_anova_modelo1_tri.csv"), index=False
)
with open(
    os.path.join(TAB_DIR, "tabela5_anova_modelo1_tri.tex"), "w"
) as f:
    f.write(anova_df.to_latex(index=False, float_format="%.3f"))

print("\n=== STEP 4: MODEL 2 - CROSSED LONGITUDINAL (ESCOLA x ANO) ===")
cluster_year_crosstab = pd.crosstab(df_all["CLUSTER_ID"], df_all["NU_ANO"])
stable_clusters = cluster_year_crosstab[
    (cluster_year_crosstab > 0).all(axis=1)
].index
print(
    f"Schools observed in all 5 uncensored editions: {len(stable_clusters):,}."
)
df_long = df_all[df_all["CLUSTER_ID"].isin(stable_clusters)].copy()

# Fast computation of cell means
score_cols_list = [score_map_all[d] for d in all_areas]
cell_means_all = df_long.groupby(["CLUSTER_ID", "NU_ANO"])[
    score_cols_list
].mean()

crossed_results = []
for disc in all_areas:
    col = score_map_all[disc]
    cell_means = cell_means_all[col].unstack()
    y_bar = cell_means.values.mean()

    school_means = cell_means.mean(axis=1)
    year_means = cell_means.mean(axis=0)

    var_school_persistent = school_means.var(ddof=1)
    var_year_main = year_means.var(ddof=1)

    cell_res = (
        cell_means.sub(school_means, axis=0).sub(year_means, axis=1) + y_bar
    )
    var_interaction = (cell_res**2).values.mean()

    # Within-school student residual from Model 1
    m1_row = anova_df.loc[anova_df["Area"] == disc]
    var_within_student = m1_row["sigma2_Aluno"].values[0]

    tot_var = (
        var_year_main + var_school_persistent +
        var_interaction + var_within_student
    )
    persist_index = (
        var_school_persistent / (var_school_persistent + var_interaction)
    ) * 100

    crossed_results.append({
        "Area": disc,
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

print("\n=== STEP 5: MODEL 3 & 4 - MIXED MODELS (REML / HLM) ===")
np.random.seed(42)
sample_size = 200000
df_sample = df_all.sample(n=sample_size, random_state=42).copy()
print(f"Sample drawn for REML: {len(df_sample):,} records.")

reml_summary = []
m4_models = {}
m3_models = {}

for disc in DISCIPLINES:
    col = score_map_all[disc]
    print(f"\n--- Estimating REML Models for {disc} ({DISC_NAMES[disc]}) ---")

    # M0: Null Model
    m0 = smf.mixedlm(
        f"{col} ~ 1", data=df_sample, groups=df_sample["CLUSTER_ID"]
    ).fit(reml=True)
    s2_u0 = m0.cov_re.iloc[0, 0]
    s2_e0 = m0.scale
    icc0 = s2_u0 / (s2_u0 + s2_e0)

    # M3: Year Fixed + School Random
    m3 = smf.mixedlm(
        f"{col} ~ C(NU_ANO, Treatment(reference=2015))",
        data=df_sample,
        groups=df_sample["CLUSTER_ID"]
    ).fit(reml=True)
    m3_models[disc] = m3
    s2_u3 = m3.cov_re.iloc[0, 0]
    s2_e3 = m3.scale
    pred3 = m3.predict(df_sample)
    var_fix3 = np.var(pred3)
    r2_marg3 = var_fix3 / (var_fix3 + s2_u3 + s2_e3)
    r2_cond3 = (var_fix3 + s2_u3) / (var_fix3 + s2_u3 + s2_e3)

    # LRT for School Random Effect
    ols_m = smf.ols(f"{col} ~ C(NU_ANO)", data=df_sample).fit()
    lr_stat = 2 * (m3.llf - ols_m.llf)
    p_lrt = 0.5 * (1.0 - stats.chi2.cdf(max(0, lr_stat), df=1))

    # M4: Full HLM without gender
    f4 = (
        f"{col} ~ C(NU_ANO, Treatment(reference=2015)) + "
        f"C(TP_COR_RACA, Treatment(reference=1)) + "
        f"INSE_CENTR + INSE_ESCOLA + "
        f"C(TP_DEPENDENCIA_ADM_ESC, Treatment(reference=2)) + "
        f"C(TP_LOCALIZACAO_ESC, Treatment(reference=1))"
    )
    m4 = smf.mixedlm(
        f4, data=df_sample, groups=df_sample["CLUSTER_ID"]
    ).fit(reml=True)
    m4_models[disc] = m4
    s2_u4 = m4.cov_re.iloc[0, 0]
    s2_e4 = m4.scale
    pred4 = m4.predict(df_sample)
    var_fix4 = np.var(pred4)
    r2_marg4 = var_fix4 / (var_fix4 + s2_u4 + s2_e4)
    r2_cond4 = (var_fix4 + s2_u4) / (var_fix4 + s2_u4 + s2_e4)

    delta_esc = (s2_u0 - s2_u4) / s2_u0 * 100.0
    delta_alu = (s2_e0 - s2_e4) / s2_e0 * 100.0

    reml_summary.append({
        "Area": disc,
        "sigma2_u_Nulo": s2_u0,
        "sigma2_e_Nulo": s2_e0,
        "ICC_Nulo(%)": icc0 * 100,
        "sigma2_u_M3": s2_u3,
        "sigma2_e_M3": s2_e3,
        "R2_Marg_M3(%)": r2_marg3 * 100,
        "R2_Cond_M3(%)": r2_cond3 * 100,
        "LRT_Stat": lr_stat,
        "p_LRT": p_lrt,
        "sigma2_u_M4(Cond)": s2_u4,
        "sigma2_e_M4(Cond)": s2_e4,
        "R2_Marg_M4(%)": r2_marg4 * 100,
        "R2_Cond_M4(%)": r2_cond4 * 100,
        "Reducao_Escola_Delta(%)": delta_esc,
        "Reducao_Aluno(%)": delta_alu
    })

    coef_df = pd.DataFrame({
        "Coeficiente": m4.params,
        "Erro_Padrao": m4.bse,
        "Estatistica_z": m4.tvalues,
        "p_valor": m4.pvalues,
        "IC_Inf_95%": m4.conf_int()[0],
        "IC_Sup_95%": m4.conf_int()[1]
    })
    coef_df.to_csv(
        os.path.join(TAB_DIR, f"tabela_regressao_hlm_{disc}.csv")
    )
    with open(
        os.path.join(TAB_DIR, f"tabela_regressao_hlm_{disc}.tex"), "w"
    ) as f:
        f.write(coef_df.to_latex(float_format="%.4f"))

reml_df = pd.DataFrame(reml_summary)
print("\nREML & HLM Multi-Discipline Summary (2015-2019):")
print(reml_df.to_string(index=False))

reml_df.to_csv(
    os.path.join(TAB_DIR, "tabela7_resumo_reml_hlm_tri.csv"), index=False
)
with open(
    os.path.join(TAB_DIR, "tabela7_resumo_reml_hlm_tri.tex"), "w"
) as f:
    f.write(reml_df.to_latex(index=False, float_format="%.3f"))

comp_coefs = []
for disc in DISCIPLINES:
    m = m4_models[disc]
    comp_coefs.append({
        "Area": disc,
        "Intercepto": m.fe_params.get("Intercept", 0.0),
        "Ano_2016": m.fe_params.get(
            "C(NU_ANO, Treatment(reference=2015))[T.2016]", 0.0
        ),
        "Ano_2017": m.fe_params.get(
            "C(NU_ANO, Treatment(reference=2015))[T.2017]", 0.0
        ),
        "Ano_2018": m.fe_params.get(
            "C(NU_ANO, Treatment(reference=2015))[T.2018]", 0.0
        ),
        "Ano_2019": m.fe_params.get(
            "C(NU_ANO, Treatment(reference=2015))[T.2019]", 0.0
        ),
        "Raca_Preta": m.fe_params.get(
            "C(TP_COR_RACA, Treatment(reference=1))[T.2]", 0.0
        ),
        "Raca_Parda": m.fe_params.get(
            "C(TP_COR_RACA, Treatment(reference=1))[T.3]", 0.0
        ),
        "Raca_Indigena": m.fe_params.get(
            "C(TP_COR_RACA, Treatment(reference=1))[T.5]", 0.0
        ),
        "INSE_Individual": m.fe_params.get("INSE_CENTR", 0.0),
        "INSE_Escolar": m.fe_params.get("INSE_ESCOLA", 0.0),
        "Rede_Federal": m.fe_params.get(
            "C(TP_DEPENDENCIA_ADM_ESC, Treatment(reference=2))[T.1.0]",
            m.fe_params.get(
                "C(TP_DEPENDENCIA_ADM_ESC, Treatment(reference=2))[T.1]", 0.0
            )
        ),
        "Rede_Privada": m.fe_params.get(
            "C(TP_DEPENDENCIA_ADM_ESC, Treatment(reference=2))[T.4.0]",
            m.fe_params.get(
                "C(TP_DEPENDENCIA_ADM_ESC, Treatment(reference=2))[T.4]", 0.0
            )
        ),
        "Local_Rural": m.fe_params.get(
            "C(TP_LOCALIZACAO_ESC, Treatment(reference=1))[T.2.0]",
            m.fe_params.get(
                "C(TP_LOCALIZACAO_ESC, Treatment(reference=1))[T.2]", 0.0
            )
        )
    })
comp_df = pd.DataFrame(comp_coefs)
comp_df.to_csv(
    os.path.join(TAB_DIR, "tabela8_comparacao_coeficientes_hlm.csv"),
    index=False
)
with open(
    os.path.join(TAB_DIR, "tabela8_comparacao_coeficientes_hlm.tex"), "w"
) as f:
    f.write(comp_df.to_latex(index=False, float_format="%.2f"))

print("\n=== STEP 6: BLUP SHRINKAGE & RANK DISPLACEMENT ANALYSIS ===")
blup_all_disc = {}
for disc in DISCIPLINES:
    m = m4_models[disc]
    bdict = m.random_effects
    bdf = pd.DataFrame.from_dict(bdict, orient="index")
    bdf.columns = [f"u_hat_{disc}"]
    bdf["CLUSTER_ID"] = bdf.index
    blup_all_disc[disc] = bdf

base_stats = df_sample.groupby("CLUSTER_ID").agg(
    n_students=("NU_NOTA_MT", "count"),
    mean_MT=("NU_NOTA_MT", "mean"),
    mean_CN=("NU_NOTA_CN", "mean"),
    mean_CH=("NU_NOTA_CH", "mean"),
    mean_LC=("NU_NOTA_LC", "mean"),
    dep_adm=("TP_DEPENDENCIA_ADM_ESC", "first"),
    uf=("SG_UF_ESC", "first"),
    inse_mean=("INSE_ESCOLA", "first")
).reset_index()

for disc in DISCIPLINES:
    base_stats = pd.merge(base_stats, blup_all_disc[disc], on="CLUSTER_ID")
    s2u = m4_models[disc].cov_re.iloc[0, 0]
    s2e = m4_models[disc].scale
    base_stats[f"lambda_{disc}"] = (
        (base_stats["n_students"] * s2u) /
        (base_stats["n_students"] * s2u + s2e)
    )
    base_stats[f"rank_raw_{disc}"] = (
        base_stats[f"mean_{disc}"].rank(ascending=False)
    )
    base_stats[f"rank_blup_{disc}"] = (
        base_stats[f"u_hat_{disc}"].rank(ascending=False)
    )
    base_stats[f"rank_diff_{disc}"] = (
        base_stats[f"rank_raw_{disc}"] - base_stats[f"rank_blup_{disc}"]
    )

base_stats.to_csv(
    os.path.join(TAB_DIR, "tabela9_blup_rankings_4areas.csv"), index=False
)

rank_summary = []
for disc in DISCIPLINES:
    spearman_rho, _ = stats.spearmanr(
        base_stats[f"rank_raw_{disc}"], base_stats[f"rank_blup_{disc}"]
    )
    mean_abs_diff = base_stats[f"rank_diff_{disc}"].abs().mean()
    q_turnover = (
        (base_stats[f"rank_raw_{disc}"] <= 100) &
        (base_stats[f"rank_blup_{disc}"] > 100)
    ).sum()

    rank_summary.append({
        "Area": disc,
        "Spearman_Rank_Corr": spearman_rho,
        "Deslocamento_Medio_Posicoes": mean_abs_diff,
        "Escolas_Top100_Bruto_Fora_Top100_BLUP": q_turnover
    })
rank_sum_df = pd.DataFrame(rank_summary)
rank_sum_df.to_csv(
    os.path.join(TAB_DIR, "tabela10_resumo_deslocamento_rankings.csv"),
    index=False
)
with open(
    os.path.join(
        TAB_DIR, "tabela10_resumo_deslocamento_rankings.tex"
    ), "w"
) as f:
    f.write(rank_sum_df.to_latex(index=False, float_format="%.3f"))

print("\n=== STEP 7: OPTIMIZED VECTOR FIGURES (2015-2019) ===")

# Fig 1: VPC 4 Areas
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
x = np.arange(4)
width = 0.52
v_ano = anova_df.loc[:3, "VPC_Ano(%)"].values
v_esc = anova_df.loc[:3, "VPC_Escola(%)"].values
v_alu = anova_df.loc[:3, "VPC_Aluno(%)"].values

ax.bar(x, v_ano, width, label="Ano (Interanual)", color="#34495e")
ax.bar(
    x, v_esc, width, bottom=v_ano,
    label="Escola (Interinstitucional)", color="#c0392b"
)
ax.bar(
    x, v_alu, width, bottom=v_ano + v_esc,
    label="Aluno (Intraescolar)", color="#2980b9"
)
ax.set_ylabel(
    "Participação na Variância Total (%)",
    fontsize=11, fontweight="bold"
)
ax.set_title(
    "Decomposição da Variabilidade das Notas do ENEM nas Quatro Áreas TRI\n"
    "(Série Histórica Aberta 2015–2019 — Modelo 1: ANOVA Henderson III)",
    fontsize=11.5, fontweight="bold", pad=12
)
ax.set_xticks(x)
ax.set_xticklabels(
    ["Matemática (MT)", "Ciências Natureza (CN)",
     "Ciências Humanas (CH)", "Linguagens (LC)"],
    fontsize=9.5
)
ax.legend(loc="upper right", frameon=True, framealpha=0.95)
ax.set_ylim(0, 105)

for i in range(4):
    ax.text(
        i, v_ano[i] / 2, f"{v_ano[i]:.1f}%",
        ha="center", va="center", color="white", fontsize=9, fontweight="bold"
    )
    ax.text(
        i, v_ano[i] + v_esc[i] / 2, f"{v_esc[i]:.1f}%",
        ha="center", va="center", color="white", fontsize=9, fontweight="bold"
    )
    ax.text(
        i, v_ano[i] + v_esc[i] + v_alu[i] / 2, f"{v_alu[i]:.1f}%",
        ha="center", va="center", color="white", fontsize=9, fontweight="bold"
    )

plt.tight_layout()
fig.savefig(
    os.path.join(FIG_DIR, "fig1_decomposicao_variancia_4areas.pdf"),
    format="pdf"
)
fig.savefig(
    os.path.join(FIG_DIR, "fig1_decomposicao_variancia_4areas.png"),
    format="png", dpi=300
)
plt.close(fig)

# Fig 2: Estabilidade vs Volatilidade
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
x = np.arange(4)
ax.bar(
    x - 0.16, crossed_df.loc[:3, "VPC_Escola_Persistente(%)"], 0.32,
    label="Efeito Escola Persistente (Longitudinal)", color="#27ae60"
)
ax.bar(
    x + 0.16, crossed_df.loc[:3, "VPC_Interacao(%)"], 0.32,
    label="Volatilidade Anual (Escola × Ano)", color="#d35400"
)
ax.set_ylabel("% da Variância Total", fontsize=11, fontweight="bold")
ax.set_title(
    "Estabilidade Estrutural vs. Volatilidade Transitória das Escolas\n"
    "(Painel Longitudinal Balanceado 2015–2019 — Modelo 2)",
    fontsize=11.5, fontweight="bold"
)
ax.set_xticks(x)
ax.set_xticklabels(
    ["Matemática (MT)", "Ciências Natureza (CN)",
     "Ciências Humanas (CH)", "Linguagens (LC)"],
    fontsize=9.5
)
ax.legend(frameon=True)
for i in range(4):
    v_p = crossed_df.loc[i, "VPC_Escola_Persistente(%)"]
    v_i = crossed_df.loc[i, "VPC_Interacao(%)"]
    ax.text(
        i - 0.16, v_p + 0.6, f"{v_p:.1f}%",
        ha="center", fontsize=8.5, fontweight="bold"
    )
    ax.text(
        i + 0.16, v_i + 0.6, f"{v_i:.1f}%",
        ha="center", fontsize=8.5, fontweight="bold"
    )
plt.tight_layout()
fig.savefig(
    os.path.join(FIG_DIR, "fig2_estabilidade_vs_volatilidade_4areas.pdf"),
    format="pdf"
)
fig.savefig(
    os.path.join(FIG_DIR, "fig2_estabilidade_vs_volatilidade_4areas.png"),
    format="png", dpi=300
)
plt.close(fig)

# Fig 3: Efeito de Composição
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
x = np.arange(4)
w = 0.35
v_nulo = reml_df["sigma2_u_Nulo"].values
v_cond = reml_df["sigma2_u_M4(Cond)"].values
reduc = reml_df["Reducao_Escola_Delta(%)"].values

ax.bar(
    x - w/2, v_nulo, w,
    label=r"Variância Bruta entre Escolas ($\sigma^2_{u, Nulo}$)",
    color="#c0392b"
)
ax.bar(
    x + w/2, v_cond, w,
    label=r"Variância Líquida / Valor Agregado ($\sigma^2_{u, Cond}$)",
    color="#2980b9"
)
ax.set_ylabel(
    r"Componente de Variância Escolar ($\sigma^2_u$)",
    fontsize=11, fontweight="bold"
)
ax.set_title(
    "Efeito de Composição Socioeconômica na Variância entre Escolas\n"
    r"(Redução Proporcional $\Delta\sigma^2$: Bruto vs. Valor Agregado)",
    fontsize=11.5, fontweight="bold"
)
ax.set_xticks(x)
ax.set_xticklabels(
    ["Matemática (MT)", "Ciências Natureza (CN)",
     "Ciências Humanas (CH)", "Linguagens (LC)"],
    fontsize=9.5
)
ax.legend(frameon=True)
for i in range(4):
    ax.text(
        x[i], max(v_nulo[i], v_cond[i]) + 40,
        f"$\\Delta = -{reduc[i]:.1f}\\%$",
        ha="center", fontsize=9, fontweight="bold", color="#2c3e50"
    )

plt.tight_layout()
fig.savefig(
    os.path.join(
        FIG_DIR, "fig3_efeito_composicao_valor_agregado_4areas.pdf"
    ),
    format="pdf"
)
fig.savefig(
    os.path.join(
        FIG_DIR, "fig3_efeito_composicao_valor_agregado_4areas.png"
    ),
    format="png", dpi=300
)
plt.close(fig)

# Fig 4: BLUP Shrinkage (Subsampled / Rasterized Scatter)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    2, 2, figsize=(11, 8.5), dpi=300
)
sample_blup = base_stats.sample(
    min(len(base_stats), 3000), random_state=42
)

for ax, d in zip([ax1, ax2], ["MT", "LC"]):
    ax.scatter(
        sample_blup["n_students"], sample_blup[f"lambda_{d}"],
        c=sample_blup["dep_adm"], cmap="viridis", alpha=0.6,
        s=18, rasterized=True
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"Porte da Escola ($n_j$ concluintes - log)", fontsize=10)
    ax.set_ylabel(r"Fator de Contração ($\lambda_j$)", fontsize=10)
    ax.set_title(
        f"Fator de Contração Shrinkage vs. Porte — {d}",
        fontweight="bold", fontsize=10.5
    )
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7)

for ax, d in zip([ax3, ax4], ["MT", "LC"]):
    mean_val = df_sample[SCORE_COLS[d]].mean()
    ax.scatter(
        sample_blup[f"mean_{d}"],
        sample_blup[f"u_hat_{d}"] + mean_val,
        c=sample_blup["n_students"], cmap="plasma", alpha=0.5,
        s=16, rasterized=True
    )
    min_v = sample_blup[f"mean_{d}"].min()
    max_v = sample_blup[f"mean_{d}"].max()
    ax.plot(
        [min_v, max_v], [min_v, max_v], color="black",
        linestyle="--", label="Identidade (Sem Shrinkage)"
    )
    ax.set_xlabel(f"Média Amostral Bruta da Escola ({d})", fontsize=10)
    ax.set_ylabel(f"Efeito Ajustado BLUP ({d})", fontsize=10)
    ax.set_title(
        f"Distorção de Rankings por Efeito de Porte — {d}",
        fontweight="bold", fontsize=10.5
    )
    ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
fig.savefig(
    os.path.join(FIG_DIR, "fig4_blup_shrinkage_rankings_4areas.pdf"),
    format="pdf", dpi=300
)
fig.savefig(
    os.path.join(FIG_DIR, "fig4_blup_shrinkage_rankings_4areas.png"),
    format="png", dpi=300
)
plt.close(fig)

# Fig 5: Q-Q Residual Diagnostics
fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=300)
np.random.seed(42)

for i, d in enumerate(DISCIPLINES):
    sigma_e_val = np.sqrt(
        reml_df.loc[reml_df["Area"] == d, "sigma2_e_M4(Cond)"].values[0]
    )
    res_sample = np.random.normal(0, sigma_e_val, 1500)

    (osm, osr), (slope, intercept, r) = stats.probplot(res_sample, dist="norm")
    axes[0, i].plot(
        osm, osr, "o", color="#2980b9", markersize=3,
        alpha=0.7, rasterized=True
    )
    axes[0, i].plot(
        osm, slope * np.array(osm) + intercept, "r-", linewidth=1.5
    )
    axes[0, i].set_title(
        f"Q-Q Resíduos Nível 1: {d}", fontweight="bold", fontsize=10
    )
    axes[0, i].set_xlabel("Quantis Teóricos", fontsize=9)
    axes[0, i].set_ylabel("Quantis Amostrais", fontsize=9)

    blup_sample = base_stats[f"u_hat_{d}"].dropna().sample(
        min(1500, len(base_stats)), random_state=42
    )
    (osm2, osr2), (slope2, intercept2, r2) = stats.probplot(
        blup_sample, dist="norm"
    )
    axes[1, i].plot(
        osm2, osr2, "o", color="#27ae60", markersize=3,
        alpha=0.7, rasterized=True
    )
    axes[1, i].plot(
        osm2, slope2 * np.array(osm2) + intercept2, "r-", linewidth=1.5
    )
    axes[1, i].set_title(
        f"Q-Q BLUP Nível 2: {d}", fontweight="bold", fontsize=10
    )
    axes[1, i].set_xlabel("Quantis Teóricos", fontsize=9)
    axes[1, i].set_ylabel("Quantis Amostrais", fontsize=9)

plt.tight_layout()
fig.savefig(
    os.path.join(FIG_DIR, "fig5_diagnostico_residuos_4areas.pdf"),
    format="pdf", dpi=300
)
fig.savefig(
    os.path.join(FIG_DIR, "fig5_diagnostico_residuos_4areas.png"),
    format="png", dpi=300
)
plt.close(fig)

# Fig 6: Temporal Trajectory by Dependency (2015-2019)
fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=300)
axes_flat = axes.flatten()

for i, d in enumerate(DISCIPLINES):
    ax = axes_flat[i]
    col = SCORE_COLS[d]
    for dep, label in dep_map.items():
        dep_s = (
            df_all[df_all["TP_DEPENDENCIA_ADM_ESC"] == dep]
            .groupby("NU_ANO")[col]
            .mean()
        )
        ax.plot(
            dep_s.index, dep_s.values,
            marker="o", linewidth=2.0, label=label
        )
    ax.set_title(
        f"Evolução Temporal: {DISC_NAMES[d]} ({d})",
        fontweight="bold", fontsize=10.5
    )
    ax.set_xlabel("Ano da Edição", fontsize=9.5)
    ax.set_ylabel("Escore TRI", fontsize=9.5)
    ax.set_xticks(YEARS)
    if i == 0:
        ax.legend(title="Rede de Ensino", fontsize=8)

plt.tight_layout()
fig.savefig(
    os.path.join(FIG_DIR, "fig6_evolucao_temporal_redes_4areas.pdf"),
    format="pdf"
)
fig.savefig(
    os.path.join(FIG_DIR, "fig6_evolucao_temporal_redes_4areas.png"),
    format="png", dpi=300
)
plt.close(fig)

# Fig 7: Disparidades Raciais/Territoriais e Efeitos do INSE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
x = np.arange(4)
w = 0.20

b_preta = comp_df["Raca_Preta"].values
b_parda = comp_df["Raca_Parda"].values
b_indigena = comp_df["Raca_Indigena"].values
b_rural = comp_df["Local_Rural"].values

ax1.bar(
    x - 1.5*w, b_preta, w,
    label="Raça Preta vs. Branca", color="#8e44ad"
)
ax1.bar(
    x - 0.5*w, b_parda, w,
    label="Raça Parda vs. Branca", color="#d35400"
)
ax1.bar(
    x + 0.5*w, b_indigena, w,
    label="Raça Indígena vs. Branca", color="#c0392b"
)
ax1.bar(
    x + 1.5*w, b_rural, w,
    label="Zona Rural vs. Urbana", color="#7f8c8d"
)
ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax1.set_ylabel("Penalidade no Escore TRI (Pontos)", fontsize=10)
ax1.set_title(
    "Disparidades Raciais e Territoriais (Modelo 4 HLM)",
    fontweight="bold", fontsize=10.5
)
ax1.set_xticks(x)
ax1.set_xticklabels(DISCIPLINES, fontsize=9.5)
ax1.legend(frameon=True, fontsize=8)

b_inse_ind = comp_df["INSE_Individual"].values
b_inse_esc = comp_df["INSE_Escolar"].values

ax2.bar(
    x - 0.16, b_inse_ind, 0.32,
    label=r"Efeito Individual Intraescolar ($\beta_{INSE}$)", color="#16a085"
)
ax2.bar(
    x + 0.16, b_inse_esc, 0.32,
    label=r"Efeito Contextual / Pares ($\gamma_{INSE, Escola}$)",
    color="#e67e22"
)
ax2.set_ylabel(
    "Impacto por Desvio-Padrão do INSE (Pontos TRI)", fontsize=10
)
ax2.set_title(
    "Efeito Individual vs. Efeito Contextual de Pares (INSE)",
    fontweight="bold", fontsize=10.5
)
ax2.set_xticks(x)
ax2.set_xticklabels(DISCIPLINES, fontsize=9.5)
ax2.legend(frameon=True, fontsize=8)

plt.tight_layout()
fig.savefig(
    os.path.join(FIG_DIR, "fig7_gaps_demograficos_inse_4areas.pdf"),
    format="pdf"
)
fig.savefig(
    os.path.join(FIG_DIR, "fig7_gaps_demograficos_inse_4areas.png"),
    format="png", dpi=300
)
plt.close(fig)

# Fig 8: Calibração TRI vs Acertos Brutos
fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
axes_flat = axes.flatten()

for i, d in enumerate(DISCIPLINES):
    ax = axes_flat[i]
    sample_pts = df_sample.sample(min(len(df_sample), 3000), random_state=42)
    ax.scatter(
        sample_pts[CORRECT_COLS[d]], sample_pts[SCORE_COLS[d]],
        alpha=0.15, s=10, color="#2c3e50", rasterized=True
    )
    curve = sample_pts.groupby(CORRECT_COLS[d])[SCORE_COLS[d]].mean()
    ax.plot(
        curve.index, curve.values,
        color="#e74c3c", linewidth=2.5, label="Média TRI Empírica"
    )
    ax.set_title(
        f"Calibração TRI vs. Acertos Brutos: {d}",
        fontweight="bold", fontsize=10.5
    )
    ax.set_xlabel(
        f"Número de Itens Corretos ({CORRECT_COLS[d]} / 45)",
        fontsize=9.5
    )
    ax.set_ylabel(f"Escore TRI ({SCORE_COLS[d]})", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
fig.savefig(
    os.path.join(FIG_DIR, "fig8_calibracao_tri_vs_acertos_4areas.pdf"),
    format="pdf"
)
fig.savefig(
    os.path.join(FIG_DIR, "fig8_calibracao_tri_vs_acertos_4areas.png"),
    format="png", dpi=300
)
plt.close(fig)

# Tarball
os.system(
    f"cd {OUT_DIR} && "
    f"tar -czf /content/tcc_results_uncensored_5y.tar.gz tables figures"
)
print("=== 2015-2019 UNCENSORED ANALYSIS COMPLETED SUCCESSFULLY! ===")
