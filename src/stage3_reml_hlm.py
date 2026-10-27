"""Stage 3: REML Estimation for Model 3 and Model 4 (HLM).

Saves individual discipline checkpoints to resume safely.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

OUT_DIR = "/content/tcc_output"
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
TAB_DIR = os.path.join(OUT_DIR, "tables")

print("=== STAGE 3: REML ESTIMATION (MODELS 3 & 4) ===")

sample_df = pd.read_parquet(os.path.join(OUT_DIR, "sample_reml_200k.parquet"))
print(f"Loaded REML sample: {len(sample_df):,} records.")

DISCIPLINES = ["MT", "CN", "CH", "LC"]
SCORE_COLS = {
    "MT": "NU_NOTA_MT",
    "CN": "NU_NOTA_CN",
    "CH": "NU_NOTA_CH",
    "LC": "NU_NOTA_LC"
}

reml_summary = []
m4_models = {}
comp_coefs = []

for disc in DISCIPLINES:
    col = SCORE_COLS[disc]
    ckpt_file = os.path.join(CKPT_DIR, f"reml_{disc}.json")
    coef_csv = os.path.join(TAB_DIR, f"tabela_regressao_hlm_{disc}.csv")

    if os.path.exists(ckpt_file) and os.path.exists(coef_csv):
        print(f"Discipline {disc} already computed. Loading checkpoint...")
        with open(ckpt_file, "r") as f:
            reml_summary.append(json.load(f))
        coef_df = pd.read_csv(coef_csv)
        coef_map = dict(zip(coef_df.iloc[:, 0], coef_df["Coeficiente"]))
        comp_coefs.append({
            "Area": disc,
            "Intercepto": coef_map.get("Intercept", 0.0),
            "Ano_2016": coef_map.get(
                "C(NU_ANO, Treatment(reference=2015))[T.2016]", 0.0
            ),
            "Ano_2017": coef_map.get(
                "C(NU_ANO, Treatment(reference=2015))[T.2017]", 0.0
            ),
            "Ano_2018": coef_map.get(
                "C(NU_ANO, Treatment(reference=2015))[T.2018]", 0.0
            ),
            "Ano_2019": coef_map.get(
                "C(NU_ANO, Treatment(reference=2015))[T.2019]", 0.0
            ),
            "Raca_Preta": coef_map.get(
                "C(TP_COR_RACA, Treatment(reference=1))[T.2]", 0.0
            ),
            "Raca_Parda": coef_map.get(
                "C(TP_COR_RACA, Treatment(reference=1))[T.3]", 0.0
            ),
            "Raca_Indigena": coef_map.get(
                "C(TP_COR_RACA, Treatment(reference=1))[T.5]", 0.0
            ),
            "INSE_Individual": coef_map.get("INSE_CENTR", 0.0),
            "INSE_Escolar": coef_map.get("INSE_ESCOLA", 0.0),
            "Rede_Federal": coef_map.get(
                "C(TP_DEPENDENCIA_ADM_ESC, Treatment(reference=2))[T.1.0]",
                coef_map.get(
                    "C(TP_DEPENDENCIA_ADM_ESC, Treatment(reference=2))[T.1]",
                    0.0
                )
            ),
            "Rede_Privada": coef_map.get(
                "C(TP_DEPENDENCIA_ADM_ESC, Treatment(reference=2))[T.4.0]",
                coef_map.get(
                    "C(TP_DEPENDENCIA_ADM_ESC, Treatment(reference=2))[T.4]",
                    0.0
                )
            ),
            "Local_Rural": coef_map.get(
                "C(TP_LOCALIZACAO_ESC, Treatment(reference=1))[T.2.0]",
                coef_map.get(
                    "C(TP_LOCALIZACAO_ESC, Treatment(reference=1))[T.2]",
                    0.0
                )
            )
        })
        continue

    print(f"\n--- Estimating REML Models for {disc} ---")

    # M0: Null Model
    m0 = smf.mixedlm(
        f"{col} ~ 1", data=sample_df, groups=sample_df["CLUSTER_ID"]
    ).fit(reml=True)
    s2_u0 = float(m0.cov_re.iloc[0, 0])
    s2_e0 = float(m0.scale)
    icc0 = s2_u0 / (s2_u0 + s2_e0)

    # M3: Year Fixed + School Random
    m3 = smf.mixedlm(
        f"{col} ~ C(NU_ANO, Treatment(reference=2015))",
        data=sample_df,
        groups=sample_df["CLUSTER_ID"]
    ).fit(reml=True)
    s2_u3 = float(m3.cov_re.iloc[0, 0])
    s2_e3 = float(m3.scale)
    pred3 = m3.predict(sample_df)
    var_fix3 = float(np.var(pred3))
    r2_marg3 = var_fix3 / (var_fix3 + s2_u3 + s2_e3)
    r2_cond3 = (var_fix3 + s2_u3) / (var_fix3 + s2_u3 + s2_e3)

    # LRT test
    ols_m = smf.ols(f"{col} ~ C(NU_ANO)", data=sample_df).fit()
    lr_stat = float(2 * (m3.llf - ols_m.llf))
    p_lrt = float(0.5 * (1.0 - stats.chi2.cdf(max(0, lr_stat), df=1)))

    # M4: Full HLM without gender
    f4 = (
        f"{col} ~ C(NU_ANO, Treatment(reference=2015)) + "
        f"C(TP_COR_RACA, Treatment(reference=1)) + "
        f"INSE_CENTR + INSE_ESCOLA + "
        f"C(TP_DEPENDENCIA_ADM_ESC, Treatment(reference=2)) + "
        f"C(TP_LOCALIZACAO_ESC, Treatment(reference=1))"
    )
    m4 = smf.mixedlm(
        f4, data=sample_df, groups=sample_df["CLUSTER_ID"]
    ).fit(reml=True)
    m4_models[disc] = m4
    s2_u4 = float(m4.cov_re.iloc[0, 0])
    s2_e4 = float(m4.scale)
    pred4 = m4.predict(sample_df)
    var_fix4 = float(np.var(pred4))
    r2_marg4 = var_fix4 / (var_fix4 + s2_u4 + s2_e4)
    r2_cond4 = (var_fix4 + s2_u4) / (var_fix4 + s2_u4 + s2_e4)

    delta_esc = (s2_u0 - s2_u4) / s2_u0 * 100.0
    delta_alu = (s2_e0 - s2_e4) / s2_e0 * 100.0

    reml_dict = {
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
    }
    reml_summary.append(reml_dict)

    with open(ckpt_file, "w") as f:
        json.dump(reml_dict, f)

    coef_df = pd.DataFrame({
        "Coeficiente": m4.params,
        "Erro_Padrao": m4.bse,
        "Estatistica_z": m4.tvalues,
        "p_valor": m4.pvalues,
        "IC_Inf_95%": m4.conf_int()[0],
        "IC_Sup_95%": m4.conf_int()[1]
    })
    coef_df.to_csv(os.path.join(TAB_DIR, f"tabela_regressao_hlm_{disc}.csv"))
    with open(
        os.path.join(TAB_DIR, f"tabela_regressao_hlm_{disc}.tex"), "w"
    ) as f:
        f.write(coef_df.to_latex(float_format="%.4f"))

    comp_coefs.append({
        "Area": disc,
        "Intercepto": m4.fe_params.get("Intercept", 0.0),
        "Ano_2016": m4.fe_params.get(
            "C(NU_ANO, Treatment(reference=2015))[T.2016]", 0.0
        ),
        "Ano_2017": m4.fe_params.get(
            "C(NU_ANO, Treatment(reference=2015))[T.2017]", 0.0
        ),
        "Ano_2018": m4.fe_params.get(
            "C(NU_ANO, Treatment(reference=2015))[T.2018]", 0.0
        ),
        "Ano_2019": m4.fe_params.get(
            "C(NU_ANO, Treatment(reference=2015))[T.2019]", 0.0
        ),
        "Raca_Preta": m4.fe_params.get(
            "C(TP_COR_RACA, Treatment(reference=1))[T.2]", 0.0
        ),
        "Raca_Parda": m4.fe_params.get(
            "C(TP_COR_RACA, Treatment(reference=1))[T.3]", 0.0
        ),
        "Raca_Indigena": m4.fe_params.get(
            "C(TP_COR_RACA, Treatment(reference=1))[T.5]", 0.0
        ),
        "INSE_Individual": m4.fe_params.get("INSE_CENTR", 0.0),
        "INSE_Escolar": m4.fe_params.get("INSE_ESCOLA", 0.0),
        "Rede_Federal": m4.fe_params.get(
            "C(TP_DEPENDENCIA_ADM_ESC, Treatment(reference=2))[T.1.0]",
            m4.fe_params.get(
                "C(TP_DEPENDENCIA_ADM_ESC, Treatment(reference=2))[T.1]", 0.0
            )
        ),
        "Rede_Privada": m4.fe_params.get(
            "C(TP_DEPENDENCIA_ADM_ESC, Treatment(reference=2))[T.4.0]",
            m4.fe_params.get(
                "C(TP_DEPENDENCIA_ADM_ESC, Treatment(reference=2))[T.4]", 0.0
            )
        ),
        "Local_Rural": m4.fe_params.get(
            "C(TP_LOCALIZACAO_ESC, Treatment(reference=1))[T.2.0]",
            m4.fe_params.get(
                "C(TP_LOCALIZACAO_ESC, Treatment(reference=1))[T.2]", 0.0
            )
        )
    })

    # Save random effects for this discipline
    blup_s = pd.Series(
        {k: v.iloc[0] for k, v in m4.random_effects.items()},
        name=f"u_hat_{disc}"
    )
    blup_s.to_frame().to_parquet(
        os.path.join(OUT_DIR, f"blup_{disc}.parquet")
    )

reml_df = pd.DataFrame(reml_summary)
print("\nREML & HLM Summary (2015-2019):")
print(reml_df.to_string(index=False))

reml_df.to_csv(
    os.path.join(TAB_DIR, "tabela7_resumo_reml_hlm_tri.csv"), index=False
)
with open(
    os.path.join(TAB_DIR, "tabela7_resumo_reml_hlm_tri.tex"), "w"
) as f:
    f.write(reml_df.to_latex(index=False, float_format="%.3f"))

comp_df = pd.DataFrame(comp_coefs)
comp_df.to_csv(
    os.path.join(TAB_DIR, "tabela8_comparacao_coeficientes_hlm.csv"),
    index=False
)
with open(
    os.path.join(TAB_DIR, "tabela8_comparacao_coeficientes_hlm.tex"), "w"
) as f:
    f.write(comp_df.to_latex(index=False, float_format="%.2f"))

with open(os.path.join(CKPT_DIR, "stage3_done.json"), "w") as f:
    json.dump({"status": "DONE"}, f)

print("=== STAGE 3 COMPLETED! ===")
