"""Stage 4: BLUP Shrinkage Factor, Ranking Displacement and Diagnostics.

Outputs Tables 9 and 10 and BLUP dataframe.
"""

import json
import os
import pandas as pd
from scipy import stats

OUT_DIR = "/content/tcc_output"
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
TAB_DIR = os.path.join(OUT_DIR, "tables")

print("=== STAGE 4: BLUP SHRINKAGE & RANK DISPLACEMENT ===")

sample_df = pd.read_parquet(os.path.join(OUT_DIR, "sample_reml_200k.parquet"))
reml_df = pd.read_csv(os.path.join(TAB_DIR, "tabela7_resumo_reml_hlm_tri.csv"))

DISCIPLINES = ["MT", "CN", "CH", "LC"]

base_stats = sample_df.groupby("CLUSTER_ID").agg(
    n_students=("NU_NOTA_MT", "count"),
    mean_MT=("NU_NOTA_MT", "mean"),
    mean_CN=("NU_NOTA_CN", "mean"),
    mean_CH=("NU_NOTA_CH", "mean"),
    mean_LC=("NU_NOTA_LC", "mean"),
    dep_adm=("TP_DEPENDENCIA_ADM_ESC", "first"),
    uf=("SG_UF_ESC", "first"),
    inse_mean=("INSE_ESCOLA", "first")
).reset_index()

base_stats["CLUSTER_ID"] = base_stats["CLUSTER_ID"].astype(str)

for disc in DISCIPLINES:
    blup_d = pd.read_parquet(
        os.path.join(OUT_DIR, f"blup_{disc}.parquet")
    )
    blup_d.index = blup_d.index.astype(str)
    base_stats = pd.merge(
        base_stats, blup_d,
        left_on="CLUSTER_ID", right_index=True, how="inner"
    )

for disc in DISCIPLINES:
    s2u = float(
        reml_df.loc[reml_df["Area"] == disc, "sigma2_u_M4(Cond)"].values[0]
    )
    s2e = float(
        reml_df.loc[reml_df["Area"] == disc, "sigma2_e_M4(Cond)"].values[0]
    )
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
base_stats.to_parquet(
    os.path.join(OUT_DIR, "base_stats_blup.parquet"), index=False
)

rank_summary = []
for disc in DISCIPLINES:
    spearman_rho, _ = stats.spearmanr(
        base_stats[f"rank_raw_{disc}"], base_stats[f"rank_blup_{disc}"]
    )
    mean_abs_diff = float(base_stats[f"rank_diff_{disc}"].abs().mean())
    q_turnover = int((
        (base_stats[f"rank_raw_{disc}"] <= 100) &
        (base_stats[f"rank_blup_{disc}"] > 100)
    ).sum())

    rank_summary.append({
        "Area": disc,
        "Spearman_Rank_Corr": float(spearman_rho),
        "Deslocamento_Medio_Posicoes": mean_abs_diff,
        "Escolas_Top100_Bruto_Fora_Top100_BLUP": q_turnover
    })

rank_sum_df = pd.DataFrame(rank_summary)
print("\nRank Displacement Summary:")
print(rank_sum_df.to_string(index=False))

rank_sum_df.to_csv(
    os.path.join(TAB_DIR, "tabela10_resumo_deslocamento_rankings.csv"),
    index=False
)
with open(
    os.path.join(TAB_DIR, "tabela10_resumo_deslocamento_rankings.tex"), "w"
) as f:
    f.write(rank_sum_df.to_latex(index=False, float_format="%.3f"))

with open(os.path.join(CKPT_DIR, "stage4_done.json"), "w") as f:
    json.dump({"status": "DONE"}, f)

print("=== STAGE 4 COMPLETED! ===")
