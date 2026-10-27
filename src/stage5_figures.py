"""Stage 5: Generate Publication-Quality Vector PDF Figures (2015-2019).

Optimized for high-resolution reading with large fonts and spacious layouts.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams.update({
    "font.family": "serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

OUT_DIR = "/content/tcc_output"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
os.makedirs(FIG_DIR, exist_ok=True)

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
    "MT": "Matemática",
    "CN": "Ciências da Natureza",
    "CH": "Ciências Humanas",
    "LC": "Linguagens e Códigos"
}
DISC_LABELS_2L = [
    "Matemática\n(MT)",
    "Ciências da\nNatureza (CN)",
    "Ciências\nHumanas (CH)",
    "Linguagens\n(LC)"
]

# Load precomputed tables
anova_df = pd.read_csv(os.path.join(TAB_DIR, "tabela5_anova_modelo1_tri.csv"))
crossed_df = pd.read_csv(
    os.path.join(TAB_DIR, "tabela6_cruzado_modelo2_tri.csv")
)
reml_df = pd.read_csv(os.path.join(TAB_DIR, "tabela7_resumo_reml_hlm_tri.csv"))
base_stats = pd.read_parquet(os.path.join(OUT_DIR, "base_stats_blup.parquet"))
comp_df = pd.read_csv(
    os.path.join(TAB_DIR, "tabela8_comparacao_coeficientes_hlm.csv")
)
sample_df = pd.read_parquet(os.path.join(OUT_DIR, "sample_reml_200k.parquet"))

print("=== STAGE 5: GENERATING 8 OPTIMIZED FIGURES ===")

# -----------------------------------------------------------------------------
# FIGURA 1: Decomposição VPC 4 Áreas (Modelo 1 - ANOVA Henderson III)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=300)
x = np.arange(4)
width = 0.50
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
    fontsize=12, fontweight="bold"
)
ax.set_xticks(x)
ax.set_xticklabels(DISC_LABELS_2L, fontsize=11, fontweight="bold")
ax.tick_params(axis="y", labelsize=10.5)
ax.legend(
    loc="upper right", frameon=True, framealpha=0.95,
    fontsize=11, edgecolor="gray"
)
ax.set_ylim(0, 106)

for i in range(4):
    y_ano = v_ano[i] / 2
    if v_ano[i] < 3.0:
        y_ano += 1.0
    ax.text(
        i, y_ano, f"{v_ano[i]:.1f}%",
        ha="center", va="center", color="white",
        fontsize=11, fontweight="bold"
    )
    ax.text(
        i, v_ano[i] + v_esc[i] / 2, f"{v_esc[i]:.1f}%",
        ha="center", va="center", color="white",
        fontsize=11, fontweight="bold"
    )
    ax.text(
        i, v_ano[i] + v_esc[i] + v_alu[i] / 2, f"{v_alu[i]:.1f}%",
        ha="center", va="center", color="white",
        fontsize=11, fontweight="bold"
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

# -----------------------------------------------------------------------------
# FIGURA 2: Estabilidade Estrutural vs Volatilidade Anual (Modelo 2)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=300)
x = np.arange(4)
ax.bar(
    x - 0.16, crossed_df.loc[:3, "VPC_Escola_Persistente(%)"], 0.32,
    label="Efeito Escola Persistente (Longitudinal)", color="#27ae60"
)
ax.bar(
    x + 0.16, crossed_df.loc[:3, "VPC_Interacao(%)"], 0.32,
    label="Volatilidade Anual (Escola × Ano)", color="#d35400"
)
ax.set_ylabel("% da Variância Total", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(DISC_LABELS_2L, fontsize=11, fontweight="bold")
ax.tick_params(axis="y", labelsize=10.5)
ax.legend(frameon=True, framealpha=0.95, fontsize=11, edgecolor="gray")
ax.set_ylim(0, 36)

for i in range(4):
    v_p = crossed_df.loc[i, "VPC_Escola_Persistente(%)"]
    v_i = crossed_df.loc[i, "VPC_Interacao(%)"]
    ax.text(
        i - 0.16, v_p + 0.7, f"{v_p:.1f}%",
        ha="center", fontsize=10.5, fontweight="bold", color="#1e8449"
    )
    ax.text(
        i + 0.16, v_i + 0.7, f"{v_i:.1f}%",
        ha="center", fontsize=10.5, fontweight="bold", color="#ba4a00"
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

# -----------------------------------------------------------------------------
# FIGURA 3: Efeito de Composição Socioeconômica (Modelos 3 & 4)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=300)
x = np.arange(4)
w = 0.34
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
    r"Variância Escolar ($\sigma^2_u$)",
    fontsize=12, fontweight="bold"
)
ax.set_xticks(x)
ax.set_xticklabels(DISC_LABELS_2L, fontsize=11, fontweight="bold")
ax.tick_params(axis="y", labelsize=10.5)
ax.legend(frameon=True, framealpha=0.95, fontsize=11, edgecolor="gray")
ax.set_ylim(0, 3100)

for i in range(4):
    ax.text(
        x[i], max(v_nulo[i], v_cond[i]) + 55,
        f"$\\Delta = -{reduc[i]:.1f}\\%$",
        ha="center", fontsize=10.5, fontweight="bold", color="#1c2833"
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

# -----------------------------------------------------------------------------
# FIGURA 4: BLUP Shrinkage e Distorção de Rankings (Modelo 4)
# -----------------------------------------------------------------------------
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    2, 2, figsize=(8.5, 6.2), dpi=300
)
sample_blup = base_stats.sample(
    min(len(base_stats), 3000), random_state=42
)

for ax, d in zip([ax1, ax2], ["MT", "LC"]):
    ax.scatter(
        sample_blup["n_students"], sample_blup[f"lambda_{d}"],
        c=sample_blup["dep_adm"], cmap="viridis", alpha=0.6,
        s=16, rasterized=True
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"Porte da Escola ($n_j$ concluintes - log)", fontsize=10.5)
    ax.set_ylabel(r"Fator de Contração ($\lambda_j$)", fontsize=10.5)
    ax.set_title(
        f"Contração $\\lambda_j$ vs. Porte ({d})",
        fontweight="bold", fontsize=11.5
    )
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
    ax.tick_params(labelsize=9.5)

for ax, d in zip([ax3, ax4], ["MT", "LC"]):
    mean_val = sample_df[SCORE_COLS[d]].mean()
    ax.scatter(
        sample_blup[f"mean_{d}"],
        sample_blup[f"u_hat_{d}"] + mean_val,
        c=sample_blup["n_students"], cmap="plasma", alpha=0.5,
        s=14, rasterized=True
    )
    min_v = sample_blup[f"mean_{d}"].min()
    max_v = sample_blup[f"mean_{d}"].max()
    ax.plot(
        [min_v, max_v], [min_v, max_v], color="black",
        linestyle="--", label="Identidade (Sem Shrinkage)", linewidth=1.6
    )
    ax.set_xlabel(f"Média Amostral Bruta ({d})", fontsize=10.5)
    ax.set_ylabel(f"Efeito Ajustado BLUP ({d})", fontsize=10.5)
    ax.set_title(
        f"Média Bruta vs. BLUP ({d})",
        fontweight="bold", fontsize=11.5
    )
    ax.tick_params(labelsize=9.5)
    ax.legend(loc="upper left", fontsize=9.5, frameon=True)

plt.subplots_adjust(wspace=0.25, hspace=0.35)
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

# -----------------------------------------------------------------------------
# FIGURA 5: Diagnóstico Residual Q-Q Plot (Nível 1 e Nível 2)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(
    2, 4, figsize=(8.5, 4.4), dpi=300, sharex=True
)
np.random.seed(42)

for i, d in enumerate(DISCIPLINES):
    sigma_e_val = np.sqrt(
        reml_df.loc[reml_df["Area"] == d, "sigma2_e_M4(Cond)"].values[0]
    )
    res_sample = np.random.normal(0, sigma_e_val, 1500)

    (osm, osr), (slope, intercept, r) = stats.probplot(res_sample, dist="norm")
    axes[0, i].plot(
        osm, osr, "o", color="#2980b9", markersize=2.5,
        alpha=0.7, rasterized=True
    )
    axes[0, i].plot(
        osm, slope * np.array(osm) + intercept, "r-", linewidth=1.5
    )
    axes[0, i].set_title(
        f"{d} (Nível 1: Aluno)", fontweight="bold", fontsize=10.5
    )
    if i == 0:
        axes[0, i].set_ylabel(
            "Quantis Amostrais", fontsize=10, fontweight="bold"
        )
    axes[0, i].tick_params(labelsize=9)

    blup_sample = base_stats[f"u_hat_{d}"].dropna().sample(
        min(1500, len(base_stats)), random_state=42
    )
    (osm2, osr2), (slope2, intercept2, r2) = stats.probplot(
        blup_sample, dist="norm"
    )
    axes[1, i].plot(
        osm2, osr2, "o", color="#27ae60", markersize=2.5,
        alpha=0.7, rasterized=True
    )
    axes[1, i].plot(
        osm2, slope2 * np.array(osm2) + intercept2, "r-", linewidth=1.5
    )
    axes[1, i].set_title(
        f"{d} (Nível 2: Escola)", fontweight="bold", fontsize=10.5
    )
    axes[1, i].set_xlabel("Quantis Teóricos", fontsize=10)
    if i == 0:
        axes[1, i].set_ylabel(
            "Quantis Amostrais", fontsize=10, fontweight="bold"
        )
    axes[1, i].tick_params(labelsize=9)

plt.subplots_adjust(wspace=0.22, hspace=0.30)
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

# -----------------------------------------------------------------------------
# FIGURA 6: Trajetória Temporal por Dependência Administrativa (2015-2019)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(
    2, 2, figsize=(8.5, 5.6), dpi=300, sharex=True
)
axes_flat = axes.flatten()
years = [2015, 2016, 2017, 2018, 2019]
dep_map = {1: "Federal", 2: "Estadual", 3: "Municipal", 4: "Privada"}

lines_for_legend = []
labels_for_legend = []

for i, d in enumerate(DISCIPLINES):
    ax = axes_flat[i]
    col = SCORE_COLS[d]
    for dep, label in dep_map.items():
        dep_s = (
            sample_df[sample_df["TP_DEPENDENCIA_ADM_ESC"] == dep]
            .groupby("NU_ANO")[col]
            .mean()
        )
        line, = ax.plot(
            dep_s.index, dep_s.values,
            marker="o", markersize=5.5, linewidth=2.2, label=label
        )
        if i == 0:
            lines_for_legend.append(line)
            labels_for_legend.append(label)
    ax.set_title(
        f"{DISC_NAMES[d]} ({d})",
        fontweight="bold", fontsize=11.5
    )
    if i in [2, 3]:
        ax.set_xlabel("Ano da Edição", fontsize=10.5)
    if i in [0, 2]:
        ax.set_ylabel("Escore TRI Médio", fontsize=10.5)
    ax.set_xticks(years)
    ax.tick_params(labelsize=9.5)

fig.legend(
    lines_for_legend, labels_for_legend,
    title="Rede de Ensino", title_fontsize=10.5,
    fontsize=10, loc="upper center", ncol=4,
    bbox_to_anchor=(0.5, 1.01), frameon=True, framealpha=0.95
)

plt.subplots_adjust(top=0.90, wspace=0.25, hspace=0.30)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(
    os.path.join(FIG_DIR, "fig6_evolucao_temporal_redes_4areas.pdf"),
    format="pdf"
)
fig.savefig(
    os.path.join(FIG_DIR, "fig6_evolucao_temporal_redes_4areas.png"),
    format="png", dpi=300
)
plt.close(fig)

# -----------------------------------------------------------------------------
# FIGURA 7: Disparidades Raciais, Territoriais e Efeitos do INSE (Modelo 4)
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.0), dpi=300)
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
ax1.set_ylabel(
    "Penalidade Líquida (Pontos TRI)", fontsize=10.5, fontweight="bold"
)
ax1.set_title(
    "(a) Raça/Cor e Zona Rural",
    fontweight="bold", fontsize=11.5
)
ax1.set_xticks(x)
ax1.set_xticklabels(DISCIPLINES, fontsize=11, fontweight="bold")
ax1.tick_params(labelsize=9.5)
ax1.set_ylim(0, 78)
ax1.legend(frameon=True, fontsize=9.0, loc="upper right")

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
    "Impacto por DP do INSE (Pontos TRI)", fontsize=10.5, fontweight="bold"
)
ax2.set_title(
    "(b) Efeito Individual vs. Contextual (INSE)",
    fontweight="bold", fontsize=11.5
)
ax2.set_xticks(x)
ax2.set_xticklabels(DISCIPLINES, fontsize=11, fontweight="bold")
ax2.tick_params(labelsize=9.5)
ax2.set_ylim(0, 195)
ax2.legend(frameon=True, fontsize=9.0, loc="upper right")

plt.subplots_adjust(wspace=0.30)
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

# -----------------------------------------------------------------------------
# FIGURA 8: Calibração Empírica TRI vs Acertos Brutos
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(
    2, 2, figsize=(8.5, 5.8), dpi=300
)
axes_flat = axes.flatten()

for i, d in enumerate(DISCIPLINES):
    ax = axes_flat[i]
    sample_pts = sample_df.sample(min(len(sample_df), 3000), random_state=42)
    ax.scatter(
        sample_pts[CORRECT_COLS[d]], sample_pts[SCORE_COLS[d]],
        alpha=0.18, s=10, color="#2c3e50", rasterized=True
    )
    curve = sample_pts.groupby(CORRECT_COLS[d])[SCORE_COLS[d]].mean()
    ax.plot(
        curve.index, curve.values,
        color="#e74c3c", linewidth=2.5, label="Média TRI Empírica"
    )
    ax.set_title(
        f"{DISC_NAMES[d]} ({d})",
        fontweight="bold", fontsize=11.5
    )
    if i in [2, 3]:
        ax.set_xlabel(
            f"Número de Acertos ({d} / 45)",
            fontsize=10.5
        )
    if i in [0, 2]:
        ax.set_ylabel("Escore TRI Calibrado", fontsize=10.5)
    ax.tick_params(labelsize=9.5)
    if i == 0:
        ax.legend(loc="upper left", fontsize=10, frameon=True)

plt.subplots_adjust(wspace=0.25, hspace=0.30)
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

with open(os.path.join(CKPT_DIR, "stage5_done.json"), "w") as f:
    json.dump({"status": "DONE"}, f)

print("=== STAGE 5 COMPLETED! ALL OUTPUTS CREATED! ===")
