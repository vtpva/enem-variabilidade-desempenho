import os
import pandas as pd

TCC_DIR = "/content/drive/MyDrive/TCC"
years = [2018, 2019, 2020, 2021, 2022]

print("=== INSPECTING COHORTS (2018-2022) ===")

for y in years:
    path = os.path.join(TCC_DIR, f"MICRODADOS_ENEM_{y}.parquet")
    df = pd.read_parquet(path)
    print(f"\n================ YEAR {y} ================")
    print(f"Total records: {len(df):,}")

    concluintes = df[df["TP_ST_CONCLUSAO"] == 2]
    pct_conc = len(concluintes) / len(df) * 100
    print(
        f"Concluintes (TP_ST_CONCLUSAO == 2): {len(concluintes):,} "
        f"({pct_conc:.1f}%)"
    )

    score_cols = [
        "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC",
        "NU_NOTA_MT", "NU_NOTA_REDACAO"
    ]
    valid_scores = concluintes.dropna(subset=score_cols)
    valid_scores = valid_scores[
        (valid_scores["NU_NOTA_CN"] > 0) &
        (valid_scores["NU_NOTA_CH"] > 0) &
        (valid_scores["NU_NOTA_LC"] > 0) &
        (valid_scores["NU_NOTA_MT"] > 0) &
        (valid_scores["NU_NOTA_REDACAO"] > 0)
    ]
    print(
        f"Concluintes with valid scores in all 5 areas: "
        f"{len(valid_scores):,}"
    )

    print("Non-null counts in valid concluintes:")
    cols_to_check = [
        "CO_MUNICIPIO_ESC", "SG_UF_ESC", "TP_DEPENDENCIA_ADM_ESC",
        "TP_LOCALIZACAO_ESC", "TP_ESCOLA", "CO_MUNICIPIO_PROVA",
        "SG_UF_PROVA", "TP_COR_RACA", "TP_SEXO", "TP_FAIXA_ETARIA"
    ]
    for col in cols_to_check:
        if col in valid_scores.columns:
            non_nulls = valid_scores[col].notna().sum()
            pct_nn = non_nulls / len(valid_scores) * 100
            print(f"  {col}: {non_nulls:,} ({pct_nn:.1f}%)")

    print("TP_DEPENDENCIA_ADM_ESC distribution:")
    print(valid_scores["TP_DEPENDENCIA_ADM_ESC"].value_counts(dropna=False))
    print("TP_ESCOLA distribution:")
    print(valid_scores["TP_ESCOLA"].value_counts(dropna=False))
