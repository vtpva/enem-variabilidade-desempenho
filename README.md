# Decomposição da Variabilidade do Desempenho no ENEM (2015–2019)
### Modelagem Hierárquica e Modelos Lineares Mistos nas Quatro Áreas da TRI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)

Repositório com o código-fonte, pipeline de processamento, rotinas econométricas e outputs analíticos da pesquisa sobre a decomposição da variância das notas do Exame Nacional do Ensino Médio (ENEM).

---

## 🎯 Objetivo da Pesquisa

O objetivo central deste projeto é investigar a origem da variabilidade das notas do ENEM ao longo do quinquênio **2015 a 2019** (últimas cinco edições públicas antes da censura cadastral do código da escola `CO_ESCOLA`), decompondo o desempenho dos estudantes em componentes de variância hierárquicos:
1. **Nível 1 (Intraescolar/Individual):** Dispersão decorrente do esforço discente, histórico educacional e perfil socioeconômico individual.
2. **Nível 2 (Interinstitucional/Escola):** Heterogeneidade estrutural associada à eficácia pedagógica, rede administrativa (Federal, Estadual, Municipal, Privada) e composição socioeconômica agregada (efeito de pares).
3. **Dimensão Temporal (Ano):** Flutuações macroeconômicas e calibração de itens ao longo das sucessivas edições do exame.

A análise é segmentada de forma estrita nas **quatro áreas objetivas avaliadas pela Teoria de Resposta ao Item (TRI)**:
- **MT:** Matemática e suas Tecnologias
- **CN:** Ciências da Natureza e suas Tecnologias
- **CH:** Ciências Humanas e suas Tecnologias
- **LC:** Linguagens, Códigos e suas Tecnologias

A base harmonizada abrange o universo de **6.181.835 concluintes regulares** do Ensino Médio em **38.336 agrupamentos escolares**.

---

## 🔬 Modelos Implementados

1. **Modelo 1 (ANOVA Henderson Tipo III - Encaixado Puro):** Estimação incondicional dos componentes de variância ($\sigma^2_{Ano}$, $\sigma^2_{Escola}$, $\sigma^2_{Aluno}$) e cálculo do Coeficiente de Correlação Intraclasse (ICC) e Coeficiente de Partição de Variância (VPC).
2. **Modelo 2 (Modelo Longitudinal Cruzado Escola $\times$ Ano):** Decomposição sobre painel balanceado de 6.186 escolas presentes nas 5 edições, mensurando o **Índice de Persistência Institucional** e isolando a variância permanente da volatilidade transitória de turmas.
3. **Modelo 3 (Modelo Misto Linear - LMM/REML):** Modelagem com Ano como termo de efeito fixo histórico e Escola como intercepto aleatório ($u_j \sim \mathcal{N}(0, \sigma^2_u)$), avaliado via Máxima Verossimilhança Restrita (REML) e $R^2$ de Nakagawa & Schielzeth.
4. **Modelo 4 (Modelo Hierárquico Multinível Completo - HLM):** Controle simultâneo por fatores individuais (raça/cor autodeclarada e escore centrado de INSE) e contextuais (INSE médio da escola, rede de ensino e localização rural), calculando a **Redução da Variância Escolar por Efeito de Composição** ($\Delta\sigma^2_{escola}$).
5. **Estimação de Preditores BLUP e Análise de Rankings:** Cálculo dos Melhores Preditores Lineares Não Viesados (BLUP) com fator de contração empírica de Bayes ($\lambda_j$), diagnosticando a volatilidade de pequenos estabelecimentos e comparando rankings brutos vs. rankings ajustados por valor agregado.

---

## 📊 Outputs do Projeto

Os principais resultados analíticos gerados pelo pipeline estão organizados nas pastas de outputs:

### 1. Figuras Vetoriais (`figures/`)
- `fig1_decomposicao_variancia_4areas.pdf`: Decomposição percentual da variância (VPC) nas 4 áreas.
- `fig2_estabilidade_vs_volatilidade_4areas.pdf`: Variância persistente vs. volatilidade transitória anual.
- `fig3_efeito_composicao_valor_agregado_4areas.pdf`: Redução da variância escolar após controle por perfil socioeconômico.
- `fig4_blup_shrinkage_rankings_4areas.pdf`: Fator de encolhimento $\lambda_j$ vs. porte escolar e distorção em rankings.
- `fig5_diagnostico_residuos_4areas.pdf`: Gráficos Q-Q de normalidade residual (Nível 1) e efeitos aleatórios (Nível 2).
- `fig6_evolucao_temporal_redes_4areas.pdf`: Trajetórias temporais de desempenho por rede administrativa (2015–2019).
- `fig7_gaps_demograficos_inse_4areas.pdf`: Penalidades raciais líquidas e coeficientes contextuais vs. individuais do INSE.
- `fig8_calibracao_tri_vs_acertos_4areas.pdf`: Curvas empíricas de calibração TRI vs. número de acertos brutos.

### 2. Tabelas Consolidadas (`tables/`)
- `tabela1_painel_cohortes_tri.csv`: Estatísticas descritivas das cinco coortes (2015–2019).
- `tabela2_descritiva_redes_ensino.csv`: Médias e desvios por rede escolar (Federal, Estadual, Municipal, Privada).
- `tabela3_descritiva_raca_cor.csv`: Distribuição e escores por raça/cor autodeclarada.
- `tabela4_matriz_correlacao_tri_raw.csv`: Matriz de correlação entre as 4 áreas da TRI.
- `tabela5_anova_modelo1_tri.csv`: Estimativas de variância e testes $F$ (Henderson III).
- `tabela6_cruzado_modelo2_tri.csv`: Componentes do modelo longitudinal e índices de persistência.
- `tabela7_resumo_reml_hlm_tri.csv`: Comparativo de variâncias e métricas de ajuste entre Modelos 3 e 4.
- `tabela8_comparacao_coeficientes_hlm.csv`: Coeficientes fixos estimados para as 4 áreas (raça, redes, INSE).
- `tabela9_blup_rankings_4areas.csv`: Estimativas BLUP e ordenamento escolar por área.
- `tabela10_resumo_deslocamento_rankings.csv`: Deslocamento médio de posições e taxa de exclusão do Top 100.

---

## 📁 Estrutura de Diretórios

```text
├── data/                       # Instruções para obtenção dos microdados brutos do INEP
│   └── README.md
├── figures/                    # Figuras analíticas em alta resolução (PDF vetorial e PNG)
├── notebooks/                  # Notebooks Jupyter para inspeção exploratória
├── src/                        # Código-fonte modular do pipeline analítico (Python)
│   ├── stage1_etl.py           # Ingestão, filtros de concluintes e cálculo do INSE
│   ├── stage2_anova_crossed.py # Modelo 1 (Henderson III) e Modelo 2 (Cruzado Escola x Ano)
│   ├── stage3_reml_hlm.py      # Modelo 3 (LMM/REML) e Modelo 4 (HLM Multinível)
│   ├── stage4_blup_rankings.py # Preditores BLUP, encolhimento e rankings escolares
│   ├── stage5_figures.py       # Renderização vetorial das 8 figuras analíticas
│   └── optimize_figures.py     # Otimizações de layout, fontes e proporções gráficas
├── tables/                     # Tabelas econométricas geradas em CSV e LaTeX
├── dicionario-enem.csv         # Dicionário de variáveis e códigos do ENEM
├── .gitignore                  # Regras de exclusão de arquivos pesados e temporários
├── LICENSE                     # Licença de uso MIT
├── README.md                   # Documentação do projeto
└── requirements.txt            # Dependências Python necessárias
```

---

## 🛠️ Como Executar o Código

### 1. Clonar o Repositório e Instalar Dependências

```bash
git clone https://github.com/vtpva/enem-variabilidade-desempenho.git
cd enem-variabilidade-desempenho

# Criar e ativar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Microdados do ENEM

Baixe os microdados públicos das edições de 2015 a 2019 do portal do INEP e converta-os para formato Parquet na pasta `data/`, seguindo as diretrizes descritas em [`data/README.md`](data/README.md).

### 3. Execução das Etapas

Os scripts podem ser executados sequencialmente a partir da raiz do repositório:

```bash
# Estágio 1: Extração, harmonização amostral e cálculo do INSE
python src/stage1_etl.py

# Estágio 2: Decomposição de variância ANOVA e análise longitudinal cruzada
python src/stage2_anova_crossed.py

# Estágio 3: Estimação iterativa por REML dos modelos lineares mistos e multiníveis
python src/stage3_reml_hlm.py

# Estágio 4: Estimação BLUP, contração empírica de Bayes e comparação de rankings
python src/stage4_blup_rankings.py

# Estágio 5: Geração de todas as figuras vetoriais
python src/stage5_figures.py
```

---

## 📄 Licença

Este projeto é distribuído sob a licença [MIT](LICENSE). Os microdados utilizados são públicos e mantidos pelo Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira ([INEP/MEC](https://www.gov.br/inep)).
