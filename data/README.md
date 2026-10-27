# Microdados do ENEM (INEP)

Os dados utilizados neste estudo provêm dos microdados públicos do Exame Nacional do Ensino Médio (ENEM), disponibilizados oficialmente pelo Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira (INEP).

## Edições Utilizadas (2015–2019)

Para a preservação do identificador institucional da escola (`CO_ESCOLA`), utilizam-se exclusivamente as últimas cinco edições anteriores à censura cadastral motivada pela Lei Geral de Proteção de Dados (LGPD) a partir de 2020:
- `MICRODADOS_ENEM_2015.parquet`
- `MICRODADOS_ENEM_2016.parquet`
- `MICRODADOS_ENEM_2017.parquet`
- `MICRODADOS_ENEM_2018.parquet`
- `MICRODADOS_ENEM_2019.parquet`

## Como Obter os Microdados

1. Acesse o portal oficial de dados abertos do INEP:
   [https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem)
2. Baixe os arquivos compactados das edições de 2015 a 2019.
3. Extraia os arquivos CSV correspondentes (`DADOS/MICRODADOS_ENEM_{ano}.csv`).
4. Para otimização de leitura e redução drástica do consumo de memória RAM, converta-os para o formato colunar Parquet (`MICRODADOS_ENEM_{ano}.parquet`) e armazene-os nesta pasta `data/`.
