# Previsor de Incêndios Florestais — Rio Grande do Norte

Sistema de previsão de risco de incêndio florestal por município, para o dia seguinte (D+1), baseado em Machine Learning treinado do zero sobre dados meteorológicos do INMET, focos de calor do INPE e malhas territoriais do IBGE.

Escopo atual: **167 municípios do Rio Grande do Norte, janela histórica 2020–2024**. A arquitetura permite escalar para outras UFs editando apenas `backend/configuracao.py`.

Trabalho da disciplina de Introdução à Inteligência Artificial.

## Estrutura

```
previsor-incendios-ne/
├── dados/                          gitignored (parquet, sqlite)
│   ├── brutos/                     downloads originais (INMET, INPE, IBGE)
│   ├── processados/                parquet por fonte, geojson simplificado
│   └── banco.sqlite                fato_municipio_dia
├── backend/
│   ├── configuracao.py             escopo, URLs, caminhos
│   ├── coleta/                     scripts de download (idempotentes)
│   │   ├── baixar_inmet.py
│   │   ├── baixar_dbqueimadas.py
│   │   ├── baixar_ibge.py
│   │   └── baixar_osm.py
│   ├── tratamento/                 ETL e features
│   │   ├── parsing_inmet.py        parser tolerante dos CSVs anuais
│   │   ├── calcular_fwi.py         FWI canadense (Van Wagner 1987)
│   │   ├── interpolar_meteo.py     IDW estações → municípios
│   │   ├── agregar_focos.py        spatial join INPE → polígonos IBGE
│   │   ├── features.py             lags, acumuladas, sazonalidade, alvo
│   │   └── montar_dataset.py       orquestra tudo
│   ├── modelo/
│   │   ├── base.py                 interface ModeloBase
│   │   ├── treinar.py              split temporal, RF + LightGBM
│   │   ├── avaliar.py              gera PNGs e CSVs de métricas
│   │   ├── prever.py               carga e inferência (usado pela API)
│   │   ├── artefatos/              modelos salvos (.joblib)
│   │   └── avaliacao/              métricas, curvas, plots
│   ├── api/                        FastAPI
│   │   ├── main.py                 entrypoint + lifespan
│   │   ├── estado.py               carga única do dataset e geojson
│   │   ├── rotas.py                endpoints
│   │   └── esquemas.py             modelos Pydantic
│   └── requirements.txt
├── frontend/                       Vite + React + Leaflet
│   └── src/
│       ├── App.jsx                 layout, abas, estado global
│       ├── servicos/api.js
│       └── componentes/
│           ├── Mapa.jsx
│           ├── PainelPrevisao.jsx
│           ├── PainelComparacao.jsx
│           ├── PainelFuturo.jsx
│           └── PainelRelatorio.jsx
├── notebooks/
│   └── analise_resultados.ipynb    EDA, correlações, discussão de limites
└── Makefile                        baixar | processar | treinar | servir | frontend
```

## Como rodar

```bash
# Cria o ambiente virtual
python -m venv .venv

# Ativa o ambiente
.venv\Scripts\activate

# Instala as dependências do backend
pip install -r backend/requirements.txt

#Coletar os dados, processar e treinar
# 1. Baixar os dados brutos (IBGE, INPE, INMET)
python -m backend.coleta.baixar_ibge
python -m backend.coleta.baixar_bioma
python -m backend.coleta.baixar_dbqueimadas
python -m backend.coleta.baixar_inmet

# 2. Processar, limpar e cruzar as informações
python -m backend.tratamento.montar_dataset

# 3. Treinar o modelo de Inteligência Artificial
python -m backend.modelo.treinar

# Inicia a API na porta 8000
python -m uvicorn backend.api.main:app --reload --port 8000

# Entra na pasta do frontend
cd frontend

# Instala as dependências do Node.js
npm install

# Inicia o servidor visual
npm run dev
```

O frontend está configurado para fazer proxy de `/api/*` para a API local. Suba a API antes do frontend.


## Fontes de dados

| Fonte | Padrão de URL | Notas |
|-------|---------------|-------|
| INMET | `portal.inmet.gov.br/uploads/dadoshistoricos/{ano}.zip` | ~100MB/ano, todas estações do Brasil |
| INPE  | `dataserver-coids.inpe.br/.../EstadosBr_sat_ref/{UF}/focos_br_{uf}_ref_{ano}.zip` | já filtrado pelo satélite de referência |
| IBGE  | `geoftp.ibge.gov.br/.../municipio_2024/UFs/{UF}/{UF}_Municipios_2024.zip` | shapefile, ~5MB/UF |
| OSM (opcional) | `download.geofabrik.de/south-america/brazil/nordeste-latest.osm.pbf` | ~150MB |

Satélite de referência: **AQUA_M-T** (MODIS).



## Pipeline de features

Granularidade: município × dia. 167 × 1827 = 305.109 linhas.

- **Meteorológicas**: temp_media, temp_max, temp_min, umid_media, chuva_dia, vento_medio, rad_media. Interpoladas via IDW (k=3, peso 1/d²) a partir de até 28 estações INMET (RN + PB + CE, após filtro continental).
- **FWI canadense**: ffmc, dmc, dc, isi, bui, fwi. Cálculo recursivo por estação, depois interpolação.
- **Lag de focos**: focos_lag_1, focos_lag_3, focos_lag_7 (`shift` para evitar vazamento).
- **Acumuladas**: chuva_acum_7d, chuva_acum_30d, dias_sem_chuva.
- **Sazonais**: mes_sin/cos, doy_sin/cos.
- **Geográficas**: area_km2, centro_lat, centro_lon, distancia_litoral_km, bioma (one-hot caatinga/mata_atlantica).
- **Alvo**: `houve_foco_d1` (1 se houve foco no município no dia D+1).





## Resultados (teste, último semestre de 2024)

| modelo | AUC | AP | precisão | recall | F1 |
|--------|-----|-----|----------|--------|-----|
| random_forest balanceado     | **0.777** | 0.039 | 0.046 | **0.339** | 0.081 |
| random_forest sem balance.   | 0.787 | 0.058 | 0.000 | 0.000 | 0.000 |
| lightgbm balanceado          | 0.731 | 0.031 | 0.054 | 0.083 | 0.066 |
| lightgbm sem balance.        | 0.798 | 0.044 | 0.065 | 0.010 | 0.017 |

Random Forest balanceado é o modelo de produção: AUC > 0.7 (acima do critério) e recall não-degenerado.

Sem balanceamento ambos modelos colapsam para sempre prever 0 (recall ~0) — em problema com 0.54% de positivos, balanceamento é obrigatório.

## API

| Endpoint | Descrição |
|----------|-----------|
| `GET /municipios` | lista de municípios com centroide |
| `GET /municipios/geojson` | polígonos simplificados (110KB) |
| `GET /datas` | datas disponíveis |
| `GET /previsao/{data}` | probabilidades D+1 para todos municípios |
| `GET /previsao/{data}/comparacao` | previsto vs real (TP/FP/FN/TN) |
| `POST /previsao/simulacao` | recalcula com deltas de meteorologia |
| `GET /previsao/futuro?dias=N&data_base=...` | D+1 a D+N por persistência climática |
| `GET /modelo/relatorio` | métricas e importâncias completas |

## Frontend

4 abas: **previsão**, **comparação**, **futuro**, **relatório**. Mapa Leaflet à esquerda, painel contextual à direita.

- **Previsão**: clique num município → ver probabilidade, ajustar sliders de temperatura/umidade/chuva/vento e recalcular.
- **Comparação**: layer com 4 cores (previsto baixo s/ foco verde claro, previsto alto s/ foco amarelo, previsto baixo c/ foco vermelho, previsto alto c/ foco vermelho escuro) + métricas.
- **Futuro**: 1-7 dias com persistência climática, carrossel de dias.
- **Relatório**: métricas tabuladas, matriz de confusão, importância de features (recharts).


## Limites conhecidos

- **Cobertura esparsa de estações INMET** (~8 efetivas no RN). Interpolação suaviza variação intra-municipal.
- **Desbalanceamento severo** (0.54% de positivos). Precisão baixa é estrutural; o modelo presta mais como ferramenta de priorização do que como gatilho binário.
- **Persistência climática** para previsões D+2 em diante. Sem entrada de previsão numérica de tempo, confiabilidade cai rápido.
- **Bioma aproximado** por longitude. Ideal: sobrepor com shapefile oficial IBGE/IBAMA de biomas.
- **Sem OSM** na entrega atual (densidade de estradas, cobertura natural). O script de download existe mas o ETL trata como opcional.

## Reprodução completa

```bash
make tudo   # baixa + processa + treina (cerca de 10 min em máquina mediana)
make servir &
cd frontend && npm install && npm run dev
```
