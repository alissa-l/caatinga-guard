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
- **Dinâmicas** — 14 features que capturam *tendência*, *interação* e *vizinhança espacial*, não só o nível instantâneo:
  - *Tendência de FWI/meteo*: fwi_delta_1d, fwi_delta_3d, fwi_media_3d, fwi_media_7d, fwi_max_7d, isi_media_3d, umid_delta_3d, temp_max_media_3d.
  - *Interações risco × seca*: fwi_x_dias_sem_chuva, temp_max_x_dias_sem_chuva, seca_x_vento.
  - *Vizinhança espacial*: focos_vizinhos_lag_1, focos_vizinhos_acum_3d, fwi_vizinhos (sobre os 6 municípios mais próximos por centroide). Todas usam `shift`/janelas terminando em D — sem vazamento de D+1.
- **Alvo**: `houve_foco_d1` (1 se houve foco no município no dia D+1).





## Resultados (teste, último semestre de 2024)

Com taxa base ~1%, métricas binárias a limiar 0.5 não dizem nada (após calibração quase nada passa de 50%). A leitura é **operacional, por ranking**: dos focos reais de cada dia, quantos caem no top-K de municípios apontados pelo modelo (`recall@K`) e quanto isso supera um sorteio (`lift@K`).

| modelo | AUC | AP | recall@10 | lift@10 |
|--------|-----|-----|-----------|---------|
| **random_forest** (produção) | **0.817** | **0.062** | **0.336** | **5.60×** |
| lightgbm | 0.743 | 0.034 | 0.213 | 3.55× |

O modelo de produção é um Random Forest com **balanceamento por undersampling + ensemble** (`BalancedBaggingRF`): 15 estimadores, cada um treinado em todos os positivos + negativos na razão 3:1, com probabilidades promediadas. Num experimento controlado (`backend/modelo/experimento_balanceamento.py`, que compara `class_weight='balanced'`, `balanced_subsample`, undersampling 1:1 e 3:1, `scale_pos_weight` cheio/raiz, `is_unbalance` e focal loss), o bagging 3:1 foi o melhor por AP de validação e generalizou no teste — AP 0.044→0.062 e recall@10 0.26→0.34 sobre o `class_weight='balanced'` anterior.

Sem qualquer balanceamento ambos os modelos colapsam para prever sempre 0 (recall ~0): com ~1% de positivos, balancear é obrigatório.

Para o caso de uso binário (alertar ou não), o limiar não é fixado em 0.5 — após a calibração quase nada passa de 0.5 e as métricas zeram por construção. Escolhe-se na validação o limiar que maximiza **F2** (prioriza recall) e aplica-se ao teste: o RF de produção opera em ~0.032, capturando **61% dos focos** (183 de 301) ao custo de ~4100 falsos alarmes no semestre. A calibração permanece **isotônica** — num teste direto contra Platt (sigmoid), a isotônica teve Brier marginalmente melhor (0.00951 vs 0.00957).

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
- **Simulação não recalcula features derivadas.** O endpoint `/previsao/simulacao` (sliders de temperatura/umidade/chuva/vento) ajusta apenas a meteorologia bruta; FWI e as features de tendência/vizinhança mantêm o valor do dataset. A probabilidade simulada é, portanto, uma aproximação — reage à meteorologia direta, não ao FWI recomputado.

## Reprodução completa

```bash
make tudo   # baixa + processa + treina (cerca de 10 min em máquina mediana)
make servir &
cd frontend && npm install && npm run dev
```
