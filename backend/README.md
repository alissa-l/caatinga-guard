# Previsor de Incêndios Florestais — Rio Grande do Norte

Sistema de previsão de risco de incêndio florestal por município, para o dia seguinte (D+1), baseado em Machine Learning treinado do zero sobre dados meteorológicos do INMET, focos de calor do INPE e malhas territoriais do IBGE.

Escopo atual: **167 municípios do Rio Grande do Norte, janela histórica 2020–2024**. A arquitetura permite escalar para outras UFs editando apenas `backend/configuracao.py`.

Trabalho da disciplina de Introdução à Inteligência Artificial.

## Estrutura



## Como rodar



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



## API


 `GET /municipios`  lista de municípios com centroide 
 `GET /municipios/geojson` polígonos simplificados (110KB) 
 `GET /datas` datas disponíveis 
 `GET /previsao/{data}` probabilidades D+1 para todos municípios 
 `GET /previsao/{data}/comparacao` previsto vs real (TP/FP/FN/TN) 
 `POST /previsao/simulacao` recalcula com deltas de meteorologia 
 `GET /previsao/futuro?dias=N&data_base=...` D+1 a D+N por persistência climática 
 `GET /modelo/relatorio` métricas e importâncias completas 



## Frontend



## Limites conhecidos



## Reprodução completa