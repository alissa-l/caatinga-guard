# Orquestra o pipeline ETL completo.
# Entradas: dados/brutos/{inmet,inpe,ibge}/
# Saidas:   dados/processados/fato_municipio_dia.parquet
#           dados/banco.sqlite (tabela fato_municipio_dia)

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from backend import configuracao as cfg
from backend.tratamento.parsing_inmet import montar_meteo_diaria
from backend.tratamento.calcular_fwi import calcular_fwi_serie
from backend.tratamento import features as ft


# bounding box continental do RN (com folga para PB e CE)
BBOX = {"lat_min": -8.5, "lat_max": -4.2, "lon_min": -41.5, "lon_max": -34.5}


def carregar_municipios():
    blocos = []
    for uf in cfg.UFS_ALVO:
        shp = os.path.join(cfg.DIR_DADOS_BRUTOS, "ibge", uf, f"{uf}_Municipios_2024.shp")
        g = gpd.read_file(shp).to_crs(4326)
        # CRS metrico para centroide preciso, depois volta pra 4326
        cents = g.to_crs(31984).geometry.centroid.to_crs(4326)
        g["centro_lat"] = cents.y.values
        g["centro_lon"] = cents.x.values
        g["uf"] = uf
        blocos.append(g)
    return pd.concat(blocos, ignore_index=True)


def preparar_meteo_estacoes(meteo):
    # filtra bbox continental
    m = meteo[
        meteo.lat.between(BBOX["lat_min"], BBOX["lat_max"])
        & meteo.lon.between(BBOX["lon_min"], BBOX["lon_max"])
    ].copy()
    print(f"  estacoes apos filtro bbox: {m.codigo_wmo.nunique()}")

    # garante grade diaria continua por estacao (preenche dias faltantes com NaN)
    m["data"] = pd.to_datetime(m["data"])
    todas_datas = pd.date_range(
        f"{cfg.ANO_INICIO}-01-01", f"{cfg.ANO_FIM}-12-31", freq="D"
    )
    expandidas = []
    estacoes_meta = m.groupby("codigo_wmo").agg(
        lat=("lat", "first"),
        lon=("lon", "first"),
        nome_estacao=("nome_estacao", "first"),
        uf=("uf", "first"),
    ).reset_index()
    for wmo, sub in m.groupby("codigo_wmo"):
        sub = sub.set_index("data").reindex(todas_datas)
        sub["codigo_wmo"] = wmo
        sub["data"] = sub.index
        expandidas.append(sub.reset_index(drop=True))
    m = pd.concat(expandidas, ignore_index=True)
    # ffill curto (3 dias) por estacao
    cols_num = ["temp_media", "temp_max", "temp_min", "umid_media", "chuva_dia", "vento_medio", "rad_media"]
    m[cols_num] = (
        m.groupby("codigo_wmo")[cols_num].transform(lambda s: s.ffill(limit=3))
    )
    # chuva NaN restante -> 0 (assumir dia sem chuva)
    m["chuva_dia"] = m["chuva_dia"].fillna(0)

    # rejunta lat/lon que se perderam no reindex
    m = m.drop(columns=["lat", "lon", "nome_estacao", "uf"], errors="ignore").merge(
        estacoes_meta, on="codigo_wmo", how="left"
    )
    return m


def calcular_fwi_por_estacao(meteo):
    blocos = []
    for wmo, sub in meteo.groupby("codigo_wmo"):
        sub = calcular_fwi_serie(sub.sort_values("data"))
        blocos.append(sub)
    return pd.concat(blocos, ignore_index=True)


def _haversine_km_vec(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def interpolar_para_municipios(meteo, municipios, k=3):
    """IDW vetorizado: para cada municipio, pega k estacoes mais proximas
    e para cada dia faz media ponderada das que tiverem valor."""
    estacoes = (
        meteo[["codigo_wmo", "lat", "lon"]].dropna().drop_duplicates("codigo_wmo")
    ).reset_index(drop=True)

    # distancia muni x estacao
    lat_mun = municipios["centro_lat"].values[:, None]
    lon_mun = municipios["centro_lon"].values[:, None]
    lat_est = estacoes["lat"].values[None, :]
    lon_est = estacoes["lon"].values[None, :]
    D = _haversine_km_vec(lat_mun, lon_mun, lat_est, lon_est)  # (nm, ne)

    # indices das k estacoes mais proximas por municipio
    idx_k = np.argsort(D, axis=1)[:, :k]                       # (nm, k)
    dist_k = np.take_along_axis(D, idx_k, axis=1)              # (nm, k)
    pesos_brutos = 1.0 / (dist_k ** 2 + 1e-6)                  # (nm, k)

    variaveis = ["temp_media", "temp_max", "temp_min", "umid_media",
                 "chuva_dia", "vento_medio", "rad_media",
                 "ffmc", "dmc", "dc", "isi", "bui", "fwi"]
    pivots = {
        v: meteo.pivot_table(index="data", columns="codigo_wmo",
                             values=v, aggfunc="mean")
        for v in variaveis
    }
    # garante colunas na mesma ordem das estacoes
    nomes_est = estacoes["codigo_wmo"].tolist()
    for v in variaveis:
        for n in nomes_est:
            if n not in pivots[v].columns:
                pivots[v][n] = np.nan
        pivots[v] = pivots[v][nomes_est].sort_index()

    datas = pivots["temp_media"].index
    nm = len(municipios)
    nd = len(datas)

    saidas = {v: np.full((nm, nd), np.nan, dtype=np.float32) for v in variaveis}
    cods_ibge = municipios["CD_MUN"].astype(str).values

    for i in range(nm):
        cols_idx = idx_k[i]
        w = pesos_brutos[i]
        for v in variaveis:
            X = pivots[v].values[:, cols_idx]               # (nd, k)
            mask = ~np.isnan(X)
            Wb = np.where(mask, w[None, :], 0.0)
            soma_w = Wb.sum(axis=1)
            valor = np.nansum(np.where(mask, X * w[None, :], 0.0), axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                valor = np.where(soma_w > 0, valor / soma_w, np.nan)
            saidas[v][i] = valor

    # serializa em long
    rows = []
    for i in range(nm):
        d = {
            "codigo_ibge": cods_ibge[i],
            "data": datas,
        }
        for v in variaveis:
            d[v] = saidas[v][i]
        rows.append(pd.DataFrame(d))
    out = pd.concat(rows, ignore_index=True)
    return out


def carregar_focos_por_municipio(municipios):
    import glob
    blocos = []
    raiz = os.path.join(cfg.DIR_DADOS_BRUTOS, "inpe")
    for uf in cfg.UFS_ALVO:
        for csv in sorted(glob.glob(os.path.join(raiz, uf, "*.csv"))):
            df = pd.read_csv(csv)
            blocos.append(df)
    if not blocos:
        return pd.DataFrame(columns=["codigo_ibge", "data", "n_focos"])
    df = pd.concat(blocos, ignore_index=True)
    df["data_pas"] = pd.to_datetime(df["data_pas"], errors="coerce")
    df = df.dropna(subset=["data_pas", "lat", "lon"])

    gdf_focos = gpd.GeoDataFrame(
        df, geometry=[Point(xy) for xy in zip(df.lon, df.lat)], crs=4326
    )
    gdf_mun = gpd.GeoDataFrame(
        municipios[["CD_MUN", "geometry"]].copy(), crs=municipios.crs or 4326
    ).to_crs(4326)
    joined = gpd.sjoin(gdf_focos, gdf_mun, how="inner", predicate="within")
    joined["data"] = joined["data_pas"].dt.normalize()
    contagem = (
        joined.groupby([joined["CD_MUN"].astype(str), "data"])
        .size()
        .reset_index(name="n_focos")
        .rename(columns={"CD_MUN": "codigo_ibge"})
    )
    return contagem


def _bioma_oficial_por_municipio(municipios):
    """sjoin do centroide de cada municipio com shapefile oficial IBGE
    (Biomas 1:250.000). Devolve dict codigo_ibge -> bioma normalizado.
    Se o shapefile nao foi baixado, devolve None (caller cai no fallback)."""
    shp = os.path.join(cfg.DIR_DADOS_BRUTOS, "ibge", "biomas", "lm_bioma_250.shp")
    if not os.path.exists(shp):
        return None
    biomas = gpd.read_file(shp).to_crs(4326)
    pontos = gpd.GeoDataFrame(
        {"codigo_ibge": municipios["CD_MUN"].astype(str).values},
        geometry=[Point(lon, lat) for lon, lat in zip(municipios["centro_lon"], municipios["centro_lat"])],
        crs=4326,
    )
    j = gpd.sjoin(pontos, biomas[["Bioma", "geometry"]], how="left", predicate="within")
    # se um centroide cair fora de tudo (raro, ilha de costa), busca o bioma
    # mais proximo. Aproximacao boa o suficiente para 167 municipios.
    falta = j["Bioma"].isna()
    if falta.any():
        from shapely.ops import nearest_points
        for idx in pontos[falta].index:
            ponto = pontos.geometry.iloc[idx]
            distancias = biomas.geometry.distance(ponto)
            j.loc[idx, "Bioma"] = biomas.loc[distancias.idxmin(), "Bioma"]
    mapa = {
        "Caatinga": "caatinga",
        "Mata Atlântica": "mata_atlantica",
        "Cerrado": "cerrado",
        "Amazônia": "amazonia",
        "Pantanal": "pantanal",
        "Pampa": "pampa",
    }
    j["bioma"] = j["Bioma"].map(mapa).fillna("caatinga")
    return dict(zip(j["codigo_ibge"].values, j["bioma"].values))


def _carregar_osm():
    """Le parquet com features OSM se ja foi gerado. Senao devolve None."""
    pq = os.path.join(cfg.DIR_DADOS_PROCESSADOS, "osm_municipios.parquet")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    return None


def features_estaticas(municipios):
    df = pd.DataFrame({
        "codigo_ibge": municipios["CD_MUN"].astype(str).values,
        "nome_municipio": municipios["NM_MUN"].values,
        "uf": municipios["uf"].values,
        "area_km2": municipios["AREA_KM2"].astype(float).values,
        "centro_lat": municipios["centro_lat"].values,
        "centro_lon": municipios["centro_lon"].values,
    })
    df["distancia_litoral_km"] = ft.distancia_litoral_km(df["centro_lat"], df["centro_lon"])
    # bioma: sjoin com shapefile oficial IBGE 1:250000 (rode
    # `make baixar-bioma`). Fallback para aproximacao por longitude se o
    # shapefile nao tiver sido baixado.
    mapa_bioma = _bioma_oficial_por_municipio(municipios)
    if mapa_bioma is not None:
        df["bioma"] = df["codigo_ibge"].map(mapa_bioma).fillna("caatinga")
        print(f"  bioma oficial (sjoin IBGE): {df['bioma'].value_counts().to_dict()}")
    else:
        df["bioma"] = np.where(df["centro_lon"] > -35.2, "mata_atlantica", "caatinga")
        print("  bioma aproximado por longitude (shapefile oficial nao encontrado)")
    return df


def montar():
    cfg.garantir_diretorios()
    print("[1/6] lendo municipios IBGE")
    municipios = carregar_municipios()
    print(f"  {len(municipios)} municipios")

    print("[2/6] parseando INMET")
    meteo = montar_meteo_diaria()
    print(f"  meteo bruta: {meteo.shape}")
    meteo = preparar_meteo_estacoes(meteo)
    print(f"  meteo apos preparacao: {meteo.shape}")

    print("[3/6] calculando FWI por estacao")
    meteo = calcular_fwi_por_estacao(meteo)
    print(f"  FWI medio: {meteo['fwi'].mean():.2f}, max: {meteo['fwi'].max():.2f}")

    print("[4/6] interpolando para municipios (IDW k=3)")
    meteo_mun = interpolar_para_municipios(meteo, municipios, k=3)
    print(f"  linhas meteo municipal: {len(meteo_mun)}")

    print("[5/6] carregando focos INPE e fazendo spatial join")
    focos = carregar_focos_por_municipio(municipios)
    print(f"  focos agregados: {len(focos)} pares municipio-dia com foco")

    print("[6/6] montando fato_municipio_dia")
    estaticas = features_estaticas(municipios)

    osm = _carregar_osm()
    if osm is not None:
        estaticas = estaticas.merge(osm, on="codigo_ibge", how="left")
        for c in ["osm_estradas_km", "osm_estradas_principais_km", "osm_pasto_km2", "osm_cultivo_km2"]:
            if c in estaticas.columns:
                estaticas[c] = estaticas[c].fillna(0.0)
        print(f"  features OSM aplicadas: {[c for c in estaticas.columns if c.startswith('osm_')]}")
    else:
        # marcadores zerados para o modelo nao precisar de logica especial
        for c in ["osm_estradas_km", "osm_estradas_principais_km", "osm_pasto_km2", "osm_cultivo_km2"]:
            estaticas[c] = 0.0
        print("  parquet OSM nao encontrado, features zeradas (rode make baixar-osm + processar-osm)")

    df = meteo_mun.merge(estaticas, on="codigo_ibge", how="left")
    df = df.merge(focos, on=["codigo_ibge", "data"], how="left")
    df["n_focos"] = df["n_focos"].fillna(0).astype("int16")

    df = ft.adicionar_lags_focos(df)
    df = ft.adicionar_chuva_acumulada(df)
    df = ft.adicionar_dias_sem_chuva(df)
    df = ft.adicionar_focos_acumulados(df, janelas=(30, 90))
    df = ft.adicionar_sazonalidade(df)
    df = ft.adicionar_alvo(df)

    # taxas historicas: mean encoding com corte de treino vindo do config.
    # tem que casar com o split de treinar.py.
    datas_ordenadas = np.sort(df["data"].unique())
    corte = datas_ordenadas[int(cfg.FRAC_TREINO_HISTORICO * len(datas_ordenadas))]
    df = ft.adicionar_taxas_historicas(df, corte)
    print(f"  corte de historico: {pd.Timestamp(corte).date()}")

    # tipos otimizados
    df["uf"] = df["uf"].astype("category")
    df["bioma"] = df["bioma"].astype("category")
    df["codigo_ibge"] = df["codigo_ibge"].astype("category")

    # A ultima linha de cada municipio fica com alvo 0 (nao existe D+1 para ela),
    # mas e mantida de proposito: o modo "futuro" da API a reaproveita.

    print(f"  shape final: {df.shape}")
    print(f"  positivos (houve_foco_d1=1): {df['houve_foco_d1'].sum()} ({100*df['houve_foco_d1'].mean():.3f}%)")

    out_pq = os.path.join(cfg.DIR_DADOS_PROCESSADOS, "fato_municipio_dia.parquet")
    df.to_parquet(out_pq, index=False)
    print(f"salvo: {out_pq}")

    salvar_sqlite(df)
    return df


def salvar_sqlite(df):
    con = sqlite3.connect(cfg.CAMINHO_BANCO)
    df_sql = df.copy()
    df_sql["data"] = df_sql["data"].dt.strftime("%Y-%m-%d")
    for c in df_sql.select_dtypes(include="category").columns:
        df_sql[c] = df_sql[c].astype(str)
    df_sql.to_sql("fato_municipio_dia", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS idx_mun_data ON fato_municipio_dia(codigo_ibge, data)")
    con.commit()
    con.close()
    print(f"salvo: {cfg.CAMINHO_BANCO}")


if __name__ == "__main__":
    sys.exit(0 if montar() is not None else 1)
