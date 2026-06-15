# Interpolacao IDW da meteorologia para o centroide de cada municipio.
# Pega 3-5 estacoes mais proximas, pesa por 1/distancia^2.

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from math import radians, sin, cos, asin, sqrt

from backend import configuracao as cfg


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))


def carregar_municipios():
    """Le shapefiles das UFs alvo, calcula centroide e devolve geodf."""
    blocos = []
    for uf in cfg.UFS_ALVO:
        shp = os.path.join(cfg.DIR_DADOS_BRUTOS, "ibge", uf, f"{uf}_Municipios_2024.shp")
        if not os.path.exists(shp):
            continue
        g = gpd.read_file(shp).to_crs(4326)
        g["centroide"] = g.geometry.centroid
        g["centro_lat"] = g.centroide.y
        g["centro_lon"] = g.centroide.x
        g["uf"] = uf
        blocos.append(g)
    return pd.concat(blocos, ignore_index=True)


def _pesos_idw(distancias_km, k=3, expoente=2.0):
    idx = np.argsort(distancias_km)[:k]
    d = distancias_km[idx]
    # se uma estacao esta praticamente em cima, peso 1 nela
    if d[0] < 0.5:
        pesos = np.zeros(k)
        pesos[0] = 1.0
        return idx, pesos
    w = 1.0 / (d ** expoente)
    return idx, w / w.sum()


def interpolar(df_meteo_estacoes, municipios, k=3):
    """df_meteo_estacoes: temp_media, temp_max, etc + lat, lon por estacao-dia.
    Retorna df municipio-dia com variaveis interpoladas."""
    estacoes = df_meteo_estacoes[["codigo_wmo", "lat", "lon"]].dropna().drop_duplicates("codigo_wmo")
    estacoes = estacoes[(estacoes.lat.between(-90, 90)) & (estacoes.lon.between(-180, 180))]

    # mapa codigo_wmo -> (lat, lon)
    coords = estacoes.set_index("codigo_wmo")[["lat", "lon"]].to_dict("index")

    # pre-calcular, para cada municipio, vizinhos por distancia
    vizinhos_por_mun = {}
    for _, m in municipios.iterrows():
        codigo_ibge = m["CD_MUN"]
        dists = []
        for wmo, c in coords.items():
            d = _haversine_km(m["centro_lat"], m["centro_lon"], c["lat"], c["lon"])
            dists.append((wmo, d))
        dists.sort(key=lambda x: x[1])
        vizinhos_por_mun[codigo_ibge] = dists[:max(k, 5)]

    variaveis = ["temp_media", "temp_max", "temp_min", "umid_media", "chuva_dia", "vento_medio", "rad_media"]

    # indexa estacao-dia para lookup rapido
    pivot = {var: df_meteo_estacoes.pivot_table(index="data", columns="codigo_wmo", values=var, aggfunc="mean") for var in variaveis}

    linhas = []
    datas = pd.date_range(df_meteo_estacoes.data.min(), df_meteo_estacoes.data.max(), freq="D")
    for _, m in municipios.iterrows():
        codigo = m["CD_MUN"]
        viz = vizinhos_por_mun[codigo]
        nomes = [n for n, _ in viz]
        dists = np.array([d for _, d in viz])

        for dt in datas:
            valores = {}
            for var in variaveis:
                p = pivot[var]
                if dt not in p.index:
                    valores[var] = np.nan
                    continue
                linha = p.loc[dt]
                disponiveis = [(n, dists[i]) for i, n in enumerate(nomes) if n in linha.index and not pd.isna(linha[n])]
                if not disponiveis:
                    valores[var] = np.nan
                    continue
                d_disp = np.array([d for _, d in disponiveis[:k]])
                v_disp = np.array([linha[n] for n, _ in disponiveis[:k]])
                _, pesos = _pesos_idw(d_disp, k=len(d_disp))
                valores[var] = float((v_disp[:len(pesos)] * pesos).sum())
            valores["codigo_ibge"] = codigo
            valores["data"] = dt
            linhas.append(valores)
    return pd.DataFrame(linhas)
