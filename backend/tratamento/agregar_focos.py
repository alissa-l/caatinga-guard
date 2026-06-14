# Agrega focos do INPE por municipio-dia. Faz spatial join lat/lon -> poligono
# IBGE (em vez de confiar no campo municipio do CSV, que tem grafia diferente).

import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from backend import configuracao as cfg


def carregar_focos():
    """Concatena todos os CSVs anuais INPE das UFs alvo."""
    blocos = []
    raiz = os.path.join(cfg.DIR_DADOS_BRUTOS, "inpe")
    for uf in cfg.UFS_ALVO:
        for csv in sorted(glob.glob(os.path.join(raiz, uf, "*.csv"))):
            df = pd.read_csv(csv)
            df["uf_origem"] = uf
            blocos.append(df)
    if not blocos:
        return pd.DataFrame()
    df = pd.concat(blocos, ignore_index=True)
    df["data_pas"] = pd.to_datetime(df["data_pas"], errors="coerce")
    df = df.dropna(subset=["data_pas", "lat", "lon"])
    return df


def juntar_com_municipios(df_focos, gdf_mun):
    """Spatial join: cada foco vira (codigo_ibge, data, foco_id)."""
    gdf_focos = gpd.GeoDataFrame(
        df_focos.copy(),
        geometry=[Point(xy) for xy in zip(df_focos.lon, df_focos.lat)],
        crs=4326,
    )
    gdf_mun = gdf_mun.to_crs(4326)
    joined = gpd.sjoin(gdf_focos, gdf_mun[["CD_MUN", "geometry"]], how="inner", predicate="within")
    joined["data"] = joined["data_pas"].dt.normalize()
    return joined[["CD_MUN", "data", "foco_id"]].rename(columns={"CD_MUN": "codigo_ibge"})


def agregar_municipio_dia(df_focos_mun):
    g = df_focos_mun.groupby(["codigo_ibge", "data"]).size().reset_index(name="n_focos")
    return g
