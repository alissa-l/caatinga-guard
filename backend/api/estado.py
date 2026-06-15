
import os
import json
import pandas as pd
import geopandas as gpd

from backend import configuracao as cfg


class Estado:
    def __init__(self):
        self.df = None
        self.geojson = None
        self.municipios = None
        self.datas_disponiveis = None

    def carregar(self):
        pq = os.path.join(cfg.DIR_DADOS_PROCESSADOS, "fato_municipio_dia.parquet")
        df = pd.read_parquet(pq)
        df["data"] = pd.to_datetime(df["data"])
        # one-hot do bioma para casar com FEATURES do modelo
        df["bioma_caatinga"] = (df["bioma"].astype(str) == "caatinga").astype("int8")
        df["bioma_mata_atlantica"] = (df["bioma"].astype(str) == "mata_atlantica").astype("int8")
        df["codigo_ibge"] = df["codigo_ibge"].astype(str)
        self.df = df
        self.datas_disponiveis = sorted(df["data"].dt.strftime("%Y-%m-%d").unique().tolist())

        municipios = (
            df.groupby("codigo_ibge")
            .agg(
                nome=("nome_municipio", "first"),
                uf=("uf", "first"),
                centro_lat=("centro_lat", "first"),
                centro_lon=("centro_lon", "first"),
                area_km2=("area_km2", "first"),
            )
            .reset_index()
        )
        municipios["uf"] = municipios["uf"].astype(str)
        municipios["nome"] = municipios["nome"].astype(str)
        self.municipios = municipios

        # construir geojson simplificado
        geo_path = os.path.join(cfg.DIR_DADOS_PROCESSADOS, "municipios.geojson")
        if not os.path.exists(geo_path):
            gframes = []
            for uf in cfg.UFS_ALVO:
                shp = os.path.join(cfg.DIR_DADOS_BRUTOS, "ibge", uf, f"{uf}_Municipios_2024.shp")
                gframes.append(gpd.read_file(shp).to_crs(4326))
            g = gframes[0] if len(gframes) == 1 else gpd.GeoDataFrame(pd.concat(gframes))
            g["geometry"] = g["geometry"].simplify(0.005, preserve_topology=True)
            g["codigo_ibge"] = g["CD_MUN"].astype(str)
            g = g[["codigo_ibge", "NM_MUN", "geometry"]].rename(columns={"NM_MUN": "nome"})
            g.to_file(geo_path, driver="GeoJSON")
        with open(geo_path) as f:
            self.geojson = json.load(f)


estado = Estado()
