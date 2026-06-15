# Script pra processar os dados do OSM do Nordeste.
# A ideia é pegar a densidade das estradas e o uso do solo (pasto, plantação, etc) por município.
# Nota mental: se o arquivo PBF não tiver baixado, o código finge que nada aconteceu e segue a vida,
# aí o dataset lá na frente fica todo zerado (socorro, lembrar de baixar o PBF antes!).
#
# Tô usando a lib osmium porque foi a única que não fritou a memória do meu PC com esses arquivos pesados.
# Como funciona a estratégia:
#  1. Primeiro pega as coordenadas (lat/lon) de todos os pontinhos (nodes).
#  2. Depois pega as vias (ways) que são ruas ou área de pasto/cultivo, monta a forma geométrica
#     e faz um join espacial com os municípios.
#
# No fim, cospe um arquivo parquet bonitinho com o código do IBGE e os km/km2 das coisas.

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Polygon, Point

from backend import configuracao as cfg


PBF = os.path.join(cfg.DIR_DADOS_BRUTOS, "osm", "nordeste-latest.osm.pbf")
SAIDA_PQ = os.path.join(cfg.DIR_DADOS_PROCESSADOS, "osm_municipios.parquet")


def _existe_osmium():
    try:
        import osmium  # noqa
        return True
    except ImportError:
        return False


def _carregar_municipios_rn_pb_ce():
    """O spatial join (sjoin) só funciona se tiver polígonos.
    Aqui eu pego os shapefiles das UFs que a gente quer analisar e jogo tudo num GeoDataFrame
    já convertendo pro CRS 4326 pra não dar dor de cabeça depois com as projeções."""
    blocos = []
    for uf in cfg.UFS_ALVO:
        shp = os.path.join(cfg.DIR_DADOS_BRUTOS, "ibge", uf, f"{uf}_Municipios_2024.shp")
        if not os.path.exists(shp):
            continue
        g = gpd.read_file(shp).to_crs(4326)
        blocos.append(g[["CD_MUN", "geometry"]].assign(uf=uf))
    return gpd.GeoDataFrame(pd.concat(blocos, ignore_index=True), crs=4326)


def _bbox_municipios(municipios):
    return municipios.total_bounds


import osmium


class _ColetorOSM(osmium.SimpleHandler):
    """Classe do osmium pra ir filtrando as coisas que importam e salvar na memória.
    Cuidado: pro PBF do Nordeste que tem uns 400MB, isso aqui puxa uns 2GB de RAM...
    Melhor fechar o Chrome antes de rodar."""

    def __init__(self, bbox):
        super().__init__()
        self.minx, self.miny, self.maxx, self.maxy = bbox
        self.estradas = []
        self.poligonos_landuse = []

    def _no_bbox(self, lon, lat):
        return self.minx <= lon <= self.maxx and self.miny <= lat <= self.maxy

    def way(self, w):
        tags = dict(w.tags)
        if "highway" in tags:
            try:
                pts = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
                if len(pts) < 2:
                    return
                if not any(self._no_bbox(lon, lat) for lon, lat in pts):
                    return
                self.estradas.append({
                    "geometry": LineString(pts),
                    "tipo": tags.get("highway"),
                })
            except Exception:
                pass
        elif tags.get("landuse") in ("pasture", "farmland", "meadow"):
            try:
                pts = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
                if len(pts) < 3:
                    return
                if pts[0] != pts[-1]:
                    pts.append(pts[0])
                if not any(self._no_bbox(lon, lat) for lon, lat in pts):
                    return
                self.poligonos_landuse.append({
                    "geometry": Polygon(pts),
                    "uso": tags["landuse"],
                })
            except Exception:
                pass


def _processar():
    if not _existe_osmium():
        print("  osmium nao instalado - pulando OSM (precisa instalar, galera)")
        return None
    if not os.path.exists(PBF):
        print(f"  PBF nao encontrado em {PBF} - pulando OSM (esqueceu de baixar de novo?)")
        return None

    municipios = _carregar_municipios_rn_pb_ce()
    if municipios.empty:
        print("  shapefile municipios nao encontrado - pulando OSM")
        return None
    bbox = _bbox_municipios(municipios)
    print(f"  bbox municipios: lon[{bbox[0]:.2f}, {bbox[2]:.2f}] lat[{bbox[1]:.2f}, {bbox[3]:.2f}]")

    handler = _ColetorOSM(bbox)
    print(f"  processando PBF {os.path.getsize(PBF)/1024/1024:.1f} MB... (vai demorar um pouquinho)")
    handler.apply_file(PBF, locations=True)
    print(f"  vias extraidas: {len(handler.estradas)}, poligonos landuse: {len(handler.poligonos_landuse)}")

    return _agregar_por_municipio(handler, municipios)


def _agregar_por_municipio(handler, municipios):
    """Hora da verdade: juntar as linhas das estradas e os polígonos de terra com o mapa dos municípios."""
    cods = municipios["CD_MUN"].astype(str).reset_index(drop=True)
    saida = pd.DataFrame({"codigo_ibge": cods})

    # --- PARTE DAS ESTRADAS ---
    estradas = gpd.GeoDataFrame(handler.estradas, crs=4326) if handler.estradas else None
    if estradas is not None and not estradas.empty:
        # Convertendo pra um sistema métrico pra conseguir calcular o tamanho em km sem dar erro
        estradas_m = estradas.to_crs(31984)
        estradas_m["len_km"] = estradas_m.length / 1000.0
        # Spatial join pra "cortar" as estradas bem na fronteira dos municípios (deu trabalho entender isso)
        municipios_m = municipios.to_crs(31984)
        cortes = gpd.overlay(
            estradas_m[["geometry", "len_km", "tipo"]],
            municipios_m[["CD_MUN", "geometry"]],
            how="intersection",
            keep_geom_type=False,
        )
        cortes["len_km"] = cortes.geometry.length / 1000.0
        cortes["principal"] = cortes["tipo"].isin([
            "motorway", "trunk", "primary", "secondary", "tertiary",
            "motorway_link", "trunk_link", "primary_link", "secondary_link",
        ])
        agg = (
            cortes.groupby(cortes["CD_MUN"].astype(str))
            .agg(
                osm_estradas_km=("len_km", "sum"),
                osm_estradas_principais_km=("len_km", lambda s: s[cortes.loc[s.index, "principal"]].sum()),
            )
            .reset_index()
            .rename(columns={"CD_MUN": "codigo_ibge"})
        )
        saida = saida.merge(agg, on="codigo_ibge", how="left")
    else:
        saida["osm_estradas_km"] = 0.0
        saida["osm_estradas_principais_km"] = 0.0

    # --- PARTE DO USO DO SOLO (Pasto e Cultivo) ---
    pol = gpd.GeoDataFrame(handler.poligonos_landuse, crs=4326) if handler.poligonos_landuse else None
    if pol is not None and not pol.empty:
        pol_m = pol.to_crs(31984)
        municipios_m = municipios.to_crs(31984)
        cortes = gpd.overlay(
            pol_m[["geometry", "uso"]],
            municipios_m[["CD_MUN", "geometry"]],
            how="intersection",
            keep_geom_type=False,
        )
        cortes["area_km2"] = cortes.geometry.area / 1e6
        cortes["pasto"] = cortes["uso"].isin(["pasture", "meadow"]).astype(float)
        cortes["cultivo"] = (cortes["uso"] == "farmland").astype(float)
        cortes["pasto_km2"] = cortes["area_km2"] * cortes["pasto"]
        cortes["cultivo_km2"] = cortes["area_km2"] * cortes["cultivo"]
        agg = (
            cortes.groupby(cortes["CD_MUN"].astype(str))
            .agg(osm_pasto_km2=("pasto_km2", "sum"), osm_cultivo_km2=("cultivo_km2", "sum"))
            .reset_index()
            .rename(columns={"CD_MUN": "codigo_ibge"})
        )
        saida = saida.merge(agg, on="codigo_ibge", how="left")
    else:
        saida["osm_pasto_km2"] = 0.0
        saida["osm_cultivo_km2"] = 0.0

    for c in ["osm_estradas_km", "osm_estradas_principais_km", "osm_pasto_km2", "osm_cultivo_km2"]:
        saida[c] = saida[c].fillna(0.0).astype("float32")

    return saida


def main():
    cfg.garantir_diretorios()
    df = _processar()
    if df is None:
        return 1
    df.to_parquet(SAIDA_PQ, index=False)
    print(f"  salvo: {SAIDA_PQ} ({len(df)} municipios)")
    print(df.describe().T[["mean", "std", "max"]].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
