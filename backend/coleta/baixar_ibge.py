# Baixa malhas municipais do IBGE para as UFs configuradas.
# Cada UF vira um diretorio em dados/brutos/ibge/<uf>/ com o shapefile extraido.

import os
import sys
import zipfile
import requests

from backend import configuracao as cfg
from backend.coleta._http import sessao_com_retry


SESSAO = sessao_com_retry()


def baixar_arquivo(url, destino):
    if os.path.exists(destino) and os.path.getsize(destino) > 0:
        print(f"  ja existe: {os.path.basename(destino)}")
        return
    print(f"  baixando {url}")
    r = SESSAO.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    baixado = 0
    with open(destino, "wb") as f:
        for chunk in r.iter_content(chunk_size=64 * 1024):
            if chunk:
                f.write(chunk)
                baixado += len(chunk)
                if total:
                    pct = 100 * baixado / total
                    print(f"  {pct:5.1f}%", end="\r")
    print()


def main():
    cfg.garantir_diretorios()
    raiz_ibge = os.path.join(cfg.DIR_DADOS_BRUTOS, "ibge")
    os.makedirs(raiz_ibge, exist_ok=True)

    for uf in cfg.UFS_ALVO + (cfg.UFS_VIZINHAS if cfg.INCLUIR_ESTACOES_VIZINHAS else []):
        print(f"[IBGE] {uf}")
        dir_uf = os.path.join(raiz_ibge, uf)
        os.makedirs(dir_uf, exist_ok=True)

        url = cfg.URL_IBGE_MUNICIPIOS.format(uf=uf)
        nome_zip = os.path.basename(url)
        zip_dest = os.path.join(dir_uf, nome_zip)

        try:
            baixar_arquivo(url, zip_dest)
        except requests.HTTPError as e:
            print(f"  falhou: {e}")
            continue

        # extrair se ainda nao foi
        marcadores = [f for f in os.listdir(dir_uf) if f.endswith(".shp")]
        if not marcadores:
            print(f"  extraindo {nome_zip}")
            with zipfile.ZipFile(zip_dest) as z:
                z.extractall(dir_uf)

        shps = [f for f in os.listdir(dir_uf) if f.endswith(".shp")]
        print(f"  shapefiles: {shps}")


if __name__ == "__main__":
    sys.exit(main())
