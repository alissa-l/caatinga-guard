# Etapa OPCIONAL. Baixa extrato do Nordeste do Geofabrik (pesado, ~150MB).
# Se nao rodar, o ETL deixa as features de OSM como NaN/0 e segue.

import os
import sys
import requests

from backend import configuracao as cfg
from backend.coleta._http import sessao_com_retry


SESSAO = sessao_com_retry()


def baixar(url, destino):
    if os.path.exists(destino) and os.path.getsize(destino) > 10_000_000:
        print(f"  ja existe: {os.path.basename(destino)}")
        return
    print(f"  baixando {url}")
    r = SESSAO.get(url, stream=True, timeout=600)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    baixado = 0
    with open(destino, "wb") as f:
        for chunk in r.iter_content(chunk_size=256 * 1024):
            if chunk:
                f.write(chunk)
                baixado += len(chunk)
                if total:
                    pct = 100 * baixado / total
                    print(f"  {pct:5.1f}% ({baixado // (1024 * 1024)} MB)", end="\r")
    print()


def main():
    cfg.garantir_diretorios()
    raiz = os.path.join(cfg.DIR_DADOS_BRUTOS, "osm")
    os.makedirs(raiz, exist_ok=True)
    dest = os.path.join(raiz, "nordeste-latest.osm.pbf")
    try:
        baixar(cfg.URL_OSM_NORDESTE, dest)
    except requests.RequestException as e:
        print(f"falha no download OSM: {e}")
        print("etapa opcional, seguindo sem.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
