# Baixa o shapefile oficial de biomas brasileiros do IBGE (1:250.000).
# Pequeno (~30MB) e estavel. Substitui a aproximacao por longitude no ETL.

import os
import sys
import zipfile

from backend import configuracao as cfg
from backend.coleta._http import sessao_com_retry


SESSAO = sessao_com_retry()


def baixar(url, destino):
    if os.path.exists(destino) and os.path.getsize(destino) > 1_000_000:
        print(f"  ja existe: {os.path.basename(destino)}")
        return True
    print(f"  baixando {url}")
    r = SESSAO.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(destino, "wb") as f:
        for chunk in r.iter_content(chunk_size=128 * 1024):
            if chunk:
                f.write(chunk)
    return True


def main():
    cfg.garantir_diretorios()
    raiz = os.path.join(cfg.DIR_DADOS_BRUTOS, "ibge", "biomas")
    os.makedirs(raiz, exist_ok=True)
    zip_dest = os.path.join(raiz, "Biomas_250mil.zip")
    try:
        baixar(cfg.URL_IBGE_BIOMAS, zip_dest)
    except Exception as e:
        print(f"  falha: {e}")
        print("  etapa opcional, seguindo sem.")
        return 0

    # extrai shapefile se ainda nao
    shps = [f for f in os.listdir(raiz) if f.lower().endswith(".shp")]
    if not shps:
        print(f"  extraindo {os.path.basename(zip_dest)}")
        with zipfile.ZipFile(zip_dest) as z:
            z.extractall(raiz)
    shps = sorted([f for f in os.listdir(raiz) if f.lower().endswith(".shp")])
    print(f"  shapefiles: {shps}")


if __name__ == "__main__":
    sys.exit(main() or 0)
