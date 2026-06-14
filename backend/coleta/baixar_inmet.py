# Baixa ZIPs anuais de estacoes automaticas do INMET. Cada ZIP tem ~100MB
# e contem CSVs de todas as estacoes do Brasil. A filtragem por UF e feita
# no ETL para evitar duplicar logica em coleta e tratamento.

import os
import sys
import requests

from backend import configuracao as cfg
from backend.coleta._http import sessao_com_retry


SESSAO = sessao_com_retry()


def baixar(url, destino):
    if os.path.exists(destino) and os.path.getsize(destino) > 5_000_000:
        # heuristica: arquivo INMET tem ~100MB. <5MB e download incompleto.
        print(f"  ja existe: {os.path.basename(destino)}")
        return True
    print(f"  baixando {url}")
    r = SESSAO.get(url, stream=True, timeout=300)
    if r.status_code == 404:
        print(f"  nao encontrado (404)")
        return False
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    baixado = 0
    with open(destino, "wb") as f:
        for chunk in r.iter_content(chunk_size=128 * 1024):
            if chunk:
                f.write(chunk)
                baixado += len(chunk)
                if total:
                    pct = 100 * baixado / total
                    print(f"  {pct:5.1f}% ({baixado // (1024 * 1024)} MB)", end="\r")
    print()
    return True


def main():
    cfg.garantir_diretorios()
    raiz = os.path.join(cfg.DIR_DADOS_BRUTOS, "inmet")
    os.makedirs(raiz, exist_ok=True)

    for ano in range(cfg.ANO_INICIO, cfg.ANO_FIM + 1):
        print(f"[INMET] {ano}")
        url = cfg.URL_INMET.format(ano=ano)
        dest = os.path.join(raiz, f"{ano}.zip")
        try:
            baixar(url, dest)
        except requests.HTTPError as e:
            print(f"  falhou: {e}")


if __name__ == "__main__":
    sys.exit(main())
