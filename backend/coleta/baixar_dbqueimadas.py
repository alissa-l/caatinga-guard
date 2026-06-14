# Baixa focos de calor do INPE/DBQueimadas, satelite de referencia,
# por UF e por ano. Usa o servico anual EstadosBr_sat_ref que ja vem
# filtrado pelo satelite de referencia. Para anos parciais (ano corrente)
# poderiamos cair no arquivo mensal Brasil, mas no escopo do projeto
# ficamos so com anos completos.

import os
import sys
import zipfile
import requests

from backend import configuracao as cfg
from backend.coleta._http import sessao_com_retry


SESSAO = sessao_com_retry()


def baixar(url, destino):
    if os.path.exists(destino) and os.path.getsize(destino) > 0:
        return True
    print(f"  baixando {url}")
    r = SESSAO.get(url, stream=True, timeout=180)
    if r.status_code == 404:
        print(f"  nao encontrado (404)")
        return False
    r.raise_for_status()
    with open(destino, "wb") as f:
        for chunk in r.iter_content(chunk_size=64 * 1024):
            if chunk:
                f.write(chunk)
    return True


def main():
    cfg.garantir_diretorios()
    raiz_inpe = os.path.join(cfg.DIR_DADOS_BRUTOS, "inpe")
    os.makedirs(raiz_inpe, exist_ok=True)

    ufs = list(cfg.UFS_ALVO)
    if cfg.INCLUIR_ESTACOES_VIZINHAS:
        # focos das UFs vizinhas nao entram no alvo, mas podem ajudar
        # em analises exploratorias. nao baixamos por padrao para nao
        # inflar dados/.
        pass

    for uf in ufs:
        print(f"[INPE] {uf}")
        dir_uf = os.path.join(raiz_inpe, uf)
        os.makedirs(dir_uf, exist_ok=True)

        for ano in range(cfg.ANO_INICIO, cfg.ANO_FIM + 1):
            url = cfg.URL_INPE_ANUAL.format(uf=uf, uf_lower=uf.lower(), ano=ano)
            nome_zip = os.path.basename(url)
            zip_dest = os.path.join(dir_uf, nome_zip)

            ok = baixar(url, zip_dest)
            if not ok:
                continue

            # extrai CSV no mesmo diretorio se ainda nao tem
            csvs = [f for f in os.listdir(dir_uf) if f.startswith(f"focos_br_{uf.lower()}_ref_{ano}") and f.endswith(".csv")]
            if not csvs:
                with zipfile.ZipFile(zip_dest) as z:
                    z.extractall(dir_uf)

        print(f"  arquivos: {sorted(os.listdir(dir_uf))[:6]}...")


if __name__ == "__main__":
    sys.exit(main())
