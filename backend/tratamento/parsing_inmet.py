# Parser tolerante dos CSVs anuais do INMET. Schema mudou entre anos
# (nomes de colunas com pequenas variacoes, header com 8-9 linhas de
# metadados de estacao). Estrategia: extrair cabecalho com regex e
# agregar valores horarios para o dia.

import os
import re
import io
import zipfile
import pandas as pd
import numpy as np

from backend import configuracao as cfg


def _parse_metadados(linhas_cabecalho):
    meta = {}
    for ln in linhas_cabecalho:
        if ":" not in ln:
            continue
        chave, _, valor = ln.partition(";")
        chave = chave.strip().rstrip(":").upper()
        valor = (valor or "").strip().strip(";")
        meta[chave] = valor
    return meta


def _achar_coluna(colunas, padroes):
    for p in padroes:
        for c in colunas:
            if p in c.upper():
                return c
    return None


def _ler_csv_de_bytes(bruto, nome_arquivo):
    try:
        texto = bruto.decode("latin-1")
    except UnicodeDecodeError:
        texto = bruto.decode("utf-8", errors="ignore")

    linhas = texto.splitlines()
    idx_dados = None
    for i, ln in enumerate(linhas[:20]):
        u = ln.upper()
        if u.startswith("DATA;") or u.startswith("DATA (") or u.startswith("DATA YYYY"):
            idx_dados = i
            break
    if idx_dados is None:
        return None

    meta = _parse_metadados(linhas[:idx_dados])

    csv_texto = "\n".join(linhas[idx_dados:])
    df = pd.read_csv(
        io.StringIO(csv_texto),
        sep=";",
        decimal=",",
        na_values=["-9999", "-9999,0", ""],
        low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]

    col_data = _achar_coluna(df.columns, ["DATA"])
    col_temp = _achar_coluna(df.columns, ["TEMPERATURA DO AR - BULBO SECO", "TEMPERATURA DO AR"])
    col_umid = _achar_coluna(df.columns, ["UMIDADE RELATIVA DO AR, HORARIA", "UMIDADE RELATIVA DO AR"])
    col_chuva = _achar_coluna(df.columns, ["PRECIPITACAO TOTAL", "PRECIPITAÇÃO TOTAL"])
    col_vento = _achar_coluna(df.columns, ["VENTO, VELOCIDADE HORARIA", "VENTO - VELOCIDADE", "VELOCIDADE HORARIA"])
    col_rad = _achar_coluna(df.columns, ["RADIACAO GLOBAL", "RADIAÇÃO GLOBAL"])

    if not col_data or not col_temp:
        return None

    s = df[col_data].astype(str).str.strip()
    data = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=True)

    out = pd.DataFrame({
        "data_hora": data,
        "temp": pd.to_numeric(df[col_temp], errors="coerce"),
        "umid": pd.to_numeric(df[col_umid], errors="coerce") if col_umid else np.nan,
        "chuva": pd.to_numeric(df[col_chuva], errors="coerce") if col_chuva else np.nan,
        "vento": pd.to_numeric(df[col_vento], errors="coerce") if col_vento else np.nan,
        "rad": pd.to_numeric(df[col_rad], errors="coerce") if col_rad else np.nan,
    }).dropna(subset=["data_hora"])

    diario = out.groupby(out["data_hora"].dt.date).agg(
        temp_media=("temp", "mean"),
        temp_max=("temp", "max"),
        temp_min=("temp", "min"),
        umid_media=("umid", "mean"),
        chuva_dia=("chuva", "sum"),
        vento_medio=("vento", "mean"),
        rad_media=("rad", "mean"),
    ).reset_index().rename(columns={"data_hora": "data"})
    diario["data"] = pd.to_datetime(diario["data"])

    diario["codigo_wmo"] = meta.get("CODIGO (WMO)", os.path.basename(nome_arquivo))
    diario["uf"] = meta.get("UF", "")
    diario["nome_estacao"] = meta.get("ESTACAO", "")

    def _to_float(v):
        try:
            return float(v.replace(",", "."))
        except (ValueError, AttributeError):
            return np.nan

    diario["lat"] = _to_float(meta.get("LATITUDE", ""))
    diario["lon"] = _to_float(meta.get("LONGITUDE", ""))
    diario["alt"] = _to_float(meta.get("ALTITUDE", ""))
    return diario


def ler_zip_ano(caminho_zip, ufs_filtro):
    resultados = []
    ufs_up = [u.upper() for u in ufs_filtro]
    with zipfile.ZipFile(caminho_zip) as z:
        nomes = [n for n in z.namelist() if n.lower().endswith(".csv")]
        alvos = []
        for n in nomes:
            partes = re.split(r"[/_]", os.path.basename(n).upper())
            if any(uf in partes for uf in ufs_up):
                alvos.append(n)
        print(f"  {os.path.basename(caminho_zip)}: {len(alvos)} estacoes nas UFs alvo")
        for nome in alvos:
            with z.open(nome) as f:
                bruto = f.read()
            df = _ler_csv_de_bytes(bruto, nome)
            if df is not None and len(df) > 0:
                resultados.append(df)
    if not resultados:
        return pd.DataFrame()
    return pd.concat(resultados, ignore_index=True)


def montar_meteo_diaria():
    """Le todos os ZIPs anuais, filtra estacoes das UFs alvo + vizinhas,
    devolve dataframe diario por estacao."""
    raiz = os.path.join(cfg.DIR_DADOS_BRUTOS, "inmet")
    ufs = list(cfg.UFS_ALVO)
    if cfg.INCLUIR_ESTACOES_VIZINHAS:
        ufs += cfg.UFS_VIZINHAS

    blocos = []
    for ano in range(cfg.ANO_INICIO, cfg.ANO_FIM + 1):
        zp = os.path.join(raiz, f"{ano}.zip")
        if not os.path.exists(zp):
            print(f"  pulando {ano} (ZIP nao baixado)")
            continue
        df = ler_zip_ano(zp, ufs)
        if not df.empty:
            blocos.append(df)
    if not blocos:
        return pd.DataFrame()
    todos = pd.concat(blocos, ignore_index=True)
    todos["uf"] = todos["uf"].astype("category")
    return todos


if __name__ == "__main__":
    df = montar_meteo_diaria()
    print(df.shape)
    print(df.head())
    out = os.path.join(cfg.DIR_DADOS_PROCESSADOS, "meteo_diaria_estacoes.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"salvo em {out}")
