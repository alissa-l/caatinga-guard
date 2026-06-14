# Construcao de features lag, acumuladas, sazonais e geograficas.
# IMPORTANTE: features lag e acumuladas usam SOMENTE dados anteriores ao
# dia em questao - evitar vazamento temporal.

import numpy as np
import pandas as pd


def adicionar_lags_focos(df):
    df = df.sort_values(["codigo_ibge", "data"]).copy()
    for lag in [1, 3, 7]:
        df[f"focos_lag_{lag}"] = (
            df.groupby("codigo_ibge")["n_focos"]
            .shift(lag)
            .fillna(0)
        )
    return df


def adicionar_chuva_acumulada(df):
    df = df.sort_values(["codigo_ibge", "data"]).copy()
    g = df.groupby("codigo_ibge")["chuva_dia"]
    df["chuva_acum_7d"] = g.transform(lambda x: x.shift(1).rolling(7, min_periods=1).sum())
    df["chuva_acum_30d"] = g.transform(lambda x: x.shift(1).rolling(30, min_periods=1).sum())
    return df


def adicionar_dias_sem_chuva(df, limiar=1.0):
    df = df.sort_values(["codigo_ibge", "data"]).copy()

    def streak(s):
        seco = (s.shift(1).fillna(0) < limiar).astype(int)
        # contagem cumulativa que reinicia quando seco vira 0
        return seco * (seco.groupby((seco != seco.shift()).cumsum()).cumcount() + 1)

    df["dias_sem_chuva"] = df.groupby("codigo_ibge")["chuva_dia"].transform(streak)
    return df


def adicionar_sazonalidade(df):
    df = df.copy()
    mes = df["data"].dt.month
    doy = df["data"].dt.dayofyear
    df["mes_sin"] = np.sin(2 * np.pi * mes / 12)
    df["mes_cos"] = np.cos(2 * np.pi * mes / 12)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    return df


def adicionar_alvo(df):
    """alvo: houve_foco_d1 = 1 se houver foco no proximo dia naquele municipio."""
    df = df.sort_values(["codigo_ibge", "data"]).copy()
    df["focos_d1"] = df.groupby("codigo_ibge")["n_focos"].shift(-1).fillna(0)
    df["houve_foco_d1"] = (df["focos_d1"] >= 1).astype("int8")
    return df


def distancia_litoral_km(centro_lat, centro_lon, costa_lon=-35.0):
    """Aproximacao grosseira: o RN tem orientacao costa norte-leste,
    distancia em graus * 111 km. Para uso de feature, e suficiente."""
    return np.abs(centro_lon - costa_lon) * 111.0


def adicionar_focos_acumulados(df, janelas=(30, 90)):
    """Focos acumulados em janelas longas por municipio. Usa shift(1) para
    nao incluir o dia atual (evita vazamento ja que n_focos do dia eh feature)."""
    df = df.sort_values(["codigo_ibge", "data"]).copy()
    for j in janelas:
        df[f"focos_acum_{j}d"] = (
            df.groupby("codigo_ibge")["n_focos"]
            .transform(lambda s: s.shift(1).rolling(j, min_periods=1).sum())
        )
    return df


def adicionar_taxas_historicas(df, corte_data_treino):
    """Mean encoding por municipio e municipio-mes usando SO o periodo de
    treino (datas < corte_data_treino). Aplica como feature em todas as
    linhas (treino, val, teste). Leakage-safe porque val e teste nao
    contribuem para a media."""
    df = df.copy()
    treino = df[df["data"] < corte_data_treino]

    taxa_mun = (
        treino.groupby("codigo_ibge")["houve_foco_d1"]
        .mean()
        .rename("taxa_historica_municipio")
        .reset_index()
    )
    df = df.merge(taxa_mun, on="codigo_ibge", how="left")

    treino_mes = treino.assign(_mes=treino["data"].dt.month)
    taxa_mun_mes = (
        treino_mes.groupby(["codigo_ibge", "_mes"])["houve_foco_d1"]
        .mean()
        .rename("taxa_historica_municipio_mes")
        .reset_index()
    )
    df["_mes"] = df["data"].dt.month
    df = df.merge(taxa_mun_mes, on=["codigo_ibge", "_mes"], how="left").drop(columns="_mes")

    # municipios que nunca tiveram foco no periodo de treino: usa 0
    df["taxa_historica_municipio"] = df["taxa_historica_municipio"].fillna(0)
    df["taxa_historica_municipio_mes"] = df["taxa_historica_municipio_mes"].fillna(
        df["taxa_historica_municipio"]
    )
    return df
