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


# ---------------------------------------------------------------------------
# FASE 1 - recuperacao do sinal dinamico (meteorologia/FWI sub-aproveitado).
#
# Diagnostico (ver README, secao "Diagnostico e fase 1"): o modelo de producao
# tinha ~64% da importancia concentrada em priors estaticos (taxa historica do
# municipio-mes, focos acumulados) e so ~13% nas 13 features meteo/FWI somadas.
# Ou seja, era quase um climatologico: aprendia "que municipio queima em que
# mes" e quase ignorava o tempo do dia.
#
# As funcoes abaixo extraem TENDENCIA (nao so nivel), INTERACOES risco x seca e
# VIZINHANCA espacial. Todas terminam no dia D (condicoes de hoje, usadas para
# prever D+1) ou usam shift no passado - nenhuma enxerga o futuro D+1.
# ---------------------------------------------------------------------------


def adicionar_tendencias_meteo(df):
    """Derivadas dinamicas de FWI/ISI/meteo. Capturam a TENDENCIA recente
    (secagem, agravamento do risco) e nao apenas o nivel instantaneo - que e
    ruidoso porque o IDW interpola de poucas estacoes. Janelas terminam no dia
    D (incluem D); deltas usam D vs dias anteriores. Sem vazamento de D+1."""
    df = df.sort_values(["codigo_ibge", "data"]).copy()
    g = df.groupby("codigo_ibge")

    # variacao recente do FWI: positivo = risco subindo
    df["fwi_delta_1d"] = g["fwi"].diff(1)
    df["fwi_delta_3d"] = g["fwi"].diff(3)
    # nivel suavizado e pico recente do FWI (janela inclui D)
    df["fwi_media_3d"] = g["fwi"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["fwi_media_7d"] = g["fwi"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["fwi_max_7d"] = g["fwi"].transform(lambda s: s.rolling(7, min_periods=1).max())
    # propagacao inicial recente (ISI suavizado)
    df["isi_media_3d"] = g["isi"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    # secagem do ar nos ultimos 3 dias (positivo = umidade caindo)
    df["umid_delta_3d"] = -g["umid_media"].diff(3)
    # tendencia de calor
    df["temp_max_media_3d"] = g["temp_max"].transform(lambda s: s.rolling(3, min_periods=1).mean())

    # diff gera NaN nas primeiras linhas de cada municipio -> 0 (sem variacao conhecida)
    for c in ["fwi_delta_1d", "fwi_delta_3d", "umid_delta_3d"]:
        df[c] = df[c].fillna(0.0)
    return df


def adicionar_interacoes(df):
    """Interacoes explicitas risco x seca. Arvores capturam interacoes
    implicitamente, mas entregar o produto pronto ajuda a isolar os dias
    realmente perigosos (FWI alto E seca longa, calor E muitos dias sem chuva)."""
    df = df.copy()
    df["fwi_x_dias_sem_chuva"] = df["fwi"].fillna(0) * df["dias_sem_chuva"]
    df["temp_max_x_dias_sem_chuva"] = df["temp_max"].fillna(0) * df["dias_sem_chuva"]
    # seca combinada com vento: condicao classica de espalhamento de fogo
    df["seca_x_vento"] = df["dias_sem_chuva"] * df["vento_medio"].fillna(0)
    return df


def adicionar_vizinhanca(df, k=6):
    """Features de vizinhanca espacial: focos e risco nos k municipios mais
    proximos (por centroide). Ignicao e propagacao tem autocorrelacao espacial
    - um foco ontem num vizinho eleva o risco de hoje->amanha aqui. Os focos do
    vizinho usam shift(1) (passado); o FWI do vizinho e do dia D (disponivel).
    Antes desta feature cada municipio era tratado como uma ilha isolada."""
    df = df.copy()
    df["_cod"] = df["codigo_ibge"].astype(str)

    coords = (
        df[["_cod", "centro_lat", "centro_lon"]]
        .drop_duplicates("_cod")
        .reset_index(drop=True)
    )
    cods = coords["_cod"].values
    lat = coords["centro_lat"].values.astype(float)
    lon = coords["centro_lon"].values.astype(float)

    # matriz de distancia haversine municipio x municipio
    la1 = np.radians(lat)[:, None]
    lo1 = np.radians(lon)[:, None]
    la2 = np.radians(lat)[None, :]
    lo2 = np.radians(lon)[None, :]
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    np.fill_diagonal(dist, np.inf)  # exclui o proprio municipio
    viz_idx = np.argsort(dist, axis=1)[:, :k]  # (n, k)

    # pivots data x municipio na ordem de `cods`
    pn = df.pivot_table(index="data", columns="_cod", values="n_focos", aggfunc="sum").reindex(columns=cods)
    pf = df.pivot_table(index="data", columns="_cod", values="fwi", aggfunc="mean").reindex(columns=cods)
    mat_n = pn.values.astype(np.float32)
    mat_f = pf.values.astype(np.float32)
    nd, n = mat_n.shape

    # focos do vizinho com 1 dia de defasagem, e acumulado dos 3 dias anteriores
    n_lag1 = np.vstack([np.zeros((1, n), dtype=np.float32), mat_n[:-1]])
    n_acum3 = pd.DataFrame(mat_n).shift(1).rolling(3, min_periods=1).sum().fillna(0).values

    focos_viz_lag1 = np.zeros((nd, n), dtype=np.float32)
    focos_viz_acum3 = np.zeros((nd, n), dtype=np.float32)
    fwi_viz = np.zeros((nd, n), dtype=np.float32)
    for i in range(n):
        js = viz_idx[i]
        focos_viz_lag1[:, i] = n_lag1[:, js].sum(axis=1)
        focos_viz_acum3[:, i] = n_acum3[:, js].sum(axis=1)
        fwi_viz[:, i] = np.nanmean(mat_f[:, js], axis=1)

    datas = pn.index
    blocos = []
    for i in range(n):
        blocos.append(pd.DataFrame({
            "_cod": cods[i],
            "data": datas,
            "focos_vizinhos_lag_1": focos_viz_lag1[:, i],
            "focos_vizinhos_acum_3d": focos_viz_acum3[:, i],
            "fwi_vizinhos": fwi_viz[:, i],
        }))
    viz = pd.concat(blocos, ignore_index=True)

    df = df.merge(viz, on=["_cod", "data"], how="left")
    df["fwi_vizinhos"] = df["fwi_vizinhos"].fillna(df["fwi"])
    for c in ["focos_vizinhos_lag_1", "focos_vizinhos_acum_3d"]:
        df[c] = df[c].fillna(0.0)
    return df.drop(columns="_cod")


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
