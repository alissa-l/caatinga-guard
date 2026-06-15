# Testes da camada de feature engineering. Cobrem vazamento temporal,
# determinismo e sanidade dos artefatos intermediarios. Sao mais rapidos
# que test_modelo.py porque trabalham com dados sinteticos pequenos.
#
# rodar:   pytest tests/test_features.py -v

import numpy as np
import pandas as pd
import pytest

from backend.tratamento import features as ft
from backend.tratamento.calcular_fwi import calcular_fwi_serie


def _serie_sintetica(n_dias=120, codigo="9999999", semente=1):
    rng = np.random.default_rng(semente)
    datas = pd.date_range("2024-01-01", periods=n_dias, freq="D")
    df = pd.DataFrame({
        "codigo_ibge": codigo,
        "data": datas,
        "n_focos": rng.integers(0, 4, size=n_dias),
        "chuva_dia": rng.gamma(0.3, 5.0, size=n_dias).round(1),
        "temp_media": rng.normal(28, 3, size=n_dias),
        "temp_max": rng.normal(33, 3, size=n_dias),
        "temp_min": rng.normal(23, 2, size=n_dias),
        "umid_media": rng.uniform(40, 90, size=n_dias),
        "vento_medio": rng.uniform(1, 6, size=n_dias),
    })
    return df


def _duas_series_sinteticas(n_dias=90):
    a = _serie_sintetica(n_dias, "1111111", semente=1)
    b = _serie_sintetica(n_dias, "2222222", semente=2)
    return pd.concat([a, b], ignore_index=True)


# ---- lags ----

def test_lags_sem_vazamento_temporal():
    """focos_lag_1 no dia D bate com n_focos do dia D-1 do mesmo municipio.
    Garante que o shift esta na direcao certa."""
    df = _serie_sintetica()
    out = ft.adicionar_lags_focos(df)
    sub = out.sort_values("data").reset_index(drop=True)
    # primeiro dia: lag=0 por convencao (fillna)
    assert sub.loc[0, "focos_lag_1"] == 0
    # dia 1 em diante: bate com n_focos do anterior
    for i in range(1, len(sub)):
        assert sub.loc[i, "focos_lag_1"] == sub.loc[i - 1, "n_focos"], (
            f"focos_lag_1 quebrou em i={i}"
        )


def test_lags_nao_misturam_municipios():
    df = _duas_series_sinteticas()
    out = ft.adicionar_lags_focos(df)
    # primeira linha de cada municipio tem lag zerado
    primeiras = out.groupby("codigo_ibge").head(1)
    assert (primeiras["focos_lag_1"] == 0).all()
    assert (primeiras["focos_lag_3"] == 0).all()
    assert (primeiras["focos_lag_7"] == 0).all()


# ---- chuva acumulada ----

def test_chuva_acum_nao_inclui_dia_atual():
    """chuva_acum_7d no dia D usa chuva de D-7..D-1 (nao inclui D).
    Sem isso, vaza informacao do dia para a previsao do proprio dia."""
    df = _serie_sintetica()
    out = ft.adicionar_chuva_acumulada(df).sort_values("data").reset_index(drop=True)
    # primeira linha: rolling sobre shift(1) so tem NaN, transformado em 0 ou NaN
    assert pd.isna(out.loc[0, "chuva_acum_7d"]) or out.loc[0, "chuva_acum_7d"] == 0
    # dia 7: deve bater com soma dos dias 0..5 (shift 1 + janela 7)
    soma_esperada = df.loc[0:5, "chuva_dia"].sum()
    np.testing.assert_allclose(out.loc[6, "chuva_acum_7d"], soma_esperada, atol=1e-8)


# ---- focos acumulados ----

def test_focos_acum_sem_vazamento():
    """focos_acum_30d no dia D nao inclui n_focos do proprio dia D."""
    df = _serie_sintetica(n_dias=60)
    out = ft.adicionar_focos_acumulados(df, janelas=(30,)).sort_values("data").reset_index(drop=True)
    # dia 30: deve bater com soma dos focos dos dias 0..29 (shift 1 + janela 30)
    soma_esperada = df.loc[0:29, "n_focos"].sum()
    np.testing.assert_allclose(out.loc[30, "focos_acum_30d"], soma_esperada, atol=1e-8)


# ---- mean encoding leakage-safe ----

def test_taxa_historica_sem_vazamento():
    """taxa_historica_municipio calculada com corte X usa SO datas < X."""
    df = _duas_series_sinteticas(n_dias=120)
    df["houve_foco_d1"] = (df["n_focos"] >= 1).astype(int)
    corte = pd.Timestamp("2024-03-01")

    out = ft.adicionar_taxas_historicas(df, corte)

    # recalcula manualmente usando so o periodo de treino
    treino = df[df["data"] < corte]
    esperado = treino.groupby("codigo_ibge")["houve_foco_d1"].mean()

    for cod, taxa_esperada in esperado.items():
        observado = out.loc[out["codigo_ibge"] == cod, "taxa_historica_municipio"].iloc[0]
        np.testing.assert_allclose(observado, taxa_esperada, atol=1e-10), (
            f"taxa_historica de {cod} usou dados fora do periodo de treino"
        )


def test_taxa_historica_constante_por_municipio():
    """A taxa historica do municipio nao pode variar dentro do mesmo municipio."""
    df = _duas_series_sinteticas()
    df["houve_foco_d1"] = (df["n_focos"] >= 1).astype(int)
    out = ft.adicionar_taxas_historicas(df, pd.Timestamp("2024-02-01"))
    g = out.groupby("codigo_ibge")["taxa_historica_municipio"].nunique()
    assert (g == 1).all(), "taxa_historica nao deveria variar dentro do municipio"


# ---- FWI ----

def test_fwi_deterministico():
    """Rodar duas vezes na mesma serie produz saidas identicas."""
    df = _serie_sintetica(n_dias=60).sort_values("data").reset_index(drop=True)
    r1 = calcular_fwi_serie(df.copy())
    r2 = calcular_fwi_serie(df.copy())
    for col in ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]:
        np.testing.assert_array_equal(
            r1[col].values, r2[col].values,
            err_msg=f"FWI nao deterministico na coluna {col}",
        )


def test_fwi_nao_negativo():
    df = _serie_sintetica(n_dias=120)
    out = calcular_fwi_serie(df.sort_values("data").reset_index(drop=True))
    for col in ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]:
        assert (out[col].dropna() >= 0).all(), f"{col} produziu valor negativo"


def test_fwi_chuva_extrema_reduz_secagem():
    """Cenario chuvoso prolongado deve gerar FWI menor que cenario seco."""
    base = _serie_sintetica(n_dias=60)
    seco = base.copy()
    seco["chuva_dia"] = 0
    seco["umid_media"] = 30
    chuvoso = base.copy()
    chuvoso["chuva_dia"] = 15
    chuvoso["umid_media"] = 90

    s_out = calcular_fwi_serie(seco.sort_values("data").reset_index(drop=True))
    c_out = calcular_fwi_serie(chuvoso.sort_values("data").reset_index(drop=True))
    assert s_out["fwi"].mean() > c_out["fwi"].mean(), (
        "FWI medio em cenario seco precisa ser > cenario chuvoso"
    )


# ---- sazonalidade ----

def test_sazonalidade_seno_cosseno_no_circulo():
    df = _serie_sintetica()
    out = ft.adicionar_sazonalidade(df)
    # sin^2 + cos^2 = 1 (com folga numerica)
    r_mes = (out["mes_sin"] ** 2 + out["mes_cos"] ** 2).round(6).unique()
    r_doy = (out["doy_sin"] ** 2 + out["doy_cos"] ** 2).round(6).unique()
    assert (r_mes == 1).all()
    assert (r_doy == 1).all()


# ---- alvo ----

def test_alvo_eh_proximo_dia():
    """houve_foco_d1 no dia D olha n_focos no dia D+1."""
    df = _serie_sintetica()
    out = ft.adicionar_alvo(df).sort_values("data").reset_index(drop=True)
    for i in range(len(out) - 1):
        esperado = 1 if df.loc[i + 1, "n_focos"] >= 1 else 0
        assert out.loc[i, "houve_foco_d1"] == esperado, f"alvo errado em i={i}"
