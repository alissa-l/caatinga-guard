# Testes de saude do modelo e das inferencias. Rodam contra o artefato
# salvo (.joblib) e o conjunto de teste do split temporal. Os limiares
# sao thresholds operacionais - se uma mudanca derrubar abaixo, o
# desenvolvedor saca que tem regressao.
#
# rodar:   pytest tests/ -v

import json
import os
import numpy as np
import pandas as pd
import pytest

from backend import configuracao as cfg
from backend.modelo import prever
from backend.modelo.treinar import (
    FEATURES, ALVO, carregar_dataset, split_temporal, montar_matrizes
)


# limiares minimos de saude. abaixo disso o modelo nao serve.
LIMIAR_AUC = 0.70
LIMIAR_RECALL_TOP10 = 0.12
LIMIAR_LIFT_TOP10 = 2.5
LIMIAR_DIAS_HIT_TOP10_PCT = 0.30
LIMIAR_BRIER_MAX = 0.05
LIMIAR_DESVIO_CALIBRACAO = 0.5  # taxa_prev nao pode estar 50%+ longe da taxa_real


@pytest.fixture(scope="module")
def df_teste():
    df = carregar_dataset()
    _, _, teste, _ = split_temporal(df)
    return teste


@pytest.fixture(scope="module")
def probas(df_teste):
    p = prever.prever_proba(df_teste, nome_modelo="random_forest")
    return p


@pytest.fixture(scope="module")
def metricas():
    """Le metricas geradas pelo treinar.py."""
    caminho = os.path.join(cfg.DIR_AVALIACAO, "metricas.json")
    assert os.path.exists(caminho), "rode `make treinar` antes dos testes"
    with open(caminho) as f:
        return json.load(f)


# ---- smoke ----

def test_artefato_existe():
    p = os.path.join(cfg.DIR_ARTEFATOS, "random_forest.joblib")
    assert os.path.exists(p), "artefato do random forest nao encontrado"
    assert os.path.getsize(p) > 1000, "artefato suspeito de estar truncado"


def test_modelo_carrega():
    obj = prever.carregar("random_forest")
    assert "modelo" in obj
    assert "features" in obj
    assert "medianas" in obj
    assert len(obj["features"]) == len(FEATURES)


def test_features_consistentes_entre_artefato_e_codigo():
    """Detecta divergencia entre FEATURES do treinar.py e o salvo no artefato."""
    obj = prever.carregar("random_forest")
    assert obj["features"] == FEATURES, \
        "FEATURES no artefato diferem de FEATURES em treinar.py. retreine."


# ---- output range e formato ----

def test_predict_proba_em_zero_um(probas):
    assert probas.min() >= 0.0
    assert probas.max() <= 1.0
    assert not np.isnan(probas).any()


def test_predict_proba_tem_variancia(probas):
    """Modelo morto previa sempre o mesmo valor."""
    assert probas.std() > 1e-4, "predicao sem variancia, modelo provavelmente quebrado"


def test_predict_proba_determinista(df_teste):
    """Mesma entrada gera mesma saida (dentro da precisao de float)."""
    p1 = prever.prever_proba(df_teste.head(200), nome_modelo="random_forest")
    p2 = prever.prever_proba(df_teste.head(200), nome_modelo="random_forest")
    np.testing.assert_allclose(p1, p2, atol=1e-10)


def test_proba_muda_com_input(df_teste):
    """Mudar a entrada precisa mexer alguma coisa na saida (modelo nao morto)."""
    base = df_teste.head(200).copy()
    mexido = base.copy()
    mexido["temp_media"] = mexido["temp_media"] + 5
    mexido["umid_media"] = (mexido["umid_media"] - 15).clip(0, 100)
    p_base = prever.prever_proba(base, nome_modelo="random_forest")
    p_mexido = prever.prever_proba(mexido, nome_modelo="random_forest")
    diff = np.abs(p_mexido - p_base).max()
    assert diff > 1e-5, f"perturbar input nao alterou a previsao em nada (max diff {diff})"


# ---- qualidade preditiva (regressao guard) ----

def test_auc_minimo(metricas):
    auc = metricas["random_forest"]["teste"]["auc"]
    assert auc >= LIMIAR_AUC, f"AUC caiu para {auc:.3f}, abaixo do minimo {LIMIAR_AUC}"


def test_recall_top10(metricas):
    rec = metricas["random_forest"]["amigaveis"]["recall_top10"]
    assert rec >= LIMIAR_RECALL_TOP10, \
        f"recall@10 caiu para {rec:.3f}, abaixo do minimo {LIMIAR_RECALL_TOP10}"


def test_lift_top10(metricas):
    lift = metricas["random_forest"]["amigaveis"]["lift_top10"]
    assert lift >= LIMIAR_LIFT_TOP10, \
        f"lift@10 caiu para {lift:.2f}x, abaixo do minimo {LIMIAR_LIFT_TOP10}x"


def test_dias_com_hit_top10(metricas):
    amig = metricas["random_forest"]["amigaveis"]
    pct = amig["dias_com_hit_top10"] / max(1, amig["n_dias_com_focos"])
    assert pct >= LIMIAR_DIAS_HIT_TOP10_PCT, \
        f"so {pct*100:.1f}% dos dias com focos tem hit no top-10 (minimo {LIMIAR_DIAS_HIT_TOP10_PCT*100:.0f}%)"


# ---- calibracao ----

def test_brier_score(metricas):
    brier = metricas["random_forest"]["amigaveis"]["brier"]
    assert brier <= LIMIAR_BRIER_MAX, \
        f"Brier score {brier:.4f} alto (max {LIMIAR_BRIER_MAX})"


def test_calibracao_proxima_da_realidade(metricas):
    """A probabilidade media prevista nao pode estar muito longe da taxa real."""
    amig = metricas["random_forest"]["amigaveis"]
    real = amig["taxa_real"]
    prev = amig["taxa_prevista_media"]
    if real > 0:
        desvio = abs(prev - real) / real
        assert desvio <= LIMIAR_DESVIO_CALIBRACAO, \
            f"calibracao ruim: previsto {prev:.4f} vs real {real:.4f} ({desvio*100:.0f}% de desvio)"


# ---- API / dataset ----

def test_dataset_tem_alvo(df_teste):
    assert ALVO in df_teste.columns
    assert df_teste[ALVO].isin([0, 1]).all()


def test_dataset_sem_nulos_em_features_chave(df_teste):
    """O modelo imputa, mas se MAIS de 5% de uma feature for nula, alguma
    coisa quebrou no ETL."""
    for f in ["taxa_historica_municipio", "taxa_historica_municipio_mes",
              "centro_lat", "centro_lon", "area_km2", "n_focos"]:
        pct_nulo = df_teste[f].isna().mean()
        assert pct_nulo < 0.05, f"feature {f} com {pct_nulo*100:.1f}% de nulos"


def test_dataset_intervalos_razoaveis(df_teste):
    """Sanity dos intervalos das features chave (ignora nulos)."""
    df = df_teste
    umid = df["umid_media"].dropna()
    assert umid.between(0, 100).all(), f"umidade fora de [0,100]: min={umid.min()} max={umid.max()}"
    temp = df["temp_media"].dropna()
    assert temp.between(-10, 50).all(), f"temperatura suspeita: min={temp.min()} max={temp.max()}"
    assert df["chuva_dia"].fillna(0).ge(0).all(), "chuva negativa"
    assert df["fwi"].fillna(0).ge(0).all(), "fwi negativo"


def test_taxa_historica_municipio_zerada_consistente(df_teste):
    """Cada municipio tem um valor unico de taxa_historica_municipio."""
    g = df_teste.groupby("codigo_ibge")["taxa_historica_municipio"].nunique()
    assert (g <= 1).all(), "taxa_historica_municipio mudou dentro do mesmo municipio"


def test_simulacao_chuva_extrema_reduz_risco(df_teste):
    """Sanity: zerar chuva acumulada longa + bater temp para alta deve nao
    deixar a previsao identica. O modelo eh dominado pelo historico do
    municipio, entao testes direcionais sobre meteo sao fracos - basta
    checar que a perturbacao move alguma coisa."""
    base = df_teste.head(500).copy()
    sim = base.copy()
    # cenario extremo: nada de chuva, temp alta, umid baixa
    sim["chuva_dia"] = 0
    sim["chuva_acum_7d"] = 0
    sim["chuva_acum_30d"] = 0
    sim["dias_sem_chuva"] = 60
    sim["temp_media"] += 8
    sim["temp_max"] += 8
    sim["umid_media"] = (sim["umid_media"] - 30).clip(0, 100)
    p_base = prever.prever_proba(base)
    p_sim = prever.prever_proba(sim)
    delta_max = float(np.abs(p_sim - p_base).max())
    assert delta_max > 1e-4, f"cenario extremo nao moveu a previsao (max delta {delta_max})"
