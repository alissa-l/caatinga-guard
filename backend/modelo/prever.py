# Utilitarios de previsao usados pela API.
# Carrega o modelo em memoria uma unica vez (cache).

import os
import joblib
import numpy as np
import pandas as pd

from backend import configuracao as cfg


_cache = {}


def carregar(nome="random_forest"):
    if nome in _cache:
        return _cache[nome]
    path = os.path.join(cfg.DIR_ARTEFATOS, f"{nome}.joblib")
    obj = joblib.load(path)
    _cache[nome] = obj
    return obj


def prever_proba(df_linhas, nome_modelo="random_forest"):
    """df_linhas: dataframe com as colunas listadas em FEATURES.
    Retorna array de probabilidades."""
    obj = carregar(nome_modelo)
    modelo = obj["modelo"]
    features = obj["features"]
    medianas = obj["medianas"]

    X = df_linhas.reindex(columns=features).copy()
    for c in features:
        if X[c].isna().any():
            X[c] = X[c].fillna(medianas.get(c, 0))
    return modelo.predict_proba(X)[:, 1]
