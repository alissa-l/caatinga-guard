# Busca de hiperparametros com TimeSeriesSplit otimizando AP.
# Salva o melhor conjunto em dados/processados/melhores_hiperparams.json.
# treinar.py le esse arquivo se existir e usa os valores como override.
#
# Justificativa: hiperparametros chutados em treinar.py podem deixar AP na
# mesa. AP eh a metrica relevante aqui (desbalanceamento severo) e nao AUC
# ou acuracia. Random search com 25 trials por modelo eh suficiente: o
# espaco util eh pequeno e ja cobre boa parte da regiao boa.
#
# rodar: python -m backend.modelo.buscar_hiperparams

import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import average_precision_score
from scipy.stats import randint, uniform
import lightgbm as lgb

from backend import configuracao as cfg
from backend.modelo.base import BalancedBaggingRF
from backend.modelo.treinar import (
    FEATURES, ALVO, carregar_dataset, split_temporal, montar_matrizes, imputar
)


N_TRIALS = 25
SEMENTE = 42


def _buscar_rf_bagging(Xt, yt, Xv, yv, n_trials, semente=SEMENTE):
    """Busca aleatoria FIEL ao modelo de producao: treina o proprio
    BalancedBaggingRF e avalia AP na validacao temporal. RandomizedSearchCV nao
    serve aqui porque clona o estimador (BalancedBaggingRF nao e um estimador
    sklearn). O espaco inclui os params do RF-base e tambem os do balanceamento
    (n_ensembles, razao_neg). AP na val tem ruido (poucos positivos), mas e o
    mesmo sinal que selecionou o bagging no experimento de desbalanceamento.

    min_samples_leaf usa faixa baixa (1-20): cada RF-base ve so ~5k linhas
    subamostradas, nao as ~244k do treino completo."""
    rng = np.random.RandomState(semente)
    mf_opts = ["sqrt", "log2", 0.5]
    melhor_ap, melhor_cfg = -1.0, None
    print(f"\n=== Random Forest (BalancedBaggingRF, {n_trials} trials, AP na validacao) ===")
    for t in range(n_trials):
        cfg_t = {
            "n_estimators": int(rng.randint(120, 320)),
            "max_depth": int(rng.randint(8, 22)),
            "min_samples_leaf": int(rng.randint(1, 20)),
            "max_features": mf_opts[rng.randint(len(mf_opts))],
            "n_ensembles": int(rng.randint(10, 21)),
            "razao_neg": int(rng.choice([2, 3, 4, 5])),
        }
        rf_params = {k: cfg_t[k] for k in
                     ("n_estimators", "max_depth", "min_samples_leaf", "max_features")}
        modelo = BalancedBaggingRF(
            n_ensembles=cfg_t["n_ensembles"], razao_neg=cfg_t["razao_neg"],
            semente=semente, n_jobs=-1, **rf_params,
        ).fit(Xt, yt)
        ap = float(average_precision_score(yv, modelo.predict_proba(Xv)[:, 1]))
        if ap > melhor_ap:
            melhor_ap, melhor_cfg = ap, cfg_t
        print(f"  trial {t + 1:2d}/{n_trials}  AP_val={ap:.4f}  {cfg_t}")
    print(f"  melhor AP val: {melhor_ap:.4f}")
    print(f"  melhores params: {melhor_cfg}")
    return melhor_cfg, melhor_ap


def _espaco_lgbm():
    return {
        "n_estimators": randint(200, 700),
        "learning_rate": uniform(0.01, 0.09),
        "num_leaves": randint(15, 127),
        "min_child_samples": randint(10, 80),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.0, 1.0),
    }


def _buscar(modelo_base, espaco, X, y, n_trials, label):
    """RandomizedSearchCV com TimeSeriesSplit otimizando AP."""
    cv = TimeSeriesSplit(n_splits=4)
    busca = RandomizedSearchCV(
        modelo_base,
        param_distributions=espaco,
        n_iter=n_trials,
        scoring="average_precision",
        cv=cv,
        random_state=SEMENTE,
        n_jobs=-1,
        verbose=1,
        refit=False,
    )
    print(f"\n=== {label} ({n_trials} trials, TimeSeriesSplit 4 folds, otimiza AP) ===")
    busca.fit(X, y)
    print(f"  melhor AP CV: {busca.best_score_:.4f}")
    print(f"  melhores params: {busca.best_params_}")
    return busca.best_params_, float(busca.best_score_)


def main():
    cfg.garantir_diretorios()
    print("carregando dataset")
    df = carregar_dataset()
    treino, val, teste, cortes = split_temporal(df)
    print(f"  treino: {len(treino)} ({treino[ALVO].sum()} pos) | "
          f"val: {len(val)} ({val[ALVO].sum()} pos)")

    # --- Random Forest (BalancedBaggingRF): treina no treino, avalia AP na val
    # temporal. RF nao aceita NaN -> imputa com mediana do treino (consistente
    # com treinar.py). Jamais toca o teste.
    Xt, Xv, Xte = montar_matrizes(treino), montar_matrizes(val), montar_matrizes(teste)
    yt, yv = treino[ALVO].values, val[ALVO].values
    Xt_i, Xv_i, _, _ = imputar(Xt, Xv, Xte)
    params_rf, score_rf = _buscar_rf_bagging(Xt_i, yt, Xv_i, yv, N_TRIALS)

    # --- LightGBM: RandomizedSearchCV com TimeSeriesSplit em treino+val (LGBM
    # lida com NaN nativo). TimeSeriesSplit ja corta dentro dessa janela.
    busca_df = pd.concat([treino, val]).sort_values("data")
    X = montar_matrizes(busca_df)
    y = busca_df[ALVO].values
    spw = float((y == 0).sum() / max(1, (y == 1).sum()))
    lg = lgb.LGBMClassifier(scale_pos_weight=spw, random_state=SEMENTE, verbose=-1)
    params_lg, score_lg = _buscar(lg, _espaco_lgbm(), X, y, N_TRIALS, "LightGBM")

    # converte numpy ints/floats para tipos nativos (json-friendly)
    def _serializar(d):
        return {k: (int(v) if isinstance(v, np.integer)
                    else float(v) if isinstance(v, np.floating)
                    else v) for k, v in d.items()}

    saida = {
        "random_forest": _serializar(params_rf),
        "lightgbm": _serializar(params_lg),
        "_meta": {
            "n_trials": N_TRIALS,
            "metrica": "average_precision",
            "ap_val_random_forest": score_rf,
            "ap_cv_lightgbm": score_lg,
            "tamanho_busca": len(busca_df),
            "features": FEATURES,
        },
    }
    out = os.path.join(cfg.DIR_DADOS_PROCESSADOS, "melhores_hiperparams.json")
    with open(out, "w") as f:
        json.dump(saida, f, indent=2)
    print(f"\nsalvo: {out}")
    print("rode `make treinar` para retreinar com os hiperparams encontrados.")


if __name__ == "__main__":
    main()
