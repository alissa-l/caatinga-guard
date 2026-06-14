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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import average_precision_score
from scipy.stats import randint, uniform
import lightgbm as lgb

from backend import configuracao as cfg
from backend.modelo.treinar import (
    FEATURES, ALVO, carregar_dataset, split_temporal, montar_matrizes, imputar
)


N_TRIALS = 25
SEMENTE = 42


def _espaco_rf():
    return {
        "n_estimators": randint(150, 350),
        "max_depth": randint(8, 22),
        "min_samples_leaf": randint(10, 60),
        "max_features": ["sqrt", "log2", 0.5],
        "min_samples_split": randint(2, 20),
    }


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
    print(f"  treino: {len(treino)} ({treino[ALVO].sum()} pos)")

    # busca usa apenas treino+val (jamais teste). Como TimeSeriesSplit ja
    # corta dentro dessa janela, basta concatenar e ordenar por data.
    busca_df = pd.concat([treino, val]).sort_values("data")
    X = montar_matrizes(busca_df)
    y = busca_df[ALVO].values

    # RF nao aceita NaN. Imputa com mediana do treino (consistente com
    # treinar.py). LightGBM pode lidar com NaN nativo.
    X_imp = X.fillna(X.median(numeric_only=True))

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_jobs=-1, class_weight="balanced", random_state=SEMENTE,
    )
    params_rf, score_rf = _buscar(rf, _espaco_rf(), X_imp, y, N_TRIALS, "Random Forest")

    # --- LightGBM ---
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
            "ap_cv_random_forest": score_rf,
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
