# Compara estrategias de tratamento de desbalanceamento (taxa base ~1%).
# Treina cada candidato no TREINO, seleciona por AP + recall@10 na VALIDACAO
# e reporta o TESTE so como referencia (jamais usado para escolher).
#
# Nenhuma dependencia nova: as estrategias do imbalanced-learn
# (BalancedRandomForest) sao reproduzidas a mao com undersampling + ensemble.
#
# rodar: python -m backend.modelo.experimento_balanceamento

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import lightgbm as lgb

from backend.modelo.treinar import (
    FEATURES, ALVO, carregar_dataset, split_temporal,
    montar_matrizes, imputar, _metricas_amigaveis,
)


SEMENTE = 42


# --------------------------------------------------------------------------
# candidato RF: balanced bagging manual (undersampling + ensemble).
# Cada estimador ve todos os positivos + uma amostra de negativos na razao
# `razao_neg`:1. Reproduz a ideia do BalancedRandomForest sem imblearn.
# --------------------------------------------------------------------------
class BalancedBaggingRF:
    def __init__(self, n_ensembles=15, razao_neg=1, **rf_params):
        self.n_ensembles = n_ensembles
        self.razao_neg = razao_neg
        self.rf_params = rf_params
        self.modelos = []

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        n_neg = min(len(neg), len(pos) * self.razao_neg)
        rng = np.random.RandomState(SEMENTE)
        self.modelos = []
        for i in range(self.n_ensembles):
            amostra_neg = rng.choice(neg, size=n_neg, replace=False)
            idx = np.concatenate([pos, amostra_neg])
            rng.shuffle(idx)
            params = {**self.rf_params, "random_state": SEMENTE + i}
            rf = RandomForestClassifier(**params)
            rf.fit(X.iloc[idx], y[idx])
            self.modelos.append(rf)
        return self

    def predict_proba(self, X):
        ps = np.mean([m.predict_proba(X)[:, 1] for m in self.modelos], axis=0)
        return np.column_stack([1 - ps, ps])


# --------------------------------------------------------------------------
# focal loss para LightGBM (objetivo customizado).
# Gradiente/Hessiana por diferencas finitas (implementacao robusta de
# jrzaurin/LightGBM-with-Focal-Loss) - evita bugs de derivada analitica.
# Foca o treino nos exemplos dificeis, util quando positivos sao raros.
# --------------------------------------------------------------------------
def focal_loss_obj(alpha=0.25, gamma=2.0):
    def obj(y_true, y_pred):
        a, g = alpha, gamma

        def fl(x, t):
            p = 1.0 / (1.0 + np.exp(-x))
            p = np.clip(p, 1e-9, 1 - 1e-9)
            return -(a * t + (1 - a) * (1 - t)) * (
                (1 - (t * p + (1 - t) * (1 - p))) ** g
            ) * (t * np.log(p) + (1 - t) * np.log(1 - p))

        eps = 1e-6
        grad = (fl(y_pred + eps, y_true) - fl(y_pred - eps, y_true)) / (2 * eps)
        hess = (fl(y_pred + eps, y_true) + fl(y_pred - eps, y_true)
                - 2 * fl(y_pred, y_true)) / (eps ** 2)
        return grad, hess
    return obj


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def avaliar(nome, p_val, yv, datas_val, p_teste, yte, datas_teste):
    av_v = _metricas_amigaveis(yv, p_val, datas_val)
    av_t = _metricas_amigaveis(yte, p_teste, datas_teste)
    return {
        "modelo": nome,
        "ap_val": average_precision_score(yv, p_val),
        "auc_val": roc_auc_score(yv, p_val),
        "rec10_val": av_v["recall_top10"],
        "lift10_val": av_v["lift_top10"],
        "ap_teste": average_precision_score(yte, p_teste),
        "auc_teste": roc_auc_score(yte, p_teste),
        "rec10_teste": av_t["recall_top10"],
        "lift10_teste": av_t["lift_top10"],
    }


def main():
    print("carregando dataset e split temporal")
    df = carregar_dataset()
    treino, val, teste, _ = split_temporal(df)
    Xt, Xv, Xte = montar_matrizes(treino), montar_matrizes(val), montar_matrizes(teste)
    yt, yv, yte = treino[ALVO].values, val[ALVO].values, teste[ALVO].values
    dv, dte = val["data"].values, teste["data"].values
    Xt_i, Xv_i, Xte_i, _ = imputar(Xt, Xv, Xte)
    spw = float((yt == 0).sum() / max(1, (yt == 1).sum()))
    print(f"  treino {len(treino)} ({yt.sum()} pos) | val {len(val)} ({yv.sum()} pos) "
          f"| teste {len(teste)} ({yte.sum()} pos) | scale_pos_weight cheio = {spw:.0f}")

    rf_base = dict(n_estimators=200, max_depth=14, min_samples_leaf=20, n_jobs=-1, random_state=SEMENTE)
    lgb_base = dict(n_estimators=400, learning_rate=0.05, num_leaves=63,
                    subsample=0.9, colsample_bytree=0.8, random_state=SEMENTE, verbose=-1)

    resultados = []

    print("[1] RF class_weight='balanced' (baseline atual)")
    m = RandomForestClassifier(class_weight="balanced", **rf_base).fit(Xt_i, yt)
    resultados.append(avaliar("rf_balanced (atual)", m.predict_proba(Xv_i)[:, 1], yv, dv,
                              m.predict_proba(Xte_i)[:, 1], yte, dte))

    print("[2] RF class_weight='balanced_subsample'")
    m = RandomForestClassifier(class_weight="balanced_subsample", **rf_base).fit(Xt_i, yt)
    resultados.append(avaliar("rf_balanced_subsample", m.predict_proba(Xv_i)[:, 1], yv, dv,
                              m.predict_proba(Xte_i)[:, 1], yte, dte))

    print("[3] RF balanced bagging (undersampling 1:1, 15 estimadores)")
    m = BalancedBaggingRF(n_ensembles=15, razao_neg=1, **rf_base).fit(Xt_i, yt)
    resultados.append(avaliar("rf_bagging_1:1", m.predict_proba(Xv_i)[:, 1], yv, dv,
                              m.predict_proba(Xte_i)[:, 1], yte, dte))

    print("[4] RF balanced bagging (undersampling 3:1, 15 estimadores)")
    m = BalancedBaggingRF(n_ensembles=15, razao_neg=3, **rf_base).fit(Xt_i, yt)
    resultados.append(avaliar("rf_bagging_3:1", m.predict_proba(Xv_i)[:, 1], yv, dv,
                              m.predict_proba(Xte_i)[:, 1], yte, dte))

    print("[5] LightGBM scale_pos_weight cheio (baseline atual)")
    m = lgb.LGBMClassifier(scale_pos_weight=spw, **lgb_base).fit(Xt, yt)
    resultados.append(avaliar("lgb_spw_cheio (atual)", m.predict_proba(Xv)[:, 1], yv, dv,
                              m.predict_proba(Xte)[:, 1], yte, dte))

    print("[6] LightGBM scale_pos_weight = sqrt(cheio)")
    m = lgb.LGBMClassifier(scale_pos_weight=np.sqrt(spw), **lgb_base).fit(Xt, yt)
    resultados.append(avaliar("lgb_spw_sqrt", m.predict_proba(Xv)[:, 1], yv, dv,
                              m.predict_proba(Xte)[:, 1], yte, dte))

    print("[7] LightGBM is_unbalance=True")
    m = lgb.LGBMClassifier(is_unbalance=True, **lgb_base).fit(Xt, yt)
    resultados.append(avaliar("lgb_is_unbalance", m.predict_proba(Xv)[:, 1], yv, dv,
                              m.predict_proba(Xte)[:, 1], yte, dte))

    print("[8] LightGBM focal loss (alpha=0.25, gamma=2.0)")
    m = lgb.LGBMClassifier(objective=focal_loss_obj(0.25, 2.0), **lgb_base).fit(Xt, yt)
    resultados.append(avaliar("lgb_focal", _sigmoid(m.predict(Xv, raw_score=True)), yv, dv,
                              _sigmoid(m.predict(Xte, raw_score=True)), yte, dte))

    res = pd.DataFrame(resultados).sort_values("ap_val", ascending=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n=== SELECAO POR VALIDACAO (ordenado por AP val) ===")
    print(res[["modelo", "ap_val", "auc_val", "rec10_val", "lift10_val"]].round(4).to_string(index=False))
    print("\n=== REFERENCIA NO TESTE (nao usado para escolher) ===")
    print(res[["modelo", "ap_teste", "auc_teste", "rec10_teste", "lift10_teste"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
