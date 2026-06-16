# Interface comum dos modelos. Permite plugar uma rede neural depois sem
# refatorar o resto da pipeline.

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class ModeloBase:
    nome = "base"

    def treinar(self, X, y, X_val=None, y_val=None):
        raise NotImplementedError

    def prever_proba(self, X):
        raise NotImplementedError

    def salvar(self, caminho):
        joblib.dump(self, caminho)

    @classmethod
    def carregar(cls, caminho):
        return joblib.load(caminho)


class BalancedBaggingRF:
    """Random Forest com balanceamento por undersampling + ensemble.

    Cada um dos `n_ensembles` estimadores ve TODOS os positivos mais uma
    amostra aleatoria de negativos na razao `razao_neg`:1, e as probabilidades
    sao promediadas. Reproduz a ideia do BalancedRandomForest (imbalanced-learn)
    sem dependencia nova. Com taxa base ~1%, supera class_weight='balanced':
    cada arvore enxerga uma fracao decente de positivos em vez de reponderar um
    punhado deles num mar de negativos.

    Vive em base.py (e nao em treinar.py) para que joblib desserialize o
    artefato em qualquer entrypoint - API, notebook, etc."""

    def __init__(self, n_ensembles=15, razao_neg=3, semente=42, **rf_params):
        self.n_ensembles = n_ensembles
        self.razao_neg = razao_neg
        self.semente = semente
        self.rf_params = rf_params
        self.modelos = []
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        n_neg = min(len(neg), len(pos) * self.razao_neg)
        rng = np.random.RandomState(self.semente)
        self.modelos = []
        for i in range(self.n_ensembles):
            amostra_neg = rng.choice(neg, size=n_neg, replace=False)
            idx = np.concatenate([pos, amostra_neg])
            rng.shuffle(idx)
            params = {**self.rf_params, "random_state": self.semente + i}
            rf = RandomForestClassifier(**params)
            rf.fit(X.iloc[idx], y[idx])
            self.modelos.append(rf)
        return self

    def predict_proba(self, X):
        ps = np.mean([m.predict_proba(X)[:, 1] for m in self.modelos], axis=0)
        return np.column_stack([1 - ps, ps])

    @property
    def feature_importances_(self):
        return np.mean([m.feature_importances_ for m in self.modelos], axis=0)


class ModeloCalibrado:
    """Wrapper de um classificador binario com isotonic regression aplicado
    sobre a probabilidade de classe positiva, calibrada num conjunto held-out.

    Importavel via backend.modelo.base para que joblib consiga desserializar
    o objeto em qualquer entrypoint (treinar.py, uvicorn, notebook, etc)."""

    def __init__(self, base, calibrador):
        self.base = base
        self.calibrador = calibrador

    def predict_proba(self, X):
        p = self.base.predict_proba(X)[:, 1]
        p_cal = np.clip(self.calibrador.transform(p), 0.0, 1.0)
        # isotonic produz plateaus (muitos empates), o que destroi o ranking
        # fino e faz com que a simulacao em linha unica fique "presa". Mistura
        # uma fracao pequena (peso 1e-3) da probabilidade bruta na saida final:
        # como p_cal e isotonica em p, p_final fica estritamente crescente em p
        # e a calibracao se desloca, no maximo, 1e-3 — fora do alcance pratico
        # do Brier. Funciona em batch e em linha unica.
        p_final = np.clip(0.999 * p_cal + 1e-3 * p, 0.0, 1.0)
        return np.column_stack([1.0 - p_final, p_final])

    @property
    def feature_importances_(self):
        return getattr(self.base, "feature_importances_", None)

    @property
    def classes_(self):
        return self.base.classes_
