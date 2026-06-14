# Interface comum dos modelos. Permite plugar uma rede neural depois sem
# refatorar o resto da pipeline.

import joblib
import numpy as np


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
