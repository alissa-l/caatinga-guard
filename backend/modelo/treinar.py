# Treina os modelos com split temporal estrito.
# Salva artefatos e metricas para a fase de avaliacao consumir.

import os
import json
import hashlib
import shutil
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score,
)
import lightgbm as lgb

from backend.modelo.base import ModeloCalibrado

from backend import configuracao as cfg


# arquivo opcional com hiperparametros encontrados pela busca. Se nao
# existir, treinar.py cai nos defaults chutados (provavelmente piores).
CAMINHO_HIPERPARAMS = os.path.join(cfg.DIR_DADOS_PROCESSADOS, "melhores_hiperparams.json")


FEATURES = [
    # meteorologicas
    "temp_media", "temp_max", "temp_min",
    "umid_media", "chuva_dia", "vento_medio", "rad_media",
    # FWI
    "ffmc", "dmc", "dc", "isi", "bui", "fwi",
    # lag e acumuladas de focos
    "n_focos",
    "focos_lag_1", "focos_lag_3", "focos_lag_7",
    "focos_acum_30d", "focos_acum_90d",
    # chuva acumulada
    "chuva_acum_7d", "chuva_acum_30d", "dias_sem_chuva",
    # historico do municipio (mean encoding leakage-safe)
    "taxa_historica_municipio", "taxa_historica_municipio_mes",
    # geograficas
    "area_km2", "centro_lat", "centro_lon", "distancia_litoral_km",
    "bioma_caatinga", "bioma_mata_atlantica",
    # uso do solo via OSM (zero quando o ETL nao rodou)
    "osm_estradas_km", "osm_estradas_principais_km",
    "osm_pasto_km2", "osm_cultivo_km2",
    # sazonais
    "mes_sin", "mes_cos", "doy_sin", "doy_cos",
]

ALVO = "houve_foco_d1"


def carregar_dataset():
    pq = os.path.join(cfg.DIR_DADOS_PROCESSADOS, "fato_municipio_dia.parquet")
    df = pd.read_parquet(pq)
    # one-hot do bioma
    df["bioma_caatinga"] = (df["bioma"] == "caatinga").astype("int8")
    df["bioma_mata_atlantica"] = (df["bioma"] == "mata_atlantica").astype("int8")
    df["data"] = pd.to_datetime(df["data"])
    return df


def split_temporal(df, frac_treino=0.8, frac_val=0.1):
    datas = np.sort(df["data"].unique())
    n = len(datas)
    corte1 = datas[int(frac_treino * n)]
    corte2 = datas[int((frac_treino + frac_val) * n)]
    treino = df[df["data"] < corte1]
    val = df[(df["data"] >= corte1) & (df["data"] < corte2)]
    teste = df[df["data"] >= corte2]
    return treino, val, teste, (corte1, corte2)


def _metricas(y, p, thr=0.5):
    yp = (p >= thr).astype(int)
    return {
        "auc": float(roc_auc_score(y, p)) if y.sum() > 0 else None,
        "ap": float(average_precision_score(y, p)) if y.sum() > 0 else None,
        "acc": float(accuracy_score(y, yp)),
        "precision": float(precision_score(y, yp, zero_division=0)),
        "recall": float(recall_score(y, yp, zero_division=0)),
        "f1": float(f1_score(y, yp, zero_division=0)),
        "confusao": confusion_matrix(y, yp).tolist(),
        "n": int(len(y)),
        "positivos": int(y.sum()),
    }


def _metricas_amigaveis(y, p, datas_por_linha, n_munis=167, ks=(5, 10, 20)):
    """Computa metricas faceis de explicar:
    - recall_topK: dos focos reais, quantos % estavam no top-K do dia
    - lift_topK: recall / fracao do estado (=K/n_munis)
    - dias_com_hit_topK: dos dias com focos reais, quantos tiveram >=1 foco no top-K
    - brier: erro medio quadratico, mede calibracao
    - taxa_real, taxa_prevista_media: comparacao direta
    """
    df = pd.DataFrame({"data": pd.to_datetime(datas_por_linha), "y": y, "p": p})
    total_pos = int(df["y"].sum())
    dias_unicos = df["data"].nunique()
    dias_com_focos = int((df.groupby("data")["y"].sum() > 0).sum())

    res = {
        "taxa_real": float(y.mean()),
        "taxa_prevista_media": float(p.mean()),
        "brier": float(((p - y) ** 2).mean()),
        "n_dias": dias_unicos,
        "n_dias_com_focos": dias_com_focos,
        "total_focos": total_pos,
    }
    for K in ks:
        tp_total = 0
        dias_hit = 0
        for _, dia in df.groupby("data"):
            top = dia.nlargest(K, "p")
            tp_dia = int(top["y"].sum())
            tp_total += tp_dia
            if tp_dia > 0:
                dias_hit += 1
        rec = tp_total / total_pos if total_pos else 0.0
        lift = rec / (K / n_munis) if K > 0 else 0.0
        res[f"recall_top{K}"] = float(rec)
        res[f"lift_top{K}"] = float(lift)
        res[f"dias_com_hit_top{K}"] = int(dias_hit)
    return res


def _curva_recall(y, p, datas_por_linha, n_munis=167, k_max=50):
    """Recall acumulado conforme N aumenta (top-N por dia)."""
    df = pd.DataFrame({"data": pd.to_datetime(datas_por_linha), "y": y, "p": p})
    total_pos = int(df["y"].sum()) or 1
    out = []
    for K in range(1, k_max + 1):
        tp = 0
        for _, dia in df.groupby("data"):
            top = dia.nlargest(K, "p")
            tp += int(top["y"].sum())
        recall = tp / total_pos
        baseline = K / n_munis
        out.append({"k": K, "recall": float(recall), "baseline_aleatorio": float(baseline)})
    return out


def _treinar_rf(Xt, yt, Xv, yv, balanceado=True, params=None):
    # class_weight='balanced' eh obrigatorio aqui porque a taxa base de
    # houve_foco_d1 fica perto de 1%. Sem isso, o RF "ganha" o treino prevendo
    # tudo como negativo e o split do no fica insensivel a focos. A
    # probabilidade fica inflada, mas isso eh corrigido depois pela
    # IsotonicRegression (ver docs/diagnostico_falsos_positivos.md).
    base_params = dict(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=20,
        n_jobs=-1,
        class_weight="balanced" if balanceado else None,
        random_state=42,
    )
    if params:
        base_params.update(params)
    rf = RandomForestClassifier(**base_params)
    rf.fit(Xt, yt)
    return rf


def _treinar_lgbm(Xt, yt, Xv, yv, balanceado=True, params=None):
    spw = float((yt == 0).sum() / max(1, (yt == 1).sum())) if balanceado else None
    base_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    if spw is not None:
        base_params["scale_pos_weight"] = spw
    if params:
        base_params.update(params)
    m = lgb.LGBMClassifier(**base_params)
    m.fit(
        Xt, yt,
        eval_set=[(Xv, yv)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return m


def _carregar_hiperparams():
    """Le melhores hiperparams encontrados por buscar_hiperparams.py.
    Devolve dict por nome de modelo. Se nao houver arquivo, dict vazio."""
    if not os.path.exists(CAMINHO_HIPERPARAMS):
        return {}
    with open(CAMINHO_HIPERPARAMS) as f:
        return json.load(f)


def _versionar_artefato(nome, payload, hiperparams):
    """Salva o artefato em duas vias:
    - artefatos/{nome}.joblib (em uso pela API)
    - artefatos/historico/{nome}__{timestamp}__{hash}.joblib (rastreamento)

    Hash deriva de hiperparams + lista de FEATURES + janela temporal.
    Salva tambem metadados em json paralelo."""
    historico = os.path.join(cfg.DIR_ARTEFATOS, "historico")
    os.makedirs(historico, exist_ok=True)

    info = {
        "hiperparams": hiperparams,
        "features": FEATURES,
        "ano_inicio": cfg.ANO_INICIO,
        "ano_fim": cfg.ANO_FIM,
    }
    h = hashlib.sha1(json.dumps(info, sort_keys=True).encode()).hexdigest()[:10]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    versao = f"{nome}__{ts}__{h}.joblib"

    caminho_versao = os.path.join(historico, versao)
    joblib.dump(payload, caminho_versao)
    caminho_corrente = os.path.join(cfg.DIR_ARTEFATOS, f"{nome}.joblib")
    shutil.copyfile(caminho_versao, caminho_corrente)

    meta_path = caminho_versao.replace(".joblib", ".json")
    with open(meta_path, "w") as f:
        json.dump({**info, "versao": versao, "criado_em": ts}, f, indent=2)
    print(f"  salvo {caminho_corrente}")
    print(f"  versao historica: {versao}")
    return caminho_versao


def montar_matrizes(df, features=FEATURES):
    X = df[features].copy()
    # lightgbm aceita NaN, mas RF nao. Imputar com mediana das features
    # (treino calcula, val/teste reutilizam).
    return X


def imputar(X_treino, X_val, X_teste):
    medianas = X_treino.median(numeric_only=True)
    return (
        X_treino.fillna(medianas),
        X_val.fillna(medianas),
        X_teste.fillna(medianas),
        medianas,
    )


def main():
    cfg.garantir_diretorios()
    print("carregando dataset")
    df = carregar_dataset()
    print(f"  {len(df)} linhas, positivos: {df[ALVO].sum()}")

    print("split temporal")
    treino, val, teste, cortes = split_temporal(df)
    print(f"  treino: {len(treino)} ({treino[ALVO].sum()} pos)")
    print(f"  val:    {len(val)} ({val[ALVO].sum()} pos)")
    print(f"  teste:  {len(teste)} ({teste[ALVO].sum()} pos)")
    print(f"  cortes: {pd.Timestamp(cortes[0]).date()} | {pd.Timestamp(cortes[1]).date()}")

    Xt = montar_matrizes(treino)
    Xv = montar_matrizes(val)
    Xte = montar_matrizes(teste)
    yt = treino[ALVO].values
    yv = val[ALVO].values
    yte = teste[ALVO].values
    datas_teste = teste["data"].values  # para metricas amigaveis

    Xt_imp, Xv_imp, Xte_imp, medianas = imputar(Xt, Xv, Xte)

    resultados = {}
    artefatos = {}
    hp_busca = _carregar_hiperparams()
    if hp_busca:
        print(f"hiperparams da busca: {list(hp_busca.keys())}")

    print("treinando RandomForest (balanceado) + calibracao isotonic")
    rf = _treinar_rf(Xt_imp, yt, Xv_imp, yv, balanceado=True, params=hp_busca.get("random_forest"))
    importancias_rf = dict(zip(FEATURES, rf.feature_importances_.tolist()))
    # probabilidades infladas (antes da calibracao) usadas como referencia
    p_val_raw = rf.predict_proba(Xv_imp)[:, 1]
    p_teste_raw = rf.predict_proba(Xte_imp)[:, 1]
    # isotonic regression sobre as probabilidades brutas no conjunto de validacao
    iso_rf = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_rf.fit(p_val_raw, yv)
    rf_cal = ModeloCalibrado(rf, iso_rf)
    p_val = rf_cal.predict_proba(Xv_imp)[:, 1]
    p_teste = rf_cal.predict_proba(Xte_imp)[:, 1]
    resultados["random_forest"] = {
        "val": _metricas(yv, p_val),
        "teste": _metricas(yte, p_teste),
        "teste_pre_calibracao": _metricas(yte, p_teste_raw),
        "amigaveis": _metricas_amigaveis(yte, p_teste, datas_teste),
        "importancias": importancias_rf,
        "calibrado": True,
    }
    artefatos["random_forest"] = rf_cal

    print("treinando RandomForest (sem balanceamento)")
    rf2 = _treinar_rf(Xt_imp, yt, Xv_imp, yv, balanceado=False)
    p_val2 = rf2.predict_proba(Xv_imp)[:, 1]
    p_teste2 = rf2.predict_proba(Xte_imp)[:, 1]
    resultados["random_forest_nao_balanceado"] = {
        "val": _metricas(yv, p_val2),
        "teste": _metricas(yte, p_teste2),
    }

    print("treinando LightGBM (balanceado) + calibracao isotonic")
    lg = _treinar_lgbm(Xt, yt, Xv, yv, balanceado=True, params=hp_busca.get("lightgbm"))
    importancias_lg = dict(zip(FEATURES, lg.feature_importances_.tolist()))
    p_val_raw_lg = lg.predict_proba(Xv)[:, 1]
    p_teste_raw_lg = lg.predict_proba(Xte)[:, 1]
    iso_lg = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_lg.fit(p_val_raw_lg, yv)
    lg_cal = ModeloCalibrado(lg, iso_lg)
    p_val = lg_cal.predict_proba(Xv)[:, 1]
    p_teste = lg_cal.predict_proba(Xte)[:, 1]
    resultados["lightgbm"] = {
        "val": _metricas(yv, p_val),
        "teste": _metricas(yte, p_teste),
        "teste_pre_calibracao": _metricas(yte, p_teste_raw_lg),
        "amigaveis": _metricas_amigaveis(yte, p_teste, datas_teste),
        "importancias": importancias_lg,
        "calibrado": True,
    }
    artefatos["lightgbm"] = lg_cal

    print("treinando LightGBM (sem balanceamento)")
    lg2 = _treinar_lgbm(Xt, yt, Xv, yv, balanceado=False)
    p_val2 = lg2.predict_proba(Xv)[:, 1]
    p_teste2 = lg2.predict_proba(Xte)[:, 1]
    resultados["lightgbm_nao_balanceado"] = {
        "val": _metricas(yv, p_val2),
        "teste": _metricas(yte, p_teste2),
    }

    # salvar artefatos com versionamento (historico mantem rastreio dos runs)
    for nome, m in artefatos.items():
        payload = {"modelo": m, "features": FEATURES, "medianas": medianas.to_dict()}
        _versionar_artefato(nome, payload, hp_busca.get(nome, {}))

    # salvar metricas e cortes
    resultados["_meta"] = {
        "ano_inicio": cfg.ANO_INICIO,
        "ano_fim": cfg.ANO_FIM,
        "corte_treino_val": str(pd.Timestamp(cortes[0]).date()),
        "corte_val_teste": str(pd.Timestamp(cortes[1]).date()),
        "n_features": len(FEATURES),
        "features": FEATURES,
    }
    out = os.path.join(cfg.DIR_AVALIACAO, "metricas.json")
    with open(out, "w") as f:
        json.dump(resultados, f, indent=2)
    print(f"  metricas em {out}")

    # tambem salva curvas ROC e PR para plot posterior (usa modelos calibrados).
    # adiciona metricas amigaveis e curva de recall acumulada por top-K.
    curvas = {}
    for nome, mdl in [("random_forest", rf_cal), ("lightgbm", lg_cal)]:
        if nome == "lightgbm":
            p = mdl.predict_proba(Xte)[:, 1]
        else:
            p = mdl.predict_proba(Xte_imp)[:, 1]
        fpr, tpr, _ = roc_curve(yte, p)
        prec, rec, _ = precision_recall_curve(yte, p)
        amigaveis = _metricas_amigaveis(yte, p, datas_teste)
        curva_recall = _curva_recall(yte, p, datas_teste)
        curvas[nome] = {
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr":  {"precision": prec.tolist(), "recall": rec.tolist()},
            "probas_teste": p.tolist(),
            "y_teste": yte.tolist(),
            "amigaveis": amigaveis,
            "recall_por_k": curva_recall,
        }
    with open(os.path.join(cfg.DIR_AVALIACAO, "curvas.json"), "w") as f:
        json.dump(curvas, f)
    print(f"  curvas salvas")

    print()
    print("=== resumo teste ===")
    for nome, r in resultados.items():
        if nome.startswith("_"):
            continue
        t = r["teste"]
        print(f"  {nome:35s} AUC={t['auc']:.3f}  AP={t['ap']:.3f}  "
              f"P={t['precision']:.3f}  R={t['recall']:.3f}  F1={t['f1']:.3f}")


if __name__ == "__main__":
    main()
