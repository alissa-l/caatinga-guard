import os
import json
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from datetime import timedelta

from backend import configuracao as cfg
from backend.api.estado import estado
from backend.api.esquemas import (
    Municipio, PrevisaoDia, PrevisaoMunicipio,
    ComparacaoDia, ComparacaoMunicipio,
    PedidoSimulacao, RespostaSimulacao,
    PrevisaoFuturo, PrevisaoFuturoDia,
)
from backend.modelo import prever


router = APIRouter()


def _linhas_data(data_str):
    """Devolve subset do fato_municipio_dia para essa data."""
    d = pd.to_datetime(data_str)
    sub = estado.df[estado.df["data"] == d]
    if sub.empty:
        raise HTTPException(404, f"sem dados para {data_str}")
    return sub


@router.get("/municipios", response_model=list[Municipio])
def listar_municipios():
    return [
        Municipio(
            codigo_ibge=str(r["codigo_ibge"]),
            nome=r["nome"],
            uf=r["uf"],
            centro_lat=float(r["centro_lat"]),
            centro_lon=float(r["centro_lon"]),
            area_km2=float(r["area_km2"]),
        )
        for _, r in estado.municipios.iterrows()
    ]


@router.get("/municipios/geojson")
def municipios_geojson():
    return estado.geojson


@router.get("/previsao/futuro", response_model=PrevisaoFuturo)
def previsao_futuro(dias: int = 3, modelo: str = "random_forest", data_base: str | None = None):
    if dias < 1 or dias > 7:
        raise HTTPException(400, "dias deve estar em [1, 7]")
    if data_base is None:
        data_base = estado.datas_disponiveis[-1]
    base = _linhas_data(data_base)

    saida_dias = []
    cur = base.copy()
    d0 = pd.to_datetime(data_base)
    for k in range(1, dias + 1):
        dk = d0 + timedelta(days=k)
        mes = dk.month
        doy = dk.timetuple().tm_yday
        cur = cur.copy()
        cur["mes_sin"] = np.sin(2 * np.pi * mes / 12)
        cur["mes_cos"] = np.cos(2 * np.pi * mes / 12)
        cur["doy_sin"] = np.sin(2 * np.pi * doy / 365)
        cur["doy_cos"] = np.cos(2 * np.pi * doy / 365)
        p = prever.prever_proba(cur, nome_modelo=modelo)
        saida_dias.append(PrevisaoFuturoDia(
            data_alvo=dk.strftime("%Y-%m-%d"),
            previsoes=[PrevisaoMunicipio(codigo_ibge=str(c), probabilidade=float(pp))
                       for c, pp in zip(cur["codigo_ibge"].tolist(), p)],
        ))
    return PrevisaoFuturo(
        data_base=data_base,
        n_dias=dias,
        observacao="previsao baseada em persistencia climatica - confiabilidade decresce com a janela",
        dias=saida_dias,
    )


@router.post("/previsao/simulacao", response_model=RespostaSimulacao)
def simulacao(pedido: PedidoSimulacao):
    d = pd.to_datetime(pedido.data_base)
    base = estado.df[(estado.df["data"] == d) & (estado.df["codigo_ibge"] == pedido.codigo_ibge)]
    if base.empty:
        raise HTTPException(404, "municipio/data nao encontrados")

    p_orig = float(prever.prever_proba(base)[0])

    sim = base.copy()
    aj = pedido.ajustes
    sim["temp_media"] = sim["temp_media"] + (aj.temperatura or 0)
    sim["temp_max"] = sim["temp_max"] + (aj.temperatura or 0)
    sim["temp_min"] = sim["temp_min"] + (aj.temperatura or 0)
    sim["umid_media"] = (sim["umid_media"] + (aj.umidade or 0)).clip(0, 100)
    sim["chuva_dia"] = sim["chuva_dia"] * (aj.precipitacao if aj.precipitacao is not None else 1)
    sim["vento_medio"] = sim["vento_medio"] * (aj.vento if aj.vento is not None else 1)
    p_sim = float(prever.prever_proba(sim)[0])

    return RespostaSimulacao(
        codigo_ibge=pedido.codigo_ibge,
        data_base=pedido.data_base,
        probabilidade_original=p_orig,
        probabilidade_simulada=p_sim,
        ajustes_aplicados=aj.model_dump(),
    )


@router.get("/previsao/{data}", response_model=PrevisaoDia)
def previsao(data: str, modelo: str = "random_forest"):
    sub = _linhas_data(data)
    proba = prever.prever_proba(sub, nome_modelo=modelo)
    d = pd.to_datetime(data)
    return PrevisaoDia(
        data=data,
        data_alvo=(d + timedelta(days=1)).strftime("%Y-%m-%d"),
        modelo=modelo,
        previsoes=[
            PrevisaoMunicipio(codigo_ibge=str(c), probabilidade=float(p))
            for c, p in zip(sub["codigo_ibge"].tolist(), proba)
        ],
    )


@router.get("/previsao/{data}/comparacao", response_model=ComparacaoDia)
def comparacao(data: str, modelo: str = "random_forest", k: int = 10):
    sub = _linhas_data(data)
    proba = prever.prever_proba(sub, nome_modelo=modelo)
    y_real = sub["houve_foco_d1"].astype(int).values

    # ranking top-K e quantos focos reais entraram no top — o limiar 0,5
    # nao faz sentido apos calibracao isotonica em problema tao desbalanceado
    # (a taxa base e ~1%). top-K e a leitura operacional consistente com o
    # painel de saude do modelo.
    n_focos_real = int(y_real.sum())
    k = max(1, min(int(k), len(proba)))
    ordem = np.argsort(-proba)
    top_idx = ordem[:k]
    hits_topk = int(y_real[top_idx].sum())
    recall_topk = hits_topk / n_focos_real if n_focos_real > 0 else 0.0
    precisao_topk = hits_topk / k
    fora_topk_com_foco = n_focos_real - hits_topk

    d = pd.to_datetime(data)
    return ComparacaoDia(
        data=data,
        data_alvo=(d + timedelta(days=1)).strftime("%Y-%m-%d"),
        modelo=modelo,
        previsoes=[
            ComparacaoMunicipio(
                codigo_ibge=str(c),
                probabilidade=float(p),
                teve_foco=int(y),
            )
            for c, p, y in zip(sub["codigo_ibge"].tolist(), proba, y_real)
        ],
        acerto={
            "k": k,
            "n_focos_reais": n_focos_real,
            "hits_topk": hits_topk,
            "recall_topk": float(recall_topk),
            "precisao_topk": float(precisao_topk),
            "focos_fora_topk": fora_topk_com_foco,
        },
    )


@router.get("/modelo/relatorio")
def relatorio_modelo():
    with open(os.path.join(cfg.DIR_AVALIACAO, "metricas.json")) as f:
        m = json.load(f)
    return m


@router.get("/modelo/curvas")
def curvas_modelo():
    """Curva de recall por top-K e calibracao agregada. Retorno enxuto
    (sem os 30k probas individuais que estao em curvas.json)."""
    with open(os.path.join(cfg.DIR_AVALIACAO, "curvas.json")) as f:
        c = json.load(f)
    saida = {}
    for nome, dados in c.items():
        # calibracao agregada em bins
        import numpy as np
        p = np.array(dados["probas_teste"])
        y = np.array(dados["y_teste"])
        bins_calib = []
        edges = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.50, 1.0]
        for a, b in zip(edges[:-1], edges[1:]):
            mask = (p >= a) & (p < b)
            if mask.sum() > 0:
                bins_calib.append({
                    "bin": f"{a:.3f}-{b:.3f}",
                    "n": int(mask.sum()),
                    "previsto_medio": float(p[mask].mean()),
                    "real_medio": float(y[mask].mean()),
                })
        saida[nome] = {
            "recall_por_k": dados.get("recall_por_k", []),
            "calibracao_bins": bins_calib,
            "amigaveis": dados.get("amigaveis", {}),
        }
    return saida


@router.get("/datas")
def datas():
    """Lista as datas para as quais ha dados (frontend usa pra popular seletor)."""
    return {"datas": estado.datas_disponiveis}
