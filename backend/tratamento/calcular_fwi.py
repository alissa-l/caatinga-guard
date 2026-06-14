# Fire Weather Index (FWI) canadense - Van Wagner 1987.
# Implementacao por serie temporal: cada componente do dia D depende
# do dia D-1. Recebe dataframe diario com colunas (temp, umid, chuva, vento)
# e devolve com colunas ffmc, dmc, dc, isi, bui, fwi.

import math
import numpy as np
import pandas as pd


# valores iniciais conservadores
FFMC0 = 85.0
DMC0 = 6.0
DC0 = 15.0


def _ffmc(temp, umid, vento, chuva, ffmc_ant):
    # Van Wagner & Pickett 1985, codigo classico
    mo = 147.2 * (101.0 - ffmc_ant) / (59.5 + ffmc_ant)
    if chuva > 0.5:
        rf = chuva - 0.5
        if mo <= 150.0:
            mr = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
        else:
            mr = (mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
                  + 0.0015 * (mo - 150.0) ** 2 * math.sqrt(rf))
        mo = min(mr, 250.0)

    ed = 0.942 * (umid ** 0.679) + 11.0 * math.exp((umid - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * umid))
    if mo > ed:
        ko = 0.424 * (1.0 - (umid / 100.0) ** 1.7) + 0.0694 * math.sqrt(vento) * (1.0 - (umid / 100.0) ** 8)
        kd = ko * 0.581 * math.exp(0.0365 * temp)
        m = ed + (mo - ed) * (10.0 ** -kd)
    else:
        ew = 0.618 * (umid ** 0.753) + 10.0 * math.exp((umid - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * umid))
        if mo < ew:
            kl = 0.424 * (1.0 - ((100.0 - umid) / 100.0) ** 1.7) + 0.0694 * math.sqrt(vento) * (1.0 - ((100.0 - umid) / 100.0) ** 8)
            kw = kl * 0.581 * math.exp(0.0365 * temp)
            m = ew - (ew - mo) * (10.0 ** -kw)
        else:
            m = mo
    return 59.5 * (250.0 - m) / (147.2 + m)


def _dmc(temp, umid, chuva, dmc_ant, mes):
    # tabela de comprimento efetivo do dia (Van Wagner) por mes
    Le = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0][mes - 1]
    if chuva > 1.5:
        re = 0.92 * chuva - 1.27
        mo = 20.0 + math.exp(5.6348 - dmc_ant / 43.43)
        if dmc_ant <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc_ant)
        elif dmc_ant <= 65.0:
            b = 14.0 - 1.3 * math.log(dmc_ant)
        else:
            b = 6.2 * math.log(dmc_ant) - 17.2
        mr = mo + 1000.0 * re / (48.77 + b * re)
        pr = 244.72 - 43.43 * math.log(mr - 20.0)
        dmc_ant = max(pr, 0.0)
    k = 1.894 * (temp + 1.1) * (100.0 - umid) * Le * 1e-6 if temp > -1.1 else 0.0
    return dmc_ant + 100.0 * k


def _dc(temp, chuva, dc_ant, mes):
    Lf = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6][mes - 1]
    if chuva > 2.8:
        rd = 0.83 * chuva - 1.27
        Qo = 800.0 * math.exp(-dc_ant / 400.0)
        Qr = Qo + 3.937 * rd
        dr = 400.0 * math.log(800.0 / Qr) if Qr > 0 else 0.0
        dc_ant = max(dr, 0.0)
    v = 0.36 * (temp + 2.8) + Lf
    if v < 0:
        v = 0.0
    return dc_ant + 0.5 * v


def _isi(ffmc, vento):
    fW = math.exp(0.05039 * vento)
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    fF = 91.9 * math.exp(-0.1386 * m) * (1.0 + (m ** 5.31) / 4.93e7)
    return 0.208 * fW * fF


def _bui(dmc, dc):
    if dmc <= 0.4 * dc:
        return 0.8 * dmc * dc / (dmc + 0.4 * dc) if (dmc + 0.4 * dc) > 0 else 0.0
    return dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)


def _fwi(isi, bui):
    fD = 0.626 * (bui ** 0.809) + 2.0 if bui <= 80 else 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui))
    B = 0.1 * isi * fD
    if B > 1.0:
        return math.exp(2.72 * (0.434 * math.log(B)) ** 0.647)
    return B


def calcular_fwi_serie(df):
    """df deve estar ordenado por data e ter colunas:
    temp_media, umid_media, chuva_dia, vento_medio (ja em km/h)."""
    df = df.sort_values("data").reset_index(drop=True)
    ffmc_ant, dmc_ant, dc_ant = FFMC0, DMC0, DC0
    saida_ffmc, saida_dmc, saida_dc, saida_isi, saida_bui, saida_fwi = [], [], [], [], [], []
    for _, row in df.iterrows():
        t = row.get("temp_media")
        h = row.get("umid_media")
        v = row.get("vento_medio")
        c = row.get("chuva_dia") or 0.0
        if pd.isna(t) or pd.isna(h) or pd.isna(v):
            saida_ffmc.append(np.nan)
            saida_dmc.append(np.nan)
            saida_dc.append(np.nan)
            saida_isi.append(np.nan)
            saida_bui.append(np.nan)
            saida_fwi.append(np.nan)
            continue
        v_kmh = float(v) * 3.6 if v < 50 else float(v)  # vento em m/s -> km/h se valor baixo
        h = float(min(max(h, 0.1), 100.0))
        mes = pd.Timestamp(row["data"]).month
        ffmc_ant = _ffmc(float(t), h, v_kmh, float(c), ffmc_ant)
        dmc_ant = _dmc(float(t), h, float(c), dmc_ant, mes)
        dc_ant = _dc(float(t), float(c), dc_ant, mes)
        isi = _isi(ffmc_ant, v_kmh)
        bui = _bui(dmc_ant, dc_ant)
        fwi = _fwi(isi, bui)
        saida_ffmc.append(ffmc_ant)
        saida_dmc.append(dmc_ant)
        saida_dc.append(dc_ant)
        saida_isi.append(isi)
        saida_bui.append(bui)
        saida_fwi.append(fwi)
    df = df.copy()
    df["ffmc"] = saida_ffmc
    df["dmc"] = saida_dmc
    df["dc"] = saida_dc
    df["isi"] = saida_isi
    df["bui"] = saida_bui
    df["fwi"] = saida_fwi
    return df
