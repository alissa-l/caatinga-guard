# Le metricas.json e curvas.json e produz CSVs e PNGs para o relatorio
# do frontend e o notebook.

import os
import json
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backend import configuracao as cfg


def _carregar():
    with open(os.path.join(cfg.DIR_AVALIACAO, "metricas.json")) as f:
        m = json.load(f)
    with open(os.path.join(cfg.DIR_AVALIACAO, "curvas.json")) as f:
        c = json.load(f)
    return m, c


def plot_roc(curvas):
    fig, ax = plt.subplots(figsize=(6, 5))
    for nome, dados in curvas.items():
        fpr = dados["roc"]["fpr"]
        tpr = dados["roc"]["tpr"]
        ax.plot(fpr, tpr, label=nome)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Curva ROC - teste")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(cfg.DIR_AVALIACAO, "roc.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_pr(curvas):
    fig, ax = plt.subplots(figsize=(6, 5))
    for nome, dados in curvas.items():
        prec = dados["pr"]["precision"]
        rec = dados["pr"]["recall"]
        ax.plot(rec, prec, label=nome)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall - teste")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(cfg.DIR_AVALIACAO, "pr.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_calibracao(curvas, bins=10):
    fig, ax = plt.subplots(figsize=(6, 5))
    for nome, dados in curvas.items():
        p = np.array(dados["probas_teste"])
        y = np.array(dados["y_teste"])
        edges = np.linspace(0, 1, bins + 1)
        xs, ys = [], []
        for i in range(bins):
            mask = (p >= edges[i]) & (p < edges[i + 1])
            if mask.sum() > 30:
                xs.append(p[mask].mean())
                ys.append(y[mask].mean())
        ax.plot(xs, ys, "o-", label=nome)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("probabilidade prevista (media do bin)")
    ax.set_ylabel("frequencia observada")
    ax.set_title("Calibracao - teste")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(cfg.DIR_AVALIACAO, "calibracao.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_matriz_confusao(metricas):
    figs = []
    for nome, r in metricas.items():
        if nome.startswith("_") or "teste" not in r:
            continue
        cm = np.array(r["teste"]["confusao"])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["nao", "sim"])
        ax.set_yticklabels(["nao", "sim"])
        ax.set_xlabel("previsto")
        ax.set_ylabel("real")
        ax.set_title(f"matriz confusao - {nome}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        out = os.path.join(cfg.DIR_AVALIACAO, f"confusao_{nome}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        figs.append(out)
    return figs


def plot_feature_importance(metricas):
    figs = []
    for nome, r in metricas.items():
        if "importancias" not in r:
            continue
        imp = pd.Series(r["importancias"]).sort_values(ascending=True)
        top = imp.tail(20)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(top.index, top.values)
        ax.set_title(f"top 20 features - {nome}")
        fig.tight_layout()
        out = os.path.join(cfg.DIR_AVALIACAO, f"importancia_{nome}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        figs.append(out)
    return figs


def salvar_csvs(metricas, curvas):
    # matriz de confusao
    for nome, r in metricas.items():
        if nome.startswith("_") or "teste" not in r:
            continue
        cm = r["teste"]["confusao"]
        with open(os.path.join(cfg.DIR_AVALIACAO, f"confusao_{nome}.csv"), "w") as f:
            w = csv.writer(f)
            w.writerow(["", "previsto_nao", "previsto_sim"])
            w.writerow(["real_nao"] + cm[0])
            w.writerow(["real_sim"] + cm[1])

    # roc pontos
    for nome, dados in curvas.items():
        df = pd.DataFrame({"fpr": dados["roc"]["fpr"], "tpr": dados["roc"]["tpr"]})
        df.to_csv(os.path.join(cfg.DIR_AVALIACAO, f"roc_{nome}.csv"), index=False)
        df = pd.DataFrame({"precision": dados["pr"]["precision"], "recall": dados["pr"]["recall"]})
        df.to_csv(os.path.join(cfg.DIR_AVALIACAO, f"pr_{nome}.csv"), index=False)


def main():
    metricas, curvas = _carregar()
    print("gerando plots...")
    plot_roc(curvas)
    plot_pr(curvas)
    plot_calibracao(curvas)
    plot_matriz_confusao(metricas)
    plot_feature_importance(metricas)
    salvar_csvs(metricas, curvas)
    print("ok")


if __name__ == "__main__":
    main()
