import React from "react";

function corPorProba(p) {
  if (p < 0.005) {
    return "#15803d";
  }

  if (p < 0.01) {
    return "#84cc16";
  }
  if (p < 0.02) {
    return "#eab308";
  }

  if (p < 0.04) {
    return "#ea580c";
  }

  return "#b91c1c";
}

export default function PainelFuturo({ dias, setDias, dataBase, setDataBase, datasDisp, futuro, diaSelecionado, setDiaSelecionado }) {
  return (
    <>
      <div className="painel-header">
        <h1>Previsão para os próximos dias</h1>
        <div className="descricao">
          Projeta o risco de incêndio para os próximos 1 a 7 dias usando <b>persistência climática</b>
          {": "}assume que a meteorologia do dia base se mantém constante adiante. A confiabilidade
          decai rápido com o horizonte — é uma simulação grosseira para sinalizar tendência, não um
          forecast meteorológico.
        </div>
      </div>

      <div className="bloco">
        <h2>parâmetros</h2>
        <div className="campo">
          <label>data base</label>
          <select value={dataBase} onChange={(e) => setDataBase(e.target.value)}>
            {datasDisp.slice(-90).map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
          <div className="dica">a meteorologia desse dia é replicada para os dias seguintes.</div>
        </div>
        <div className="campo">
          <label>dias à frente</label>
          <input type="number" min="1" max="7" value={dias}
            onChange={(e) => setDias(Math.max(1, Math.min(7, +e.target.value)))} />
          <div className="dica">de 1 a 7 dias após a data base. quanto maior, menos confiável.</div>
        </div>
      </div>

      <div className="aviso">
        <strong>Aviso de horizonte.</strong> Em D+1 a estimativa ainda é razoável.
        De D+2 em diante o erro cresce, porque o sistema não consome previsão numérica de tempo
        (GFS/ECMWF) — apenas repete o último dia conhecido.
      </div>

      {futuro && (
        <div className="bloco">
          <h2>dias previstos</h2>
          <div className="dica" style={{ marginBottom: 10 }}>
            clique num dia para visualizá-lo no mapa. a barra cinza embaixo de
            cada cartão representa <b>incerteza relativa</b> — cresce com o horizonte
            porque a meteorologia futura é apenas replicada do dia base.
          </div>
          <div className="botoes-dias">
            {
              futuro.dias.map((d, i) => {
                const media = d.previsoes.reduce((s, p) => s + p.probabilidade, 0) / d.previsoes.length;
                const max = Math.max(...d.previsoes.map(p => p.probabilidade));
                const incertezaPct = Math.min(95, 15 + (i + 1) * 12);
                return (
                  <button key={i} onClick={() => setDiaSelecionado(i)}
                    className={`botao-dia ${i === diaSelecionado ? "ativo" : ""}`}>
                    <span className="data">D+{i + 1} — {d.data_alvo}</span>
                    <span className="stat">média <span className="num" style={{ color: corPorProba(media) }}>{(media * 100).toFixed(2)}%</span></span>
                    <span className="stat">máx <span className="num" style={{ color: corPorProba(max) }}>{(max * 100).toFixed(2)}%</span></span>
                    <span className="barra-incerteza" title={`incerteza estimada: ${incertezaPct}%`}>
                      <span className="preenchimento" style={{ width: `${incertezaPct}%` }} />
                      <span className="rotulo-incerteza">incerteza ~{incertezaPct}%</span>
                    </span>
                  </button>
                );
              }
            )}
          </div>
        </div>
      )}
    </>
  );
}
