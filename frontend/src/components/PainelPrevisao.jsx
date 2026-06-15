import React, { useState, useEffect, useMemo } from "react";
import { api } from "../servicos/api";

export default function PainelPrevisao({ data, setData, datasDisp, mapaProb, dadosLinhas, codSelecionado, nomeSelecionado }) {
  const [aj, setAj] = useState(
    { 
      temperatura: 0, 
      umidade: 0, 
      precipitacao: 1, 
      vento: 1 
    }
  );
  const [sim, setSim] = useState(null);
  const [carregando, setCarregando] = useState(false);

  useEffect(() => {
    setSim(null);
    setAj({ temperatura: 0, umidade: 0, precipitacao: 1, vento: 1 });
  }, [codSelecionado, data]);

  const detalhe = codSelecionado && dadosLinhas ? dadosLinhas[codSelecionado] : null;

  const stats = useMemo(() => {return null;
    const vals = Object.values(mapaProb);
    
    return {
      media: vals.reduce((a, b) => a + b, 0) / vals.length,
      max: Math.max(...vals),
      num_alto: vals.filter((v) => v >= 0.02).length,
    };
  }, [mapaProb]);

  const simular = async () => {
    if (!codSelecionado) {
        return;
    }
    
    setCarregando(true);
    try {
      const r = await api.simulacao(codSelecionado, data, aj);
      setSim(r);
    } finally {
      setCarregando(false);
    }
  };

  return (
    <>
      <div className="painel-header">
        <h1>Previsão para D+1</h1>
        <div className="descricao">
          Para a data escolhida (D), o modelo estima a probabilidade de cada município do RN
          ter pelo menos um foco no dia seguinte (D+1). Clique num município no mapa para detalhes
          e simulação.
        </div>
      </div>

      <div className="bloco">
        <h2>data base</h2>
        <div className="campo">
          <label>D (dia analisado)</label>
          <select value={data} onChange={(e) => setData(e.target.value)}>
            {datasDisp.slice(-365).map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
          <div className="dica">o modelo prevê D+1 a partir das condições de D.</div>
        </div>
        {stats && (
          <div className="cartoes" style={{ marginTop: 10 }}>
            <div className="cartao">
              <div className="rotulo">risco médio</div>
              <div className="valor">{(stats.media * 100).toFixed(2)}%</div>
            </div>
            <div className="cartao">
              <div className="rotulo">risco máximo</div>
              <div className="valor destaque">{(stats.max * 100).toFixed(2)}%</div>
            </div>
          </div>
        )}
        <div className="dica" style={{ marginTop: 8 }}>
          taxa base do RN é ~1% por dia. valores acima de 2% já são notáveis; acima de 4% são raros.
        </div>
      </div>

      <div className="bloco">
        <h2>município selecionado</h2>
        {!codSelecionado && <div className="dica">clique num município no mapa para inspecionar.</div>}
        {detalhe && (
          <div className="detalhes-mun">
            <div className="linha"><span className="rotulo">nome</span><span className="valor">{nomeSelecionado}</span></div>
            <div className="linha"><span className="rotulo">código IBGE</span><span className="valor mono">{codSelecionado}</span></div>
            <div className="linha"><span className="rotulo">risco previsto D+1</span><span className="valor destaque">{(detalhe.probabilidade * 100).toFixed(2)}%</span></div>
          </div>
        )}
      </div>

      {codSelecionado && (
        <div className="bloco sliders">
          <h2>simulação de cenário</h2>
          <div className="dica" style={{ marginBottom: 10 }}>
            altere as variáveis meteorológicas do dia D e veja como a probabilidade de D+1 muda.
            os deltas são aplicados sobre os valores observados em {data} para este município.
          </div>

          <div className="slider-campo">
            <div className="slider-cabeca">
              <span className="nome">temperatura média</span>
              <span className="valor">{aj.temperatura >= 0 ? "+" : ""}{aj.temperatura.toFixed(1)} °C</span>
            </div>
            <input type="range" min="-5" max="5" step="0.5" value={aj.temperatura}
              onChange={(e) => setAj({ ...aj, temperatura: parseFloat(e.target.value) })} />
            <div className="slider-explica">soma direto sobre a temperatura observada do dia.</div>
          </div>

          <div className="slider-campo">
            <div className="slider-cabeca">
              <span className="nome">umidade relativa</span>
              <span className="valor">{aj.umidade >= 0 ? "+" : ""}{aj.umidade} pontos</span>
            </div>
            <input type="range" min="-20" max="20" step="1" value={aj.umidade}
              onChange={(e) => setAj({ ...aj, umidade: parseFloat(e.target.value) })} />
            <div className="slider-explica">desloca a umidade em pontos percentuais (limitada entre 0 e 100%).</div>
          </div>

          <div className="slider-campo">
            <div className="slider-cabeca">
              <span className="nome">precipitação acumulada</span>
              <span className="valor">×{aj.precipitacao.toFixed(1)}</span>
            </div>
            <input type="range" min="0" max="3" step="0.1" value={aj.precipitacao}
              onChange={(e) => setAj({ ...aj, precipitacao: parseFloat(e.target.value) })} />
            <div className="slider-explica">multiplica a chuva do dia. 0 = sem chuva, 3 = chuva triplicada.</div>
          </div>

          <div className="slider-campo">
            <div className="slider-cabeca">
              <span className="nome">velocidade do vento</span>
              <span className="valor">×{aj.vento.toFixed(1)}</span>
            </div>
            <input type="range" min="0.5" max="2" step="0.1" value={aj.vento}
              onChange={(e) => setAj({ ...aj, vento: parseFloat(e.target.value) })} />
            <div className="slider-explica">multiplica o vento do dia.</div>
          </div>

          <button className="btn" onClick={simular} disabled={carregando} style={{ marginTop: 12 }}>
            {carregando ? "calculando..." : "recalcular probabilidade"}
          </button>

          {sim && (
            <div className="simulacao-resultado">
              <span className="rotulo">probabilidade original</span>
              <span className="valor">{(sim.probabilidade_original * 100).toFixed(2)}%</span>
              <span className="rotulo">probabilidade simulada</span>
              <span className="valor">{(sim.probabilidade_simulada * 100).toFixed(2)}%</span>
              <span className="rotulo">variação</span>
              <span className={sim.probabilidade_simulada > sim.probabilidade_original ? "delta-up" : "delta-down"}>
                {((sim.probabilidade_simulada - sim.probabilidade_original) * 100).toFixed(2)} pontos
              </span>
            </div>
          )}
        </div>
      )}
    </>
  );
}
