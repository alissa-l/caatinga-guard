import React, { useMemo, useState, useEffect } from "react";

export default function PainelComparacao({ data, setData, datasDisp, dadosLinhas, onTopNChange, municipios }) {
  const [N, setN] = useState(10);

  const nomePor = useMemo(() => {
    const m = {};
    (municipios || []).forEach((mu) => { m[mu.codigo_ibge] = mu.nome; });
    return m;
  }, [municipios]);

  const topN = useMemo(() => {
    if (!dadosLinhas) {
        return [];
    }

    return Object.values(dadosLinhas)
      .sort((a, b) => b.probabilidade - a.probabilidade)
      .slice(0, N);
  }, [dadosLinhas, N]);

  useEffect(() => {
    if (onTopNChange) {
        onTopNChange(new Set(topN.map((t) => t.codigo_ibge)));
    }
    
  }, [topN, onTopNChange]);

  if (!dadosLinhas) {
    return <div className="spinner">carregando...</div>;
  }

  const focosReais = Object.values(dadosLinhas).filter((d) => d.teve_foco === 1).length;
  const acertosNoTopN = topN.filter((t) => t.teve_foco === 1).length;
  const recallTopN = focosReais > 0 ? acertosNoTopN / focosReais : 0;

  return (
    <>
      <div className="painel-header">
        <h1>Comparação histórica</h1>
        <div className="descricao">
          Para uma data passada, compare a previsão do modelo com o que realmente aconteceu em D+1.
          Em vez de um limiar binário (que produziria muitos falsos positivos por causa do desbalanceamento),
          mostramos os <b>N municípios apontados como mais arriscados</b> e marcamos quais tiveram foco.
        </div>
      </div>

      <div className="bloco">
        <h2>parâmetros</h2>
        <div className="campo">
          <label>data base (D)</label>
          <select value={data} onChange={(e) => setData(e.target.value)}>
            {datasDisp.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
          <div className="dica">o real comparado é o de D+1.</div>
        </div>
        <div className="campo">
          <label>tamanho do ranking (N)</label>
          <input type="number" min="3" max="30" value={N}
            onChange={(e) => setN(Math.max(3, Math.min(30, +e.target.value)))} />
          <div className="dica">quantos municípios apontar como "em alerta". 3–30.</div>
        </div>
      </div>

      <div className="bloco">
        <h2>resumo do dia</h2>
        <div className="cartoes">
          <div className="cartao">
            <div className="rotulo">focos reais (D+1)</div>
            <div className="valor">{focosReais}</div>
          </div>
          <div className="cartao">
            <div className="rotulo">acertos no top-{N}</div>
            <div className="valor destaque">{acertosNoTopN}</div>
          </div>
          <div className="cartao">
            <div className="rotulo">precisão@{N}</div>
            <div className="valor">{(acertosNoTopN / N * 100).toFixed(1)}%</div>
          </div>
          <div className="cartao">
            <div className="rotulo">recall@{N}</div>
            <div className="valor">{(recallTopN * 100).toFixed(1)}%</div>
          </div>
        </div>
        <div className="dica" style={{ marginTop: 8 }}>
          <b>precisão@N</b>: dos N municípios alertados, fração que teve foco.
          <b> recall@N</b>: dos focos reais, fração que estava no top-N.
        </div>
      </div>

      <div className="bloco">
        <h2>top {N} municípios mais arriscados</h2>
        <table className="tabela">
          <thead>
            <tr>
              <th>#</th>
              <th>município</th>
              <th>risco</th>
              <th>foco</th>
            </tr>
          </thead>
          <tbody>
            {topN.map((t, i) => (
              <tr key={t.codigo_ibge} className={t.teve_foco === 1 ? "acerto" : ""}>
                <td>{i + 1}</td>
                <td>{nomePor[t.codigo_ibge] || t.codigo_ibge}</td>
                <td>{(t.probabilidade * 100).toFixed(2)}%</td>
                <td>{t.teve_foco === 1 ? "sim" : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {focosReais > 0 && acertosNoTopN < focosReais && (
        <div className="aviso">
          <strong>{focosReais - acertosNoTopN} foco(s) escaparam do top-{N}.</strong> Eles aparecem
          em vermelho claro no mapa — clique para inspecionar quais municípios eram.
        </div>
      )}
    </>
  );
}
