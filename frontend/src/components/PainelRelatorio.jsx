import React, { useEffect, useState } from "react";
import { api } from "../servicos/api";
import {
  CartesianGrid, XAxis, YAxis, Tooltip, Legend,
  BarChart, Bar, LineChart, Line, ResponsiveContainer,
} from "recharts";


const NOMES_FEATURE = {
  doy_sin: "dia do ano (seno)",
  doy_cos: "dia do ano (cosseno)",
  mes_sin: "mês (seno)",
  mes_cos: "mês (cosseno)",
  distancia_litoral_km: "distância do litoral (km)",
  area_km2: "área do município (km²)",
  centro_lat: "latitude do centroide",
  centro_lon: "longitude do centroide",
  chuva_acum_30d: "chuva acumulada 30 dias",
  chuva_acum_7d: "chuva acumulada 7 dias",
  dias_sem_chuva: "dias seguidos sem chuva",
  focos_acum_30d: "focos do município nos últimos 30 dias",
  focos_acum_90d: "focos do município nos últimos 90 dias",
  taxa_historica_municipio: "taxa histórica de focos do município",
  taxa_historica_municipio_mes: "taxa histórica por município × mês",
  temp_media: "temperatura média",
  temp_max: "temperatura máxima",
  temp_min: "temperatura mínima",
  umid_media: "umidade média",
  vento_medio: "vento médio",
  rad_media: "radiação média",
  chuva_dia: "chuva no dia",
  fwi: "FWI (índice canadense)",
  ffmc: "FFMC (umidade do combustível fino)",
  dmc: "DMC (umidade da camada intermediária)",
  dc: "DC (índice de seca de longo prazo)",
  isi: "ISI (índice de propagação)",
  bui: "BUI (combustível disponível)",
  n_focos: "focos no dia D",
  focos_lag_1: "focos no dia D-1",
  focos_lag_3: "focos no dia D-3",
  focos_lag_7: "focos no dia D-7",
  bioma_caatinga: "bioma caatinga",
  bioma_mata_atlantica: "bioma mata atlântica",
};


function StatusBadge({ ok, alvo, valor, sufixo = "" }) {
  // ok=verdadeiro: verde; 
  // ok=false: vermelho; 
  // ok="quase": amarelo
  const cor = ok === true ? "#166534" : ok === "quase" ? "#a16207" : "#991b1b";
  const bg = ok === true ? "#dcfce7" : ok === "quase" ? "#fef3c7" : "#fee2e2";
  
  return (
    <span style={{
      display: "inline-block", padding: "2px 10px", background: bg, color: cor,
      borderRadius: 999, fontSize: 11, fontWeight: 600, marginLeft: 8,
    }}>
      {ok === true ? "ok" : ok === "quase" ? "atenção" : "ruim"}
    </span>
  );
}


function CartaoMetrica({ rotulo, valor, descricao, status }) {
  return (
    <div className="cartao" style={{ padding: "12px 14px" }}>
      <div className="rotulo">
        {rotulo}
        {status && <StatusBadge ok={status} />}
      </div>
      <div className="valor" style={{ marginTop: 4 }}>{valor}</div>
      {descricao && (
        <div style={{ fontSize: 11, color: "var(--cor-texto-muted)", marginTop: 6, lineHeight: 1.5 }}>
          {descricao}
        </div>
      )}
    </div>
  );
}


export default function PainelRelatorio() {
  const [rel, setRel] = useState(null);
  const [curvas, setCurvas] = useState(null);
  const [aba, setAba] = useState("saude");

  useEffect(() => {
    api.relatorio().then(setRel);
    api.curvas().then(setCurvas);
  }, []);

  if (!rel || !curvas) return <div className="spinner">carregando relatório...</div>;

  const principal = rel["random_forest"];
  const amig = principal?.amigaveis;
  const auc = principal?.teste?.auc;

  const statusAUC = auc >= 0.75 ? true : auc >= 0.7 ? "quase" : false;
  const statusRecall = amig && amig.recall_top10 >= 0.20 ? true : amig?.recall_top10 >= 0.12 ? "quase" : false;
  const statusLift = amig && amig.lift_top10 >= 3 ? true : amig?.lift_top10 >= 2 ? "quase" : false;
  const statusDiasHit = amig && (amig.dias_com_hit_top10 / amig.n_dias_com_focos) >= 0.5
    ? true
    : amig && (amig.dias_com_hit_top10 / amig.n_dias_com_focos) >= 0.3 ? "quase" : false;
  const statusCalib = amig && Math.abs(amig.taxa_prevista_media - amig.taxa_real) / Math.max(amig.taxa_real, 1e-6) <= 0.3
    ? true : "quase";

  const curvaRecall = curvas.random_forest?.recall_por_k || [];
  const curvaCalib = curvas.random_forest?.calibracao_bins || [];

  const importancias = principal?.importancias ?
    Object.entries(principal.importancias)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 15)
      .map(([nome, v]) => ({ nome: NOMES_FEATURE[nome] || nome, valor: v }))
    : [];

  const modelos = ["random_forest", "lightgbm"].filter(m => rel[m]?.amigaveis);
  const linhasComparacao = modelos.map(m => ({
    nome: m === "random_forest" ? "Random Forest" : "LightGBM",
    chave: m,
    auc: rel[m].teste.auc,
    recall10: rel[m].amigaveis.recall_top10,
    lift10: rel[m].amigaveis.lift_top10,
    dias_hit_pct: rel[m].amigaveis.dias_com_hit_top10 / Math.max(1, rel[m].amigaveis.n_dias_com_focos),
  }));

  return (
    <>
      <div className="painel-header">
        <h1>Saúde do modelo</h1>
        <div className="descricao">
          Métricas medidas no conjunto de teste (último semestre de 2024). Em vez de focar em
          números técnicos, mostramos primeiro o que importa operacionalmente: o modelo pega
          os focos? Em quantos dias acerta?
        </div>
      </div>

      <div className="tabs-rel">
        {
            [
                { 
                    k: "saude", 
                    l: "visão geral" 
                },
                { 
                    k: "ranking", 
                    l: "ranking por K" 
                },
                { 
                    k: "calibracao", 
                    l: "calibração" 
                },
                { 
                    k: "features", 
                    l: "features" 
                },
                { 
                    k: "tecnico", 
                    l: "detalhes técnicos" 
                },
            ].map(t => (
            <button 
                key={t.k} 
                className={`tab ${aba === t.k ? "ativo" : ""}`} 
                onClick={() => setAba(t.k)}>
                {t.l}
            </button>
            ))
        }
      </div>

      {
        aba === "saude" && amig && (
            <>
            <div className="bloco">
                <h2>cobertura — o modelo pega os focos?</h2>
                <div className="cartoes" style={{ gap: 8 }}>
                <CartaoMetrica
                    rotulo="dias com hit no top-10"
                    valor={`${amig.dias_com_hit_top10} de ${amig.n_dias_com_focos}`}
                    status={statusDiasHit}
                    descricao={`em ${amig.dias_com_hit_top10} dos ${amig.n_dias_com_focos} dias do teste que tiveram focos, ao menos um estava entre os 10 municípios mais arriscados apontados pelo modelo. (${(amig.dias_com_hit_top10 / amig.n_dias_com_focos * 100).toFixed(0)}%)`}
                />
                <CartaoMetrica
                    rotulo="recall no top-10"
                    valor={`${(amig.recall_top10 * 100).toFixed(1)}%`}
                    status={statusRecall}
                    descricao={`de todos os ${amig.total_focos} focos reais no teste, ${(amig.recall_top10 * 100).toFixed(1)}% foram capturados pelo top-10 do dia.`}
                />
                <CartaoMetrica
                    rotulo="lift do top-10"
                    valor={`${amig.lift_top10.toFixed(1)}×`}
                    status={statusLift}
                    descricao={`o top-10 do modelo é ${amig.lift_top10.toFixed(1)}× mais eficiente que sortear 10 municípios aleatoriamente.`}
                />
                <CartaoMetrica
                    rotulo="capacidade de ordenar (AUC)"
                    valor={auc.toFixed(3)}
                    status={statusAUC}
                    descricao="probabilidade de o modelo ranquear um par foco-vs-não-foco na ordem correta. 1.0 perfeito, 0.5 acaso."
                />
                </div>
            </div>

            <div className="bloco">
                <h2>calibração — o número faz sentido?</h2>
                <div className="cartoes">
                <CartaoMetrica
                    rotulo="taxa real de focos"
                    valor={`${(amig.taxa_real * 100).toFixed(2)}%`}
                    descricao="frequência real de pares município-dia que tiveram foco em D+1, no teste."
                />
                <CartaoMetrica
                    rotulo="média prevista pelo modelo"
                    valor={`${(amig.taxa_prevista_media * 100).toFixed(2)}%`}
                    status={statusCalib}
                    descricao="média das probabilidades previstas. quanto mais próxima da taxa real, melhor calibrado."
                />
                <CartaoMetrica
                    rotulo="Brier score"
                    valor={amig.brier.toFixed(4)}
                    descricao="erro médio quadrático. mais baixo é melhor. valores próximos de 0 indicam previsões honestas."
                />
                </div>
            </div>

            <div className="aviso">
                <strong>Como ler:</strong> com taxa base de ~1% por dia, métricas binárias clássicas
                (precisão, F1 a 50%) ficam baixas por construção. O que importa é se o modelo, dado um
                dia, sabe apontar os municípios mais arriscados — o que medimos com hit@10, recall@10 e lift.
            </div>
            </>
      )}

      {aba === "ranking" && (
        <>
          <div className="bloco">
            <h2>quanto o ranking cresce conforme N aumenta</h2>
            <div className="dica" style={{ marginBottom: 12 }}>
              eixo X: tamanho do alerta (top-N municípios por dia). eixo Y: dos focos reais, quantos foram capturados.
              A linha pontilhada é o que se esperaria sorteando aleatoriamente.
            </div>
            <div style={{ width: "100%", height: 320 }}>
              <ResponsiveContainer>
                <LineChart data={curvaRecall} margin={{ left: 12, right: 12, top: 10, bottom: 24 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e7e1d4" />
                  <XAxis dataKey="k" stroke="#78716c" fontSize={11}
                    label={{ value: "N (top-N por dia)", position: "insideBottom", offset: -8, fontSize: 11, fill: "#78716c" }}
                  />
                  <YAxis stroke="#78716c" fontSize={11}
                    tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                    label={{ value: "recall", angle: -90, position: "insideLeft", fontSize: 11, fill: "#78716c" }}
                  />
                  <Tooltip
                    contentStyle={{ background: "#fff", border: "1px solid #e7e1d4", borderRadius: 6, fontSize: 12 }}
                    formatter={(v, n) => [`${(v * 100).toFixed(1)}%`, n === "recall" ? "modelo" : "aleatório"]}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line type="monotone" dataKey="recall" stroke="#c2410c" strokeWidth={2.5} dot={false} name="modelo" />
                  <Line type="monotone" dataKey="baseline_aleatorio" stroke="#a8a29e" strokeWidth={1.5} dot={false} strokeDasharray="4 4" name="aleatório" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bloco">
            <h2>comparação entre modelos</h2>
            <table className="tabela">
              <thead>
                <tr>
                  <th>modelo</th>
                  <th>AUC</th>
                  <th>recall@10</th>
                  <th>lift@10</th>
                  <th>dias hit@10</th>
                </tr>
              </thead>
              <tbody>
                {linhasComparacao.map(l => (
                  <tr key={l.chave}>
                    <td>{l.nome}</td>
                    <td>{l.auc.toFixed(3)}</td>
                    <td>{(l.recall10 * 100).toFixed(1)}%</td>
                    <td>{l.lift10.toFixed(2)}×</td>
                    <td>{(l.dias_hit_pct * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {aba === "calibracao" && (
        <div className="bloco">
          <h2>probabilidade prevista vs frequência real</h2>
          <div className="dica" style={{ marginBottom: 10 }}>
            Cada barra é um intervalo de probabilidade prevista. A cor laranja mostra a média prevista;
            a verde, a frequência observada de focos naquele intervalo. Quando elas batem, a calibração está ok.
          </div>
          <table className="tabela">
            <thead>
              <tr>
                <th>faixa prevista</th>
                <th>n (linhas)</th>
                <th>previsto médio</th>
                <th>real observado</th>
              </tr>
            </thead>
            <tbody>
              {curvaCalib.map((b, i) => {
                const desvio = Math.abs(b.previsto_medio - b.real_medio) / Math.max(b.real_medio, 0.001);
                const bom = desvio < 0.3;
                return (
                  <tr key={i} className={bom ? "acerto" : ""}>
                    <td>{(parseFloat(b.bin.split("-")[0]) * 100).toFixed(1)}%–{(parseFloat(b.bin.split("-")[1]) * 100).toFixed(1)}%</td>
                    <td>{b.n}</td>
                    <td>{(b.previsto_medio * 100).toFixed(2)}%</td>
                    <td>{(b.real_medio * 100).toFixed(2)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="dica" style={{ marginTop: 8 }}>
            linhas em verde indicam faixas com calibração próxima (desvio menor que 30%).
          </div>
        </div>
      )}

      {aba === "features" && (
        <div className="bloco">
          <h2>top 15 features (Random Forest)</h2>
          <div className="dica" style={{ marginBottom: 12 }}>
            quanto maior a barra, mais a feature foi usada nas decisões das árvores.
          </div>
          <div style={{ width: "100%", height: 480 }}>
            <ResponsiveContainer>
              <BarChart layout="vertical" data={importancias} margin={{ left: 130, right: 12, top: 6, bottom: 6 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e7e1d4" />
                <XAxis type="number" stroke="#78716c" fontSize={11} />
                <YAxis dataKey="nome" type="category" width={180} fontSize={11} stroke="#44403c" />
                <Tooltip
                  contentStyle={{ background: "#fff", border: "1px solid #e7e1d4", borderRadius: 6, fontSize: 12 }}
                  formatter={(v) => v.toFixed(4)}
                />
                <Bar dataKey="valor" fill="#c2410c" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {aba === "tecnico" && (
        <div className="bloco">
          <h2>métricas binárias (limiar 0.5)</h2>
          <div className="dica" style={{ marginBottom: 8 }}>
            após a calibração, quase nada passa de 50% de probabilidade — então estas métricas ficam
            baixas <em>por construção</em>. Servem como referência para um caso de uso binário, mas
            o modelo é melhor avaliado pelo ranking.
          </div>
          <table className="tabela">
            <thead>
              <tr><th>modelo</th><th>precisão</th><th>recall</th><th>F1</th><th>AP</th></tr>
            </thead>
            <tbody>
              {Object.keys(rel).filter(k => !k.startsWith("_")).map(k => (
                <tr key={k}>
                  <td>{k}</td>
                  <td>{rel[k].teste.precision.toFixed(3)}</td>
                  <td>{rel[k].teste.recall.toFixed(3)}</td>
                  <td>{rel[k].teste.f1.toFixed(3)}</td>
                  <td>{rel[k].teste.ap?.toFixed(3) || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h2 style={{ marginTop: 16 }}>matriz de confusão (Random Forest, limiar 0.5)</h2>
          {(() => {
            const c = rel.random_forest.teste.confusao;
            return (
              <table className="tabela">
                <thead>
                    <tr>
                        <th></th>
                        <th>previsto não</th>
                        <th>previsto sim</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>real não</td>
                        <td>{c[0][0]}</td>
                        <td>{c[0][1]}</td>
                    </tr>
                    <tr>
                        <td>real sim</td>
                        <td>{c[1][0]}</td>
                        <td>{c[1][1]}</td>
                    </tr>
                </tbody>
              </table>
            );
          })()}
        </div>
      )}
    </>
  );
}
