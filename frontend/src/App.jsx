import React, { useEffect, useState, useMemo } from "react";
import { api } from "./servicos/api";
import Mapa from "./componentes/Mapa.jsx";
import PainelPrevisao from "./componentes/PainelPrevisao.jsx";
import PainelComparacao from "./componentes/PainelComparacao.jsx";
import PainelFuturo from "./componentes/PainelFuturo.jsx";
import PainelRelatorio from "./componentes/PainelRelatorio.jsx";
import PainelSobre from "./componentes/PainelSobre.jsx";

const ABAS = ["previsão", "comparação", "futuro", "relatório", "sobre"];

export default function App() {
  const [aba, setAba] = useState("previsão");

  const [geojson, setGeojson] = useState(null);
  const [municipios, setMunicipios] = useState([]);
  const [datasDisp, setDatasDisp] = useState([]);
  const [data, setData] = useState("");

  const [previsao, setPrevisao] = useState(null);
  const [comparacao, setComparacao] = useState(null);
  const [topNCods, setTopNCods] = useState(null);
  const [codSel, setCodSel] = useState(null);
  const [nomeSel, setNomeSel] = useState(null);

  const [diasFut, setDiasFut] = useState(3);
  const [dataBaseFut, setDataBaseFut] = useState("");
  const [futuro, setFuturo] = useState(null);
  const [diaSel, setDiaSel] = useState(0);

  useEffect(() => {
    api.geojson().then(setGeojson);
    api.municipios().then(setMunicipios);
    api.datas().then((r) => {
      setDatasDisp(r.datas);
      const ultima = r.datas[r.datas.length - 1];
      setData(ultima);
      setDataBaseFut(ultima);
    });
  }, []);

  useEffect(() => {
    if (!data) {
      return;
    }

    if (aba === "previsão") {
      api.previsao(data).then(setPrevisao).catch(() => setPrevisao(null));
    } 
    else if (aba === "comparação") {
      api.comparacao(data).then(setComparacao).catch(() => setComparacao(null));
    }
  }, [data, aba]);

  useEffect(() => {
    if (aba !== "futuro" || !dataBaseFut) {
      return;
    }

    api.futuro(diasFut, dataBaseFut).then((r) => { setFuturo(r); setDiaSel(0); });
  }, [aba, diasFut, dataBaseFut]);

  const mapaProb = useMemo(() => {
    if (aba === "previsão" && previsao) {
      return Object.fromEntries(previsao.previsoes.map(p => [p.codigo_ibge, p.probabilidade]));
    }

    if (aba === "futuro" && futuro && futuro.dias[diaSel]) {
      return Object.fromEntries(futuro.dias[diaSel].previsoes.map(p => [p.codigo_ibge, p.probabilidade]));
    }

    return null;
  }, [aba, previsao, futuro, diaSel]);

  const dadosLinhasPrev = useMemo(() => {
    if (!previsao) {
      return null;
    }

    return Object.fromEntries(previsao.previsoes.map(p => [p.codigo_ibge, p]));
  }, [previsao]);

  const dadosLinhasComp = useMemo(() => {
    if (!comparacao) {
      return null;
    }

    return Object.fromEntries(comparacao.previsoes.map(p => [p.codigo_ibge, p]));
  }, [comparacao]);

  const onClicarMun = (cod, nome) => { setCodSel(cod); setNomeSel(nome); };

  const modoInfo = aba === "sobre" || aba === "relatório";

  const usaMapa = aba === "previsão" || aba === "comparação" || aba === "futuro";

  return (
    <div className="layout">
      <div className="header">
        <div className="marca">
          <span className="titulo">Previsor de Incêndios</span>
          <span className="subtitulo">Rio Grande do Norte — risco de foco em D+1</span>
        </div>
        <nav className="nav">
          {ABAS.map(a => (
            <button key={a} className={`aba ${aba === a ? "ativo" : ""}`} onClick={() => setAba(a)}>{a}</button>
          ))}
        </nav>
      </div>

      <div className={`corpo ${modoInfo ? "modo-info" : ""}`}>
        {usaMapa && (
          <div className="mapa-wrap">
            <Mapa
              geojson={geojson}
              mapaProb={aba === "comparação" ? null : mapaProb}
              dadosComparacao={aba === "comparação" ? dadosLinhasComp : null}
              topNCods={aba === "comparação" ? topNCods : null}
              onClicarMun={onClicarMun}
            />
          </div>
        )}

        {usaMapa && (
          <div className="painel">
            {aba === "previsão" && (
              <PainelPrevisao
                data={data} setData={setData} datasDisp={datasDisp}
                mapaProb={mapaProb}
                dadosLinhas={dadosLinhasPrev}
                codSelecionado={codSel}
                nomeSelecionado={nomeSel}
              />
            )}
            {aba === "comparação" && (
              <PainelComparacao
                data={data} setData={setData} datasDisp={datasDisp}
                dadosLinhas={dadosLinhasComp}
                municipios={municipios}
                onTopNChange={setTopNCods}
              />
            )}
            {aba === "futuro" && (
              <PainelFuturo
                dias={diasFut} setDias={setDiasFut}
                dataBase={dataBaseFut} setDataBase={setDataBaseFut}
                datasDisp={datasDisp}
                futuro={futuro}
                diaSelecionado={diaSel}
                setDiaSelecionado={setDiaSel}
              />
            )}
          </div>
        )}

        {aba === "relatório" && (
          <div className="conteudo-info">
            <PainelRelatorio />
          </div>
        )}

        {aba === "sobre" && <PainelSobre />}
      </div>
    </div>
  );
}
