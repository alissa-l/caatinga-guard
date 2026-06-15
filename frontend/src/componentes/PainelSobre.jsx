import React from "react";

// Lista de discentes. Substituir os placeholders pelos nomes reais.
const DISCENTES = [
  { nome: "Alissa de Lima Araújo"},
  { nome: "Pedro Lucas de Souza Martins" },
  { nome: "Nicholas Reyel Lima e Silva" },
];

const TECNOLOGIAS = [
  {
    categoria: "Coleta e ETL",
    itens: ["Python 3.13", "pandas", "geopandas", "requests"],
  },
  {
    categoria: "Modelagem",
    itens: ["scikit-learn (Random Forest, IsotonicRegression)", "LightGBM", "joblib"],
  },
  {
    categoria: "API",
    itens: ["FastAPI", "Pydantic", "uvicorn"],
  },
  {
    categoria: "Frontend",
    itens: ["React 18", "Vite", "Leaflet + react-leaflet", "Recharts"],
  },
  {
    categoria: "Fontes de dados",
    itens: ["INMET (meteorologia)", "INPE / Programa Queimadas (focos)", "IBGE (malhas municipais)"],
  },
];

const GLOSSARIO = [
  { 
    sigla: "FWI", 
    nome: "Fire Weather Index", 
    desc: "Índice canadense de risco de incêndio que combina meteorologia em um número único. Quanto maior, maior o risco." 
  },
  { 
    sigla: "FFMC", 
    nome: "Fine Fuel Moisture Code", 
    desc: "Umidade do combustível fino (folhas secas, grama) — reage rápido à mudança de temperatura e chuva." 
  },
  { 
    sigla: "DMC", 
    nome: "Duff Moisture Code", 
    desc: "Umidade da camada intermediária do solo (matéria orgânica). Mais lento que o FFMC." 
  },
  { 
    sigla: "DC", 
    nome: "Drought Code", 
    desc: "Índice de seca de longo prazo. Mede acúmulo de déficit hídrico em camadas profundas do solo." 
  },
  { 
    sigla: "ISI", 
    nome: "Initial Spread Index", 
    desc: "Velocidade inicial de propagação esperada do fogo, função do FFMC e do vento." 
  },
  { 
    sigla: "BUI", 
    nome: "Buildup Index", 
    desc: "Combustível disponível para queima, combinação de DMC e DC." 
  },
  { 
    sigla: "IDW", 
    nome: "Inverse Distance Weighting", 
    desc: "Técnica usada para estimar a meteorologia de cada município interpolando das 3 estações INMET mais próximas, ponderadas pelo inverso do quadrado da distância." 
  },
  { 
    sigla: "AUC", 
    nome: "Area Under the ROC Curve", 
    desc: "Capacidade do modelo de ordenar corretamente os pares município-dia em risco. 1.0 é perfeito, 0.5 é acaso." 
  },
  { 
    sigla: "AP", 
    nome: "Average Precision", 
    desc: "Média da precisão ao longo da curva precisão-recall. Sensível a desbalanceamento — mede valor prático com poucas amostras positivas." 
  },
  { 
    sigla: "Recall@N", 
    nome: "Recall no top-N", 
    desc: "Dos focos que realmente ocorreram em D+1, quantos estavam nos N municípios apontados como mais arriscados pelo modelo." },
  { 
    sigla: "Precisão@N", 
    nome: "Precisão no top-N", 
    desc: "Dos N municípios apontados como mais arriscados, quantos efetivamente tiveram foco em D+1." },
  { 
    sigla: "Calibração isotônica", 
    nome: "Calibração de probabilidades", 
    desc: "Ajuste pós-treino que faz a probabilidade prevista refletir a frequência real do evento, corrigindo o efeito do balanceamento de classes durante o treino." 
  },
];

export default function PainelSobre() {
  return (
    <div className="conteudo-info">
      <section className="info-section">
        <h2>Sobre o projeto</h2>
        <p>
          Sistema de previsão de risco de incêndio florestal para os 167 municípios do Rio Grande
          do Norte. Para cada município, o modelo estima a probabilidade de ocorrência de pelo
          menos um foco de calor no dia seguinte (D+1), com base em meteorologia, histórico de
          focos e variáveis geográficas.
        </p>
        <p>
          O projeto é trabalho da disciplina de Introdução à Inteligência Artificial. Toda a
          inteligência foi treinada do zero — sem APIs externas de IA, sem pesos pré-treinados.
          O pipeline coleta dados públicos (INMET, INPE, IBGE), monta features município-dia,
          treina classificadores clássicos (Random Forest e LightGBM) com split temporal e
          calibração isotônica, e serve os resultados via FastAPI para esta interface React.
        </p>
        <p>
          Janela histórica usada: <span className="tag">2020–2024</span>{" "}
          <span className="tag">5 anos</span>{" "}
          <span className="tag">~305 mil pares município-dia</span>{" "}
          <span className="tag">taxa base ~1%</span>
        </p>
      </section>

      <section className="info-section">
        <h2>Discentes</h2>
        <div className="discentes">
          {DISCENTES.map((d, i) => (
            <div key={i} className={`discente ${d.nome.startsWith("[") ? "placeholder" : ""}`}>
              <div className="nome">{d.nome}</div>
              <div className="papel">{d.papel}</div>
            </div>
          ))}
        </div>
        <p style={{ marginTop: 10, fontSize: 12, color: "var(--cor-texto-muted)" }}>
          edite <code>frontend/src/componentes/PainelSobre.jsx</code> para preencher os nomes.
        </p>
      </section>

      <section className="info-section">
        <h2>Tecnologias</h2>
        <div className="cards-tec">
          {TECNOLOGIAS.map((t, i) => (
            <div key={i} className="card-tec">
              <div className="categoria">{t.categoria}</div>
              <div className="lista">
                {t.itens.map((it, j) => <div key={j}>{it}</div>)}
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="info-section">
        <h2>Enquadramento — agente PEAS</h2>
        <p style={{ color: "var(--cor-texto-muted)", fontSize: 13, marginBottom: 10 }}>
          Recorte exigido pela disciplina (Russell & Norvig). O sistema é um <i>agente
          de informação</i> que ordena municípios por risco e devolve a previsão como ação.
        </p>
        <div className="peas-grid">
          <div className="peas-item">
            <div className="peas-rotulo">Performance</div>
            <div className="peas-texto">recall e lift no top-N de municípios; calibração medida por Brier score; AUC e AP em conjunto temporal de teste fechado.</div>
          </div>
          <div className="peas-item">
            <div className="peas-rotulo">Environment</div>
            <div className="peas-texto">cada par município-dia do Rio Grande do Norte (167 × ~1830 dias), descrito por meteorologia, FWI canadense e histórico de focos.</div>
          </div>
          <div className="peas-item">
            <div className="peas-rotulo">Actuators</div>
            <div className="peas-texto">endpoint REST que devolve ranking de risco para D+1 e horizonte até D+7, com simulação contrafactual de cenários.</div>
          </div>
          <div className="peas-item">
            <div className="peas-rotulo">Sensors</div>
            <div className="peas-texto">INMET (temperatura, umidade, chuva, vento), INPE/Programa Queimadas (focos passados), IBGE (geometria e centróides municipais).</div>
          </div>
        </div>
      </section>

      <section className="info-section">
        <h2>Como usar a interface</h2>
        <ul>
          <li><b>Previsão</b> — escolha uma data D e veja o risco de foco no dia seguinte para cada município. Clique num município no mapa para detalhes e use os sliders para simular condições alternativas (mais quente, mais seco, etc).</li>
          <li><b>Comparação histórica</b> — escolha uma data e veja os N municípios mais arriscados segundo o modelo. A interface marca quais realmente tiveram foco em D+1 e quais o modelo deixou passar.</li>
          <li><b>Futuro</b> — projeta o risco para os próximos 1 a 7 dias usando persistência climática (assume meteorologia constante do último dia). A confiabilidade decai rápido com o horizonte.</li>
          <li><b>Relatório</b> — métricas do modelo em conjunto de teste fechado (último semestre de 2024). Tabela comparativa, matriz de confusão e importância das features.</li>
        </ul>
      </section>

      <section className="info-section">
        <h2>Metodologia</h2>

        <h3 className="met-h3">1. Coleta</h3>
        <p>
          Três fontes públicas, baixadas via script com retry HTTP automático.
          <b> IBGE</b> — malha municipal 2024 (167 polígonos do RN, áreas, centróides).
          <b> INMET</b> — séries horárias das estações automáticas, agregadas para diário
          (temperatura média/máx/mín, umidade média, chuva, vento e radiação). Para reduzir
          o efeito de bordas, são incluídas estações de PB e CE em até 50 km do RN.
          <b> INPE / Programa Queimadas</b> — focos do satélite de referência {" "}
          <span className="tag">AQUA_M-T</span> em arquivos anuais por UF.
        </p>

        <h3 className="met-h3">2. Espacialização da meteorologia</h3>
        <p>
          As estações INMET não cobrem todos os municípios. Para cada centróide municipal,
          identifica-se as 3 estações mais próximas e aplica-se <b>interpolação IDW</b>{" "}
          (Inverse Distance Weighting) com peso ~ 1/d² para gerar uma série diária por
          município. Dias com gap curto na estação ficam preenchidos por <i>forward-fill</i>{" "}
          de até 3 dias antes do IDW.
        </p>

        <h3 className="met-h3">3. Construção de features</h3>
        <p>
          A partir da meteorologia por município, calcula-se o <b>FWI canadense</b> (Van
          Wagner 1987) recursivamente: FFMC e DMC respondem rápido à secagem; DC mede déficit
          hídrico de longo prazo; ISI, BUI e FWI são funções dos anteriores. Soma-se ainda{" "}
          <b>lags</b> e <b>acumuladas</b> de focos por município (1, 3, 7, 30 e 90 dias),{" "}
          <b>chuva acumulada</b> em 7 e 30 dias, <b>dias sem chuva</b>, <b>sazonalidade</b>{" "}
          (mês e dia do ano em seno/cosseno) e <b>mean encoding</b> da taxa histórica do
          município e do município-mês — calculado <i>somente</i> sobre o período de treino
          para evitar vazamento.
        </p>

        <h3 className="met-h3">4. Split temporal</h3>
        <p>
          Nunca <code>train_test_split</code> aleatório. As datas são ordenadas e cortadas
          em <span className="tag">80% treino</span>, <span className="tag">10% validação</span>{" "}
          e <span className="tag">10% teste</span>. O corte do mean encoding casa com o corte
          do treino. Garante que a avaliação é honesta — o teste contém os últimos meses,
          jamais entremeados.
        </p>

        <h3 className="met-h3">5. Modelo</h3>
        <p>
          Dois classificadores binários, com a mesma cadeia de features:{" "}
          <b>Random Forest</b> (200 árvores, max_depth=14, min_samples_leaf=20,{" "}
          <code>class_weight='balanced'</code>) e <b>LightGBM</b> com{" "}
          <code>scale_pos_weight</code>. O balanceamento é necessário porque a taxa base de
          foco no dia seguinte é ~1%; sem ele, o modelo "ganharia" prevendo tudo como zero.
        </p>

        <h3 className="met-h3">6. Calibração isotônica</h3>
        <p>
          O balanceamento durante o treino infla as probabilidades. Aplica-se{" "}
          <b>regressão isotônica</b> sobre as predições no conjunto de validação para que a
          probabilidade prevista volte a refletir a frequência real. Pequena perturbação
          aditiva é adicionada após a calibração para preservar o ranking fino dentro de
          plateaus do isotonic.
        </p>

        <h3 className="met-h3">7. Avaliação</h3>
        <p>
          Com taxa base ~1%, uma matriz de confusão a limiar 0,5 não diz nada. A leitura
          usada aqui é <b>operacional, baseada em ranking</b>:
        </p>
        <ul>
          <li><b>recall@K</b> e <b>lift@K</b> — dos focos reais no dia, quantos cabem no top-K do modelo, e quanto isso supera um sorteio aleatório.</li>
          <li><b>dias com hit no top-K</b> — em quantos dias com focos o modelo capturou pelo menos um nos N municípios apontados.</li>
          <li><b>Brier score</b> e calibração por bin — saúde da probabilidade absoluta.</li>
          <li>AUC e Average Precision como sanidade global, mas não como métrica primária.</li>
        </ul>

        <h3 className="met-h3">8. Testes automáticos</h3>
        <p>
          Antes de cada release, <code>make testar</code> roda 18+ testes pytest cobrindo:
          smoke do artefato, intervalo das probabilidades, determinismo, sensibilidade a
          perturbação, AUC mínima, recall@10 mínimo, lift@10 mínimo, Brier máximo, calibração
          dentro de margem e checks de sanidade do dataset. Se uma mudança no pipeline
          derrubar qualquer métrica abaixo do mínimo operacional, o teste falha com mensagem
          explícita.
        </p>
      </section>

      <section className="info-section">
        <h2>Glossário</h2>
        <p style={{ color: "var(--cor-texto-muted)", fontSize: 13 }}>
          Termos técnicos que aparecem nas outras telas.
        </p>
        <div className="glossario">
          {GLOSSARIO.map((g, i) => (
            <div key={i} className="termo">
              <div className="sigla">{g.sigla}</div>
              <div className="nome">{g.nome}</div>
              <div className="desc">{g.desc}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
