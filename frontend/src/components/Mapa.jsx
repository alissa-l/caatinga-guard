import React, { useEffect } from "react";
import { MapContainer, TileLayer, GeoJSON, useMap } from "react-leaflet";

// Ta sendo calculado pelo problema real
function corPorProba(p) {
  if (p == null || isNaN(p)) {
    return "#d6d3d1";
  }
  if (p < 0.005) {
    return "#15803d";
  }
  if (p < 0.01)  {
    return "#84cc16";
  }
  if (p < 0.02)  {
    return "#eab308";
  }
  if (p < 0.04)  {
    return "#ea580c";
  }
  return "#b91c1c";
}

function corComparacaoTopN(estaNoTopN, real) {
  if (estaNoTopN && real === 1) {
    return "#7c2d12";
  }
  if (estaNoTopN) {
    return "#fbbf24";
  }
  if (real === 1) {
    return "#dc2626";
  }

  return "#d4d4aa";
}

function FitBounds({ geojson }) {
  const map = useMap();
  useEffect(() => {
    if (!geojson || !window.L) {
        return;
    }

    const L = window.L;
    const layer = L.geoJSON(geojson);

    map.fitBounds(layer.getBounds(), { padding: [10, 10] });
  }, [geojson, map]);
  return null;
}

export default function Mapa({ geojson, mapaProb, dadosComparacao, topNCods, onClicarMun }) {
  if (!geojson) return <div className="spinner">carregando mapa...</div>;

  const estilo = (feat) => {
    const cod = feat.properties.codigo_ibge;
    let cor = "#e7e1d4";
    if (dadosComparacao) {
      const d = dadosComparacao[cod];
      if (d) cor = corComparacaoTopN(topNCods && topNCods.has(cod), d.teve_foco);
    } else if (mapaProb) {
      cor = corPorProba(mapaProb[cod]);
    }
    return { fillColor: cor, color: "#78716c", weight: 0.5, fillOpacity: 0.78 };
  };

  const onCadaFeat = (feature, layer) => {
    const cod = feature.properties.codigo_ibge;
    const nome = feature.properties.nome;
    layer.bindTooltip(nome, { sticky: true, className: "leaflet-tip" });
    layer.on("click", () => onClicarMun && onClicarMun(cod, nome));
    layer.on("mouseover", function () { this.setStyle({ weight: 1.5, color: "#1c1917" }); });
    layer.on("mouseout", function () { this.setStyle({ weight: 0.5, color: "#78716c" }); });
  };

  return (
    <div style={{ position: "relative", height: "100%" }}>
      <MapContainer center={[-5.7, -36.7]} zoom={7} style={{ height: "100%", background: "#faf7f2" }}>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <GeoJSON
          key={JSON.stringify(mapaProb ? "p" : "c") + (mapaProb ? Object.keys(mapaProb).length : 0)}
          data={geojson}
          style={estilo}
          onEachFeature={onCadaFeat}
        />
        <FitBounds geojson={geojson} />
      </MapContainer>

      <div className="legenda">
        {dadosComparacao ? (
          <>
            <div className="titulo">comparação histórica</div>
            <div className="item"><span className="barra" style={{ background: "#7c2d12" }}></span> no top-N e teve foco</div>
            <div className="item"><span className="barra" style={{ background: "#fbbf24" }}></span> no top-N, sem foco real</div>
            <div className="item"><span className="barra" style={{ background: "#dc2626" }}></span> teve foco mas ficou fora</div>
            <div className="item"><span className="barra" style={{ background: "#d4d4aa" }}></span> resto</div>
          </>
        ) : (
          <>
            <div className="titulo">risco D+1</div>
            <div className="item"><span className="barra" style={{ background: "#15803d" }}></span> &lt; 0,5% (irrisório)</div>
            <div className="item"><span className="barra" style={{ background: "#84cc16" }}></span> &lt; 1% (baixo)</div>
            <div className="item"><span className="barra" style={{ background: "#eab308" }}></span> &lt; 2% (atenção)</div>
            <div className="item"><span className="barra" style={{ background: "#ea580c" }}></span> &lt; 4% (alto)</div>
            <div className="item"><span className="barra" style={{ background: "#b91c1c" }}></span> ≥ 4% (muito alto)</div>
          </>
        )}
      </div>
    </div>
  );
}
