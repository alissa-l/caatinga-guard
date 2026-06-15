const BASE = "/api";

async function pegar(url) {
  const r = await fetch(BASE + url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

async function postar(url, body) {
  const r = await fetch(BASE + url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

export const api = {

  municipios: () => pegar("/municipios"),
  geojson: () => pegar("/municipios/geojson"),
  datas: () => pegar("/datas"),

  previsao: (data, modelo = "random_forest") =>
    pegar(`/previsao/${data}?modelo=${modelo}`),

  comparacao: (data, modelo = "random_forest") =>
    pegar(`/previsao/${data}/comparacao?modelo=${modelo}`),

  simulacao: (codigo_ibge, data_base, ajustes) =>
    postar("/previsao/simulacao", { codigo_ibge, data_base, ajustes }),

  futuro: (dias, data_base) =>
    pegar(`/previsao/futuro?dias=${dias}` + (data_base ? `&data_base=${data_base}` : "")),

  relatorio: () => pegar("/modelo/relatorio"),
  curvas: () => pegar("/modelo/curvas"),
};
