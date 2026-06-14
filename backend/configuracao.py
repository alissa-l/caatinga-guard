import os

UFS_ALVO = ["RN"]

INCLUIR_ESTACOES_VIZINHAS = True
UFS_VIZINHAS = ["PB", "CE"]
RAIO_VIZINHANCA_KM = 50

ANO_INICIO = 2020
ANO_FIM = 2024

SATELITE_REFERENCIA = "AQUA_M-T"

FRAC_TREINO_HISTORICO = 0.8

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_DADOS_BRUTOS = os.path.join(RAIZ, "dados", "brutos")
DIR_DADOS_PROCESSADOS = os.path.join(RAIZ, "dados", "processados")
CAMINHO_BANCO = os.path.join(RAIZ, "dados", "banco.sqlite")
DIR_ARTEFATOS = os.path.join(RAIZ, "backend", "modelo", "artefatos")
DIR_AVALIACAO = os.path.join(RAIZ, "backend", "modelo", "avaliacao")

URL_INMET = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{ano}.zip"
URL_INPE_ANUAL = (
    "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/anual/"
    "EstadosBr_sat_ref/{uf}/focos_br_{uf_lower}_ref_{ano}.zip"
)
URL_INPE_MENSAL = (
    "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/mensal/"
    "Brasil/focos_mensal_br_{aaaamm}.csv"
)
URL_IBGE_MUNICIPIOS = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/"
    "malhas_municipais/municipio_2024/UFs/{uf}/{uf}_Municipios_2024.zip"
)
URL_OSM_NORDESTE = (
    "https://download.geofabrik.de/south-america/brazil/nordeste-latest.osm.pbf"
)
URL_IBGE_BIOMAS = (
    "https://geoftp.ibge.gov.br/informacoes_ambientais/estudos_ambientais/"
    "biomas/vetores/Biomas_250mil.zip"
)


def garantir_diretorios():
    for d in [DIR_DADOS_BRUTOS, DIR_DADOS_PROCESSADOS, DIR_ARTEFATOS, DIR_AVALIACAO]:
        os.makedirs(d, exist_ok=True)
