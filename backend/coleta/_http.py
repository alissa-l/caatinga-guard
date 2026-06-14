# Helpers de download HTTP com retry exponencial. A coleta depende de
# servidores publicos (INMET, INPE, IBGE, Geofabrik) que oscilam em latencia
# e as vezes devolvem 5xx transitorio. Sem retry, um make tudo no meio da
# avaliacao pode falhar feio.

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def sessao_com_retry(total=5, backoff=1.5, statuses=(500, 502, 503, 504, 429)):
    """Cria uma session com retry para metodo GET. Backoff exponencial:
    1.5s, 3s, 6s, 12s, 24s entre tentativas."""
    s = requests.Session()
    politica = Retry(
        total=total,
        backoff_factor=backoff,
        status_forcelist=statuses,
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=politica)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s
