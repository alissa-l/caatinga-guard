PY=python3

.PHONY: instalar baixar baixar-osm baixar-bioma processar treinar testar servir frontend tudo limpar buscar-hp experimento-balanceamento

instalar:
	$(PY) -m pip install -r backend/requirements.txt

baixar:
	$(PY) -m backend.coleta.baixar_ibge
	$(PY) -m backend.coleta.baixar_bioma
	$(PY) -m backend.coleta.baixar_dbqueimadas
	$(PY) -m backend.coleta.baixar_inmet

baixar-osm:
	$(PY) -m backend.coleta.baixar_osm

baixar-bioma:
	$(PY) -m backend.coleta.baixar_bioma

processar-osm:
	$(PY) -m backend.tratamento.processar_osm

processar:
	$(PY) -m backend.tratamento.montar_dataset

buscar-hp:
	$(PY) -m backend.modelo.buscar_hiperparams

experimento-balanceamento:
	$(PY) -m backend.modelo.experimento_balanceamento

treinar:
	$(PY) -m backend.modelo.treinar

servir:
	uvicorn backend.api.main:app --reload --port 8000

testar:
	$(PY) -m pytest tests/ -v

frontend:
	cd frontend && npm run dev

tudo: baixar processar treinar

limpar:
	rm -rf dados/processados/* dados/banco.sqlite
	rm -rf backend/modelo/artefatos/* backend/modelo/avaliacao/*
